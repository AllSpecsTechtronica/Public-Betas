from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtWidgets import (
    QAbstractButton,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .dropdown_pane_stack import DropdownPaneStack

# VS Code-style: panes can be dragged down to a thin strip while remaining visible.
_PANE_MIN_WIDTH = 10
_PANE_MIN_HEIGHT = 28
_TRAY_CARD_MIN_WIDTH = 420
_TRAY_MIN_RENDER_HEIGHT = 180
_TRAY_TARGET_FRACTION = 0.30
_TRAY_MAX_AUTO_HEIGHT = 360
_SETTINGS_PANE_MIN_WIDTH = 320
_SETTINGS_PANE_DEFAULT_WIDTH = 620
_SETTINGS_DIAGNOSTICS_DEFAULT_WIDTH = 760


def _wrap_scroll(page: QWidget, parent: Optional[QWidget] = None) -> QScrollArea:
    sc = QScrollArea(parent)
    sc.setWidgetResizable(True)
    sc.setFrameShape(QScrollArea.Shape.NoFrame)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    sc.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    sc.setMinimumSize(_PANE_MIN_WIDTH, _PANE_MIN_HEIGHT)
    sc.setWidget(page)
    return sc


def _ensure_scroll(page: QWidget, parent: Optional[QWidget] = None) -> QWidget:
    if isinstance(page, QScrollArea):
        return page
    return _wrap_scroll(page, parent)


def _relax_split_child_minimums(widget: QWidget) -> None:
    """Allow splitter/card collapse without content forcing huge minimum widths."""
    widget.setMinimumWidth(_PANE_MIN_WIDTH)
    root_policy = widget.sizePolicy()
    root_policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
    widget.setSizePolicy(root_policy)
    root_layout = widget.layout()
    if isinstance(root_layout, QLayout):
        root_layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)

    for child in widget.findChildren(QWidget):
        if bool(child.property("isTitle")) or child.objectName() == "cvOpsSplitPaneTitle":
            child.setMinimumWidth(0)
            continue
        if isinstance(child, (QAbstractButton, QTabBar, QSplitter)):
            child.setMinimumWidth(0)
            continue
        child.setMinimumWidth(0)
        policy = child.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        child.setSizePolicy(policy)
        layout = child.layout()
        if isinstance(layout, QLayout):
            layout.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)


class _SplitPane(QWidget):
    def __init__(
        self,
        title: str,
        content: QWidget,
        on_close: Callable[["_SplitPane"], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cvOpsSplitPane")
        self._content = content
        self._title = title
        self.setMinimumSize(_PANE_MIN_WIDTH, _PANE_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        _relax_split_child_minimums(content)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QWidget(self)
        header.setObjectName("cvOpsSplitPaneHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 4, 6, 4)
        hl.setSpacing(6)
        label = QLabel(title, header)
        label.setObjectName("cvOpsSplitPaneTitle")
        label.setProperty("isTitle", True)
        hl.addWidget(label, stretch=0)
        hl.addStretch(1)

        close_btn = QToolButton(header)
        close_btn.setObjectName("cvOpsSplitPaneClose")
        close_btn.setText("X")
        close_btn.setToolTip(f"Close {title}")
        close_btn.setAutoRaise(True)
        close_btn.setFixedSize(22, 22)
        close_btn.clicked.connect(lambda _checked=False: on_close(self))
        hl.addWidget(close_btn)
        self._header = header

        root.addWidget(header)
        root.addWidget(content, stretch=1)

    @property
    def content(self) -> QWidget:
        return self._content

    @property
    def title(self) -> str:
        return self._title

    def set_header_visible(self, visible: bool) -> None:
        self._header.setVisible(bool(visible))


class _MainPaneHost(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("cvOpsMainPaneHost")
        # Ignored vertical policy + small explicit minimum so qSmartMinSize does NOT
        # honor a tall center page's content minimum (e.g. the ScrapePanel label
        # canvas stack in Collect mode). Without this, the vertical body splitter
        # cannot shrink this host, so the bottom tray gets pinned to its floor and
        # the divider snaps back when dragged. The layout's SetNoConstraint keeps the
        # layout from re-imposing the content minimum on this host.
        self.setMinimumHeight(_PANE_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self._content: Optional[QWidget] = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

    def set_content(self, widget: Optional[QWidget]) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
        self._content = widget
        if widget is not None:
            widget.setParent(self)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._layout.addWidget(widget, stretch=1)

    def current_content(self) -> Optional[QWidget]:
        return self._content


@dataclass(frozen=True)
class _PaneSpec:
    pane_id: str
    title: str
    widget: QWidget


class _TrayCard(QFrame):
    def __init__(
        self,
        spec: _PaneSpec,
        on_close: Callable[[str], None],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cvOpsBottomTrayCard")
        self.pane_id = spec.pane_id
        self.title = spec.title
        self._content = spec.widget
        self.setMinimumWidth(_TRAY_CARD_MIN_WIDTH)
        self.setMinimumHeight(_PANE_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QWidget(self)
        header.setObjectName("cvOpsBottomTrayCardHeader")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(8, 4, 6, 4)
        hl.setSpacing(6)
        label = QLabel(spec.title, header)
        label.setObjectName("cvOpsSplitPaneTitle")
        label.setProperty("isTitle", True)
        hl.addWidget(label, stretch=0)
        hl.addStretch(1)
        close_btn = QToolButton(header)
        close_btn.setObjectName("cvOpsSplitPaneClose")
        close_btn.setText("X")
        close_btn.setToolTip(f"Hide {spec.title} for this mode")
        close_btn.setAutoRaise(True)
        close_btn.setFixedSize(22, 22)
        close_btn.clicked.connect(lambda _checked=False, pid=spec.pane_id: on_close(pid))
        hl.addWidget(close_btn)
        root.addWidget(header)

        self._body = QWidget(self)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(0)
        spec.widget.setParent(self._body)
        spec.widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._body_layout.addWidget(spec.widget, stretch=1)
        root.addWidget(self._body, stretch=1)

    def detach_content(self) -> Optional[QWidget]:
        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                return widget
        return None


class BottomPaneTray(QWidget):
    def __init__(self, *, on_close: Callable[[str], None], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("cvOpsBottomPaneTray")
        self._on_close = on_close
        self._cards: list[_TrayCard] = []
        self.setMinimumHeight(_PANE_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._scroll = QScrollArea(self)
        self._scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._row_widget = QWidget(self._scroll)
        self._row_widget.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Expanding)
        self._row = QHBoxLayout(self._row_widget)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(6)
        self._scroll.setWidget(self._row_widget)
        root.addWidget(self._scroll, stretch=1)
        self.setVisible(False)

    def clear_panes(self) -> None:
        while self._row.count():
            item = self._row.takeAt(0)
            widget = item.widget()
            if isinstance(widget, _TrayCard):
                widget.detach_content()
                widget.setParent(None)
                widget.deleteLater()
            elif widget is not None:
                widget.setParent(None)
        self._cards.clear()

    def set_panes(self, panes: list[_PaneSpec]) -> None:
        self.clear_panes()
        for spec in panes:
            card = _TrayCard(spec, self._on_close, self._row_widget)
            self._cards.append(card)
            self._row.addWidget(card, stretch=1)
        self._sync_row_minimum_width()
        self.setVisible(bool(panes))

    def pane_ids(self) -> list[str]:
        return [card.pane_id for card in self._cards]

    def pane_titles(self) -> list[str]:
        return [card.title for card in self._cards]

    def _sync_row_minimum_width(self) -> None:
        if not self._cards:
            self._row_widget.setMinimumWidth(0)
            return
        margins = self._row.contentsMargins()
        spacing = max(0, self._row.spacing())
        minimum = margins.left() + margins.right()
        minimum += sum(max(_TRAY_CARD_MIN_WIDTH, card.minimumWidth()) for card in self._cards)
        minimum += spacing * max(0, len(self._cards) - 1)
        self._row_widget.setMinimumWidth(minimum)


@dataclass
class WorkbenchSplitRefs:
    catalog_list: QWidget
    catalog_detail: QWidget
    result_panel: QWidget
    lineage_panel: QWidget
    test_range_page: QWidget
    data_page: QWidget
    viz_page: QWidget
    collect_page: QWidget
    notes_page: QWidget
    settings_page: QWidget
    diagnostics_page: QWidget
    cells_page: QWidget
    three_d_page: QWidget
    notifications_page: QWidget
    portal_page: QWidget
    queue_panel: QWidget
    collect_database_panel: QWidget
    collect_dataset_editor: QWidget
    data_viz_selector: QWidget


class WorkbenchSplitHost(QWidget):
    """Workbench plane: Catalog left, active mode center, mode helpers in bottom tray."""

    MODE_EXPLORE = "explore"
    MODE_TEST = "test"
    MODE_DATA = "data"
    MODE_VIZ = "viz"
    MODE_COLLECT = "collect"
    MODE_NOTES = "notes"
    MODE_SETTINGS = "settings"
    MODE_CELLS = "cells"
    MODE_THREE_D = "three_d"
    MODE_NOTIFICATIONS = "notifications"
    MODE_QUEUE = "queue"
    MODE_PORTAL = "portal"

    PRESET_TRAIN = "train"
    PRESET_EVAL = "eval"
    PRESET_LINEAGE = "lineage"

    def __init__(
        self,
        refs: WorkbenchSplitRefs,
        *,
        on_splitter_moved: Optional[Callable[[], None]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._refs = refs
        self._on_splitter_moved = on_splitter_moved
        self._mode = self.MODE_EXPLORE
        self._preset = self.PRESET_TRAIN
        self._list_requested_visible = True
        self._list_user_closed = False
        self._tray_user_closed = False
        self._tray_saved_sizes: tuple[int, int] | None = None
        self._results_available = self._has_active_result_context(refs.result_panel)
        self._closed_tray: set[str] = set()

        self._detail_widget = _ensure_scroll(refs.catalog_detail, self)
        self._result_widget = _ensure_scroll(refs.result_panel, self)
        self._lineage_widget = _ensure_scroll(refs.lineage_panel, self)
        self._queue_widget = _ensure_scroll(refs.queue_panel, self)
        self._database_widget = _ensure_scroll(refs.collect_database_panel, self)
        self._dataset_editor_widget = _ensure_scroll(refs.collect_dataset_editor, self)
        self._data_viz_selector_widget = _ensure_scroll(refs.data_viz_selector, self)
        self._collect_helper_stack = DropdownPaneStack(self)
        self._collect_helper_stack.addTab(self._database_widget, "Database Catalog")
        self._collect_helper_stack.addTab(self._dataset_editor_widget, "Dataset Editor")
        self._settings_widget = refs.settings_page
        self._settings_widget.setMinimumWidth(_SETTINGS_PANE_MIN_WIDTH)
        self._settings_widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self._diagnostics_widget = refs.diagnostics_page
        self._diagnostics_widget.setMinimumWidth(_PANE_MIN_WIDTH)
        self._diagnostics_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._settings_diag_split = QSplitter(Qt.Orientation.Horizontal, self)
        self._settings_diag_split.setObjectName("cvOpsSettingsDiagSplit")
        self._settings_diag_split.setChildrenCollapsible(False)
        self._settings_diag_split.setHandleWidth(4)
        self._settings_diag_split.splitterMoved.connect(self._emit_split_moved)
        self._settings_diag_split.addWidget(self._settings_widget)
        self._settings_diag_split.addWidget(self._diagnostics_widget)
        self._settings_diag_split.setStretchFactor(0, 0)
        self._settings_diag_split.setStretchFactor(1, 1)
        self._settings_diag_split.setSizes(
            [_SETTINGS_PANE_DEFAULT_WIDTH, _SETTINGS_DIAGNOSTICS_DEFAULT_WIDTH]
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._top_split = QSplitter(Qt.Orientation.Horizontal)
        self._top_split.setChildrenCollapsible(False)
        self._top_split.setHandleWidth(2)
        self._top_split.splitterMoved.connect(self._emit_split_moved)

        list_body = QWidget()
        lw = QVBoxLayout(list_body)
        lw.setContentsMargins(0, 0, 0, 0)
        lw.addWidget(refs.catalog_list)
        self._list_wrap = _SplitPane("Catalog", list_body, self._close_list_pane, self._top_split)
        self._list_wrap.setMinimumWidth(_PANE_MIN_WIDTH)
        self._list_wrap.setMaximumWidth(520)
        self._top_split.addWidget(self._list_wrap)

        self._body_split = QSplitter(Qt.Orientation.Vertical)
        self._body_split.setChildrenCollapsible(False)
        self._body_split.setHandleWidth(3)
        self._body_split.splitterMoved.connect(self._emit_split_moved)
        self._main_host = _MainPaneHost(self._body_split)
        self._bottom_tray = BottomPaneTray(on_close=self._close_tray_pane, parent=self._body_split)
        self._body_split.addWidget(self._main_host)
        self._body_split.addWidget(self._bottom_tray)
        self._body_split.setStretchFactor(0, 7)
        self._body_split.setStretchFactor(1, 3)
        self._body_split.setSizes([700, 300])
        self._top_split.addWidget(self._body_split)
        self._top_split.setStretchFactor(0, 0)
        self._top_split.setStretchFactor(1, 1)
        self._top_split.setSizes([300, 1400])

        root.addWidget(self._top_split, stretch=1)

        result_context_signal = getattr(refs.result_panel, "activeContextChanged", None)
        if result_context_signal is not None:
            try:
                result_context_signal.connect(self.set_results_available)
            except Exception:
                pass

        self.set_mode(self.MODE_EXPLORE)
        self.apply_preset(self.PRESET_TRAIN)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._ensure_tray_size()

    def set_mode(self, mode_id: str) -> None:
        mid = str(mode_id or self.MODE_EXPLORE).strip().lower()
        if mid not in {
            self.MODE_EXPLORE,
            self.MODE_TEST,
            self.MODE_DATA,
            self.MODE_VIZ,
            self.MODE_COLLECT,
            self.MODE_NOTES,
            self.MODE_SETTINGS,
            self.MODE_CELLS,
            self.MODE_THREE_D,
            self.MODE_NOTIFICATIONS,
            self.MODE_QUEUE,
            self.MODE_PORTAL,
        }:
            mid = self.MODE_EXPLORE
        self._mode = mid
        self.set_list_visible(True)
        self._refresh_layout()

    def apply_preset(self, preset_id: str) -> None:
        preset = str(preset_id or self.PRESET_TRAIN).strip().lower() or self.PRESET_TRAIN
        if preset not in {self.PRESET_TRAIN, self.PRESET_EVAL, self.PRESET_LINEAGE}:
            preset = self.PRESET_TRAIN
        self._preset = preset
        self._refresh_layout()

    def set_results_available(self, available: bool) -> None:
        available = bool(available)
        if self._results_available == available:
            return
        self._results_available = available
        self._refresh_layout()

    def set_list_visible(self, visible: bool) -> None:
        self._list_requested_visible = bool(visible)
        self._list_wrap.setVisible(self._list_requested_visible and not self._list_user_closed)

    def is_catalog_visible(self) -> bool:
        return bool(self._list_wrap.isVisible())

    def toggle_catalog(self) -> bool:
        self._list_user_closed = not self._list_user_closed
        self._list_wrap.setVisible(self._list_requested_visible and not self._list_user_closed)
        self._emit_split_moved()
        return self.is_catalog_visible()

    def has_bottom_panes(self) -> bool:
        return bool(self._bottom_tray.pane_ids())

    def is_bottom_pane_visible(self) -> bool:
        return bool(self.has_bottom_panes() and not self._bottom_tray.isHidden())

    def toggle_bottom_pane(self) -> bool:
        if not self.has_bottom_panes():
            self._tray_user_closed = False
            self._bottom_tray.setVisible(False)
            self._emit_split_moved()
            return False
        if self.is_bottom_pane_visible():
            try:
                sizes = self._body_split.sizes()
                if len(sizes) >= 2:
                    self._tray_saved_sizes = (int(sizes[0]), int(sizes[1]))
            except Exception:
                self._tray_saved_sizes = None
            self._tray_user_closed = True
            self._bottom_tray.setVisible(False)
            try:
                total = max(1, sum(self._body_split.sizes()))
                self._body_split.setSizes([total, 0])
            except Exception:
                pass
        else:
            self._tray_user_closed = False
            self._bottom_tray.setVisible(True)
            if self._tray_saved_sizes is not None:
                try:
                    self._body_split.setSizes([self._tray_saved_sizes[0], self._tray_saved_sizes[1]])
                    self._ensure_tray_size()
                except Exception:
                    self._ensure_tray_size()
            else:
                self._ensure_tray_size()
        self._emit_split_moved()
        return self.is_bottom_pane_visible()

    def reopen_all_panes(self) -> None:
        self._list_user_closed = False
        self._tray_user_closed = False
        self._list_wrap.setVisible(self._list_requested_visible)
        context = self._tray_context()
        self._closed_tray = {key for key in self._closed_tray if not key.startswith(f"{context}:")}
        self._refresh_layout()
        self._emit_split_moved()

    def close_tray_pane(self, pane_id: str) -> None:
        self._close_tray_pane(str(pane_id or ""))

    def current_center_widget(self) -> Optional[QWidget]:
        return self._main_host.current_content()

    def tray_pane_ids(self) -> list[str]:
        return self._bottom_tray.pane_ids()

    def tray_pane_titles(self) -> list[str]:
        return self._bottom_tray.pane_titles()

    def save_split_state(self) -> dict[str, bytes]:
        try:
            main_bytes = bytes(self._body_split.saveState())
        except Exception:
            main_bytes = b""
        try:
            top_bytes = bytes(self._top_split.saveState())
        except Exception:
            top_bytes = b""
        try:
            settings_diag_bytes = bytes(self._settings_diag_split.saveState())
        except Exception:
            settings_diag_bytes = b""
        closed = sorted(self._closed_tray)
        if self._list_user_closed:
            closed.append("catalog")
        if self._tray_user_closed:
            closed.append("bottom_tray")
        return {
            "outer": main_bytes,
            "top": top_bytes,
            "settings_diag": settings_diag_bytes,
            # Backward-compatible key name used by existing settings storage.
            "explorer_tri": top_bytes,
            "closed_panes": ",".join(closed).encode("ascii", errors="ignore"),
        }

    def restore_split_state(self, data: dict[str, bytes]) -> None:
        main = data.get("outer")
        top = data.get("top") or data.get("explorer_tri")
        settings_diag = data.get("settings_diag")
        if isinstance(main, (bytes, QByteArray)) and len(main) > 0:
            try:
                self._body_split.restoreState(QByteArray(main))
            except Exception:
                pass
        if isinstance(top, (bytes, QByteArray)) and len(top) > 0:
            try:
                self._top_split.restoreState(QByteArray(top))
            except Exception:
                pass
        if isinstance(settings_diag, (bytes, QByteArray)) and len(settings_diag) > 0:
            try:
                self._settings_diag_split.restoreState(QByteArray(settings_diag))
            except Exception:
                pass
        closed = data.get("closed_panes")
        if isinstance(closed, QByteArray):
            closed_text = bytes(closed).decode("ascii", errors="ignore")
        elif isinstance(closed, bytes):
            closed_text = closed.decode("ascii", errors="ignore")
        else:
            closed_text = ""
        parts = {p.strip() for p in closed_text.split(",") if p.strip()}
        self._list_user_closed = "catalog" in parts
        self._tray_user_closed = "bottom_tray" in parts
        self._closed_tray = {p for p in parts if ":" in p}
        self._list_wrap.setVisible(self._list_requested_visible and not self._list_user_closed)
        self._refresh_layout()

    def _refresh_layout(self) -> None:
        self._main_host.set_content(None)
        self._bottom_tray.clear_panes()
        center = self._center_widget_for_state()
        self._main_host.set_content(center)
        tray = self._tray_specs_for_state(center)
        self._bottom_tray.set_panes(tray)
        if tray and not self._tray_user_closed:
            self._ensure_tray_size()
        self._bottom_tray.setVisible(bool(tray) and not self._tray_user_closed)
        self._emit_split_moved()

    def _center_widget_for_state(self) -> QWidget:
        if self._mode == self.MODE_COLLECT:
            return self._refs.collect_page
        if self._mode == self.MODE_TEST:
            return self._refs.test_range_page
        if self._mode == self.MODE_DATA:
            return self._refs.data_page
        if self._mode == self.MODE_VIZ:
            return self._refs.viz_page
        if self._mode == self.MODE_NOTES:
            return self._refs.notes_page
        if self._mode == self.MODE_SETTINGS:
            return self._settings_diag_split
        if self._mode == self.MODE_CELLS:
            return self._refs.cells_page
        if self._mode == self.MODE_THREE_D:
            return self._refs.three_d_page
        if self._mode == self.MODE_NOTIFICATIONS:
            return self._refs.notifications_page
        if self._mode == self.MODE_QUEUE:
            return self._queue_widget
        if self._mode == self.MODE_PORTAL:
            return self._refs.portal_page
        if self._preset == self.PRESET_LINEAGE:
            return self._lineage_widget
        if self._preset == self.PRESET_EVAL and self._results_available:
            return self._result_widget
        return self._detail_widget

    def _tray_specs_for_state(self, center: QWidget) -> list[_PaneSpec]:
        specs: list[_PaneSpec] = []
        if self._mode == self.MODE_COLLECT:
            specs = [
                _PaneSpec("collect_helpers", "Collect Helpers", self._collect_helper_stack),
            ]
        elif self._mode == self.MODE_TEST:
            specs = [
                _PaneSpec("queue", "Queue", self._queue_widget),
                *_result_spec(self._results_available, self._result_widget),
            ]
        elif self._mode == self.MODE_DATA:
            specs = [
                _PaneSpec("dataset_helpers", "Dataset Helpers", self._collect_helper_stack),
            ]
        elif self._mode == self.MODE_VIZ:
            specs = [_PaneSpec("data_viz_selector", "Data Source Selector", self._data_viz_selector_widget)]
        elif self._mode == self.MODE_QUEUE:
            specs = [
                *_result_spec(self._results_available, self._result_widget),
                _PaneSpec("lineage", "Lineage", self._lineage_widget),
            ]
        elif self._mode == self.MODE_CELLS:
            specs = []
        elif self._mode in {
            self.MODE_NOTES,
            self.MODE_THREE_D,
            self.MODE_NOTIFICATIONS,
            self.MODE_PORTAL,
        }:
            specs = [*_result_spec(self._results_available, self._result_widget)]
        elif self._mode == self.MODE_SETTINGS:
            specs = [
                *_result_spec(self._results_available, self._result_widget),
            ]
        else:
            if self._preset == self.PRESET_LINEAGE:
                specs = [
                    _PaneSpec("scenario_detail", "Scenario Detail", self._detail_widget),
                    *_result_spec(self._results_available, self._result_widget),
                    _PaneSpec("queue", "Queue", self._queue_widget),
                ]
            elif self._preset == self.PRESET_EVAL and self._results_available:
                specs = [
                    _PaneSpec("scenario_detail", "Scenario Detail", self._detail_widget),
                    _PaneSpec("lineage", "Lineage", self._lineage_widget),
                    _PaneSpec("queue", "Queue", self._queue_widget),
                ]
            else:
                specs = [
                    _PaneSpec("lineage", "Lineage", self._lineage_widget),
                    _PaneSpec("queue", "Queue", self._queue_widget),
                ]
        context = self._tray_context()
        out: list[_PaneSpec] = []
        for spec in specs:
            if spec.widget is center:
                continue
            if f"{context}:{spec.pane_id}" in self._closed_tray:
                continue
            out.append(spec)
        return out

    def _tray_context(self) -> str:
        if self._mode == self.MODE_EXPLORE:
            return f"{self.MODE_EXPLORE}:{self._preset}"
        return self._mode

    def _close_tray_pane(self, pane_id: str) -> None:
        if not pane_id:
            return
        self._closed_tray.add(f"{self._tray_context()}:{pane_id}")
        self._refresh_layout()

    def _close_list_pane(self, pane: _SplitPane) -> None:
        self._list_user_closed = True
        pane.setVisible(False)
        self._emit_split_moved()

    def _ensure_tray_size(self) -> None:
        try:
            if not self.has_bottom_panes() or self._tray_user_closed or self._bottom_tray.isHidden():
                return
            sizes = self._body_split.sizes()
            if len(sizes) < 2 or int(sizes[1]) >= _TRAY_MIN_RENDER_HEIGHT:
                return
            total = sum(max(0, int(s)) for s in sizes)
            if total < (_TRAY_MIN_RENDER_HEIGHT + _PANE_MIN_HEIGHT):
                total = max(total, int(self._body_split.height()), int(self.height()))
            if total < (_TRAY_MIN_RENDER_HEIGHT + _PANE_MIN_HEIGHT):
                return
            target = max(_TRAY_MIN_RENDER_HEIGHT, int(total * _TRAY_TARGET_FRACTION))
            target = min(target, _TRAY_MAX_AUTO_HEIGHT, max(_PANE_MIN_HEIGHT, total - _PANE_MIN_HEIGHT))
            if target <= int(sizes[1]):
                return
            self._body_split.setSizes([max(_PANE_MIN_HEIGHT, total - target), target])
        except Exception:
            pass

    def _emit_split_moved(self, *_args: object) -> None:
        if self._on_splitter_moved is not None:
            self._on_splitter_moved()

    @staticmethod
    def _has_active_result_context(result_panel: QWidget) -> bool:
        checker = getattr(result_panel, "has_active_context", None)
        if callable(checker):
            try:
                return bool(checker())
            except Exception:
                return False
        return False


def _result_spec(active: bool, widget: QWidget) -> list[_PaneSpec]:
    if not active:
        return []
    return [_PaneSpec("results", "Results", widget)]
