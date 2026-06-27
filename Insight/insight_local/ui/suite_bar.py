"""SuiteBar — horizontal strip for switching between named grid profiles (suites).

Layout:  [Science Suite] [Coding Suite] [Engineering Suite]  [+ suite]  ......

Signals:
    suite_selected(int)          — user clicked a suite tab
    suite_new_requested()        — user clicked [+ suite]
    suite_rename_requested(int)  — user double-clicked a tab
    suite_delete_requested(int)  — user chose [delete] from right-click menu
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMenu,
    QPushButton,
    QWidget,
)

from .theme import _scheme_rgb, current_color_scheme


def _build_suite_styles() -> tuple[str, str, str]:
    """Return (active, inactive, new) style strings for the current color scheme."""
    if current_color_scheme() == "beacon":
        # [BEACON] Dark navy substrate — bone text, vermillion bottom bar on active.
        vr, vg, vb = _scheme_rgb("accent_dark")    # vermillion
        br, bg_, bb = _scheme_rgb("bone")           # bone text
        g3r, g3g, g3b = _scheme_rgb("graphite_3")  # chip fill
        g4r, g4g, g4b = _scheme_rgb("graphite_4")  # border
        verm = f"{vr},{vg},{vb}"
        bone = f"{br},{bg_},{bb}"
        g3 = f"{g3r},{g3g},{g3b}"
        g4 = f"{g4r},{g4g},{g4b}"
        active = (
            "QPushButton {"
            "  font-size: 10px; padding: 2px 10px;"
            f"  background: rgba({g3},0.55);"
            f"  color: rgba({bone},0.95);"
            f"  border: 1px solid rgba({g4},0.50);"
            f"  border-bottom: 2px solid rgba({verm},1.0);"
            "  font-weight: 600;"
            "}"
        )
        inactive = (
            "QPushButton {"
            "  font-size: 10px; padding: 2px 10px;"
            "  background: transparent;"
            f"  color: rgba({bone},0.45);"
            "  border: none;"
            "}"
            "QPushButton:hover {"
            f"  color: rgba({bone},0.72);"
            f"  background: rgba({g3},0.25);"
            "}"
        )
        new_style = (
            "QPushButton {"
            "  font-size: 10px; padding: 2px 8px;"
            "  background: transparent;"
            f"  color: rgba({verm},0.50);"
            "  border: none;"
            "}"
            "QPushButton:hover {"
            f"  color: rgba({verm},0.85);"
            "}"
        )
        return active, inactive, new_style

    # Default: near-black text on whatever substrate the other themes use.
    active = (
        "QPushButton {"
        "  font-size: 10px; padding: 2px 10px;"
        "  background: rgba(20,8,8,0.14);"
        "  color: rgba(20,8,8,0.88);"
        "  border: 1px solid rgba(20,8,8,0.22);"
        "  border-bottom: 2px solid rgba(20,8,8,0.55);"
        "  font-weight: 600;"
        "}"
    )
    inactive = (
        "QPushButton {"
        "  font-size: 10px; padding: 2px 10px;"
        "  background: transparent;"
        "  color: rgba(20,8,8,0.42);"
        "  border: none;"
        "}"
        "QPushButton:hover {"
        "  color: rgba(20,8,8,0.68);"
        "  background: rgba(20,8,8,0.05);"
        "}"
    )
    new_style = (
        "QPushButton {"
        "  font-size: 10px; padding: 2px 8px;"
        "  background: transparent;"
        "  color: rgba(20,8,8,0.30);"
        "  border: none;"
        "}"
        "QPushButton:hover {"
        "  color: rgba(20,8,8,0.58);"
        "}"
    )
    return active, inactive, new_style


class SuiteBar(QWidget):
    suite_selected = pyqtSignal(int)
    suite_new_requested = pyqtSignal()
    suite_rename_requested = pyqtSignal(int)
    suite_delete_requested = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(24)
        self._active_idx: int = 0
        self._suite_buttons: list[QPushButton] = []
        self._ACTIVE_STYLE, self._INACTIVE_STYLE, self._NEW_STYLE = _build_suite_styles()

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(6, 0, 6, 0)
        self._layout.setSpacing(0)

        self._new_btn = QPushButton("[+ suite]")
        self._new_btn.setStyleSheet(self._NEW_STYLE)
        self._new_btn.clicked.connect(self.suite_new_requested)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate(self, names: list[str], active_idx: int) -> None:
        """Rebuild all suite tabs. Call whenever the suite list changes."""
        for btn in self._suite_buttons:
            self._layout.removeWidget(btn)
            btn.deleteLater()
        self._suite_buttons.clear()
        self._layout.removeWidget(self._new_btn)

        self._active_idx = active_idx
        for idx, name in enumerate(names):
            btn = self._make_suite_button(name, idx, idx == active_idx)
            self._layout.addWidget(btn)
            self._suite_buttons.append(btn)

        self._layout.addWidget(self._new_btn)
        self._layout.addStretch(1)

    def set_active(self, idx: int) -> None:
        """Highlight the tab at idx without rebuilding all buttons."""
        self._active_idx = idx
        for i, btn in enumerate(self._suite_buttons):
            btn.setStyleSheet(self._ACTIVE_STYLE if i == idx else self._INACTIVE_STYLE)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_suite_button(self, name: str, idx: int, active: bool) -> QPushButton:
        btn = QPushButton(name)
        btn.setStyleSheet(self._ACTIVE_STYLE if active else self._INACTIVE_STYLE)
        btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        btn.customContextMenuRequested.connect(lambda _pos, i=idx: self._show_context_menu(i))
        # double-click to rename — override mouseDoubleClickEvent per button
        btn.mouseDoubleClickEvent = lambda _e, i=idx: self.suite_rename_requested.emit(i)
        btn.clicked.connect(lambda _checked=False, i=idx: self.suite_selected.emit(i))
        return btn

    def _show_context_menu(self, idx: int) -> None:
        menu = QMenu(self)
        rename_act = menu.addAction("[rename]")
        menu.addSeparator()
        delete_act = menu.addAction("[delete]")
        chosen = menu.exec(self.cursor().pos())
        if chosen == rename_act:
            self.suite_rename_requested.emit(idx)
        elif chosen == delete_act:
            self.suite_delete_requested.emit(idx)
