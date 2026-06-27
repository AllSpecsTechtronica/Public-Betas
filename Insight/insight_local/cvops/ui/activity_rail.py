from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QBoxLayout,
    QFrame,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_HORIZONTAL_BUTTON_MIN = 58
_HORIZONTAL_PRIMARY_MIN = 92


class ActivityRailWidget(QFrame):
    """VS Code-style rail: Ecosystem, workbench modes, layout presets.

    Supports both vertical (side rail) and horizontal (top tab strip) orientations.
    """

    ecoPressed = pyqtSignal()
    modePressed = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        orientation: Qt.Orientation = Qt.Orientation.Vertical,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cvOpsActivityRail")
        self._orientation = orientation
        self._horizontal = orientation == Qt.Orientation.Horizontal
        if self._horizontal:
            self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        else:
            self.setFixedWidth(48)
            self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._eco_btn: Optional[QPushButton] = None
        self._mode_buttons: dict[str, QPushButton] = {}
        self._plane_is_eco = False
        self._current_mode = "explore"

        layout: QBoxLayout
        if self._horizontal:
            layout = QHBoxLayout(self)
            layout.setContentsMargins(6, 2, 6, 2)
        else:
            layout = QVBoxLayout(self)
            layout.setContentsMargins(4, 6, 4, 6)
        layout.setSpacing(4)

        self._eco_btn = QPushButton("Ecosystem")
        self._eco_btn.setCheckable(True)
        self._eco_btn.setToolTip("Ecosystem — ontology graph and mission control")
        self._size_rail_button(self._eco_btn, primary=True)
        self._eco_btn.clicked.connect(self._on_eco_clicked)
        layout.addWidget(self._eco_btn)

        layout.addWidget(self._build_separator())

        modes: list[tuple[str, str, str]] = [
            ("collect", "Collect & Edit", "Collect & Edit — scrape, import, edit, and promote datasets"),
            ("explore", "Train", "Train — pick a scenario and kick training"),
            ("test", "Range", "Range — submit jobs, video bench, ranges"),
            ("queue", "Queue", "Live jobs queue"),
            ("data", "Database", "Database — god's-eye view of every data store"),
            ("viz", "Data Viz", "Data Viz — Database tree (scenarios + stores) + CSV charts"),
            ("notes", "Notes", "Notes"),
            ("cells", "Cells", "Cells workspace"),
            ("three_d", "3D", "3D workspace"),
            ("portal", "Scope", "Scope — separate window or embedded web dashboard"),
            ("notifications", "Notifications", "Notification Center — grouped live events"),
            ("settings", "Settings", "Settings and diagnostics"),
        ]
        for mid, label, tip in modes:
            # Escape "&" so Qt renders it literally instead of consuming it as a
            # keyboard mnemonic (e.g. "Collect & Edit", not "Collect _Edit").
            b = QPushButton(label.replace("&", "&&"))
            b.setCheckable(True)
            b.setToolTip(tip)
            self._size_rail_button(b)
            b.clicked.connect(lambda _c=False, m=mid: self._on_mode_clicked(m))
            self._mode_buttons[mid] = b
            layout.addWidget(b)

        layout.addStretch(1)

        self.set_plane_ecosystem(False)
        self._reflect_mode("explore")

    def _build_separator(self) -> QFrame:
        sep = QFrame()
        if self._horizontal:
            sep.setFrameShape(QFrame.Shape.VLine)
            sep.setFixedWidth(1)
        else:
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setFixedHeight(1)
        return sep

    def _size_rail_button(self, btn: QPushButton, *, primary: bool = False) -> None:
        if self._horizontal:
            # Let the button size to its own rendered content. The real font
            # (JetBrains Mono 9px), padding, and the global UI-scale factor are
            # only known at layout time, so a fixed width computed here from the
            # construction-time font metrics clipped the centered labels. A
            # minimum-width floor keeps short tabs from looking cramped; the
            # natural sizeHint guarantees the full text always fits.
            btn.setFixedHeight(30)
            floor = _HORIZONTAL_PRIMARY_MIN if primary else _HORIZONTAL_BUTTON_MIN
            btn.setMinimumWidth(floor)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        else:
            if primary:
                btn.setMinimumHeight(36)
            else:
                btn.setMinimumHeight(32)

    def _on_eco_clicked(self) -> None:
        self.ecoPressed.emit()

    def _on_mode_clicked(self, mode_id: str) -> None:
        self.modePressed.emit(mode_id)

    def set_plane_ecosystem(self, is_ecosystem: bool) -> None:
        self._plane_is_eco = bool(is_ecosystem)
        if self._eco_btn is not None:
            self._eco_btn.setChecked(is_ecosystem)
        for b in self._mode_buttons.values():
            if is_ecosystem:
                b.setChecked(False)

    def set_workbench_mode(self, mode_id: str) -> None:
        self._current_mode = str(mode_id or "explore").strip().lower() or "explore"
        self._reflect_mode(self._current_mode)

    def _reflect_mode(self, mode_id: str) -> None:
        for mid, b in self._mode_buttons.items():
            b.setChecked(mid == mode_id and not self._plane_is_eco)
