from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QSizePolicy, QSplitter, QToolButton, QVBoxLayout, QWidget

# Qt's internal QWIDGETSIZE_MAX — the value used to "unset" a maximum dimension.
_QWIDGETSIZE_MAX = 16_777_215


class CollapsibleSection(QFrame):
    expandedChanged = pyqtSignal(bool)

    def __init__(
        self,
        title: str,
        *,
        expanded: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("opsCell")
        # Last known expanded size along the splitter axis (px). 0 = not yet recorded.
        self._last_expanded_size: int = 0

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 2, 6, 4)
        outer.setSpacing(3)

        self._toggle = QToolButton()
        self._toggle.setProperty("isTitle", True)
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        # Shrink the icon-area Qt reserves for the chevron arrow; the default
        # PM_SmallIconSize (~20-22px on HiDPI) is what makes the title strip
        # look chunky relative to its 10-11px caps text.
        self._toggle.setIconSize(QSize(10, 10))
        # Hug the label width instead of expanding across the whole card. Some
        # colorways fill the title with a background tag; an expanding title
        # turned that into a box stretching far past its text. Maximum keeps the
        # title (and any fill) sized to the text + chevron, left-aligned.
        self._toggle.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.toggled.connect(self.set_expanded)
        outer.addWidget(self._toggle, alignment=Qt.AlignmentFlag.AlignLeft)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(5)
        outer.addWidget(self._body)

        # Apply initial visual state without splitter interaction (not yet parented).
        self._toggle.blockSignals(True)
        self._toggle.setChecked(expanded)
        self._toggle.blockSignals(False)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._body.setVisible(expanded)
        if not expanded:
            self._pin_to_header()

    # ------------------------------------------------------------------ #
    # Qt overrides
    # ------------------------------------------------------------------ #

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        """When collapsed, only report header height so layouts respect the cap."""
        base = super().minimumSizeHint()
        if self._body.isVisible():
            return base
        return QSize(max(base.width(), 1), max(self._header_height(), 1))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def set_title(self, title: str) -> None:
        self._toggle.setText(str(title or ""))

    def set_expanded(self, expanded: bool) -> None:
        self._toggle.blockSignals(True)
        self._toggle.setChecked(expanded)
        self._toggle.blockSignals(False)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)

        if not expanded:
            self._record_size()

        self._body.setVisible(expanded)

        if not expanded:
            # Hard-cap height to header only.  This overrides any child
            # setMinimumHeight() calls that would otherwise bubble up through
            # the layout and prevent the splitter from snapping tight.
            self._pin_to_header()
        else:
            # Remove the cap so the section can grow freely again.
            self.setMinimumHeight(0)
            self.setMaximumHeight(_QWIDGETSIZE_MAX)

        self.updateGeometry()
        self._adjust_splitter(expanding=expanded)
        self.expandedChanged.emit(bool(expanded))

    def is_expanded(self) -> bool:
        return self._body.isVisible()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _header_height(self) -> int:
        margins = self.layout().contentsMargins() if self.layout() is not None else None
        extra = (margins.top() + margins.bottom()) if margins is not None else 0
        return self._toggle.sizeHint().height() + extra

    def _pin_to_header(self) -> None:
        """Force this widget to exactly header height, defeating child minimums."""
        h = self._header_height()
        self.setMinimumHeight(h)
        self.setMaximumHeight(h)

    # ------------------------------------------------------------------ #
    # Splitter integration
    # ------------------------------------------------------------------ #

    def _find_splitter(self) -> tuple[Optional[QSplitter], int]:
        p = self.parentWidget()
        if isinstance(p, QSplitter):
            for i in range(p.count()):
                if p.widget(i) is self:
                    return p, i
        return None, -1

    def _axis_size(self, sizes: list[int], idx: int) -> int:
        return sizes[idx] if idx < len(sizes) else 0

    def _record_size(self) -> None:
        splitter, idx = self._find_splitter()
        if splitter is None:
            return
        sizes = splitter.sizes()
        if idx < len(sizes) and sizes[idx] > 0:
            self._last_expanded_size = sizes[idx]

    def _adjust_splitter(self, *, expanding: bool) -> None:
        splitter, idx = self._find_splitter()
        if splitter is None:
            return
        sizes = list(splitter.sizes())
        if not sizes or idx >= len(sizes):
            return

        if not expanding:
            target = self._header_height()
            freed = sizes[idx] - target
            if freed <= 0:
                return
            sizes[idx] = target
            self._give_space(sizes, idx, freed, splitter)
        else:
            want = self._last_expanded_size if self._last_expanded_size > 0 else 200
            gain = want - sizes[idx]
            if gain <= 0:
                return
            sizes[idx] = want
            self._take_space(sizes, idx, gain, splitter)

        splitter.setSizes(sizes)

    def _give_space(
        self, sizes: list[int], idx: int, amount: int, splitter: QSplitter
    ) -> None:
        n = len(sizes)
        for delta in (1, -1, 2, -2, 3, -3):
            j = idx + delta
            if 0 <= j < n and splitter.widget(j) is not None:
                sizes[j] += amount
                return
        for j in range(n - 1, -1, -1):
            if j != idx:
                sizes[j] += amount
                return

    def _take_space(
        self, sizes: list[int], idx: int, amount: int, splitter: QSplitter
    ) -> None:
        n = len(sizes)
        remaining = amount
        for delta in (1, -1, 2, -2, 3, -3):
            if remaining <= 0:
                break
            j = idx + delta
            if not (0 <= j < n):
                continue
            w = splitter.widget(j)
            if w is None:
                continue
            if isinstance(w, CollapsibleSection) and not w.is_expanded():
                continue
            # Floor: the pinned header height for collapsed siblings, else a
            # small buffer so we don't crush an expanded one.
            floor = w._header_height() if isinstance(w, CollapsibleSection) else 30
            available = sizes[j] - floor
            take = min(max(0, available), remaining)
            if take > 0:
                sizes[j] -= take
                remaining -= take
