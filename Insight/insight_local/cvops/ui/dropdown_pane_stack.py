from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox, QStackedWidget, QVBoxLayout, QWidget


class DropdownPaneStack(QWidget):
    """QTabWidget-like pane switcher backed by a dropdown selector."""

    currentChanged = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("cvOpsDropdownPaneStack")
        self._combo = QComboBox(self)
        self._combo.setObjectName("cvOpsPaneDropdown")
        self._stack = QStackedWidget(self)
        self._combo.currentIndexChanged.connect(self._on_current_index_changed)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        root.addWidget(self._combo)
        root.addWidget(self._stack, stretch=1)

    def addTab(self, pane: QWidget, title: str) -> int:
        index = self._stack.addWidget(pane)
        self._combo.addItem(str(title or "Pane"))
        return index

    def add_pane(self, pane: QWidget, title: str) -> int:
        return self.addTab(pane, title)

    def clear(self) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.blockSignals(False)
        while self._stack.count() > 0:
            widget = self._stack.widget(0)
            self._stack.removeWidget(widget)
            widget.setParent(None)

    def count(self) -> int:
        return self._stack.count()

    def currentIndex(self) -> int:
        return self._stack.currentIndex()

    def current_index(self) -> int:
        return self.currentIndex()

    def currentWidget(self) -> Optional[QWidget]:
        return self._stack.currentWidget()

    def setCurrentIndex(self, index: int) -> None:
        if self.count() <= 0:
            return
        idx = max(0, min(int(index), self.count() - 1))
        self._combo.setCurrentIndex(idx)
        self._stack.setCurrentIndex(idx)

    def set_current_index(self, index: int) -> None:
        self.setCurrentIndex(index)

    def setCurrentWidget(self, widget: QWidget) -> None:
        idx = self._stack.indexOf(widget)
        if idx >= 0:
            self.setCurrentIndex(idx)

    def setTabText(self, index: int, text: str) -> None:
        if 0 <= int(index) < self._combo.count():
            self._combo.setItemText(int(index), str(text or "Pane"))

    def tabText(self, index: int) -> str:
        if 0 <= int(index) < self._combo.count():
            return self._combo.itemText(int(index))
        return ""

    def widget(self, index: int) -> Optional[QWidget]:
        if 0 <= int(index) < self._stack.count():
            return self._stack.widget(int(index))
        return None

    def _on_current_index_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        self.currentChanged.emit(index)
