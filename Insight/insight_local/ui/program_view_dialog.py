"""Source Picker Dialog — entry point for assigning a source to a grid cell.

Opens when the user clicks [add] next to a numbered cell in the main grid.
Left panel: catalog of source types. Right panel: config for the selected type.
"""
from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .theme import beacon_title_tag_css, current_color_scheme, text_css, theme_rgba


_CATALOG: list[dict[str, str]] = [
    {
        "id":       "web",
        "label":    "Web Page",
        "subtitle": "Open a URL in a browser tab",
    },
    {
        "id":       "terminal",
        "label":    "Terminal",
        "subtitle": "Local terminal session",
    },
    {
        "id":       "media",
        "label":    "Media File",
        "subtitle": "Image or video file",
    },
    {
        "id":       "widget",
        "label":    "Widget",
        "subtitle": "Internal utility widget (IBM Demo, etc.)",
    },
]

_WIDGETS: list[str] = ["IBM Demo", "Status Monitor", "Data Inspector"]


class SourcePickerDialog(QDialog):
    """Catalog + config dialog for assigning a source to a grid cell."""

    source_confirmed = pyqtSignal(int, dict)  # cell_num, config

    def __init__(self, cell_num: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._cell_num = cell_num
        self.setWindowTitle(f"[add] Source — Cell {cell_num}")
        self.setMinimumSize(560, 340)
        self.setModal(True)
        self.setStyleSheet(
            f"QDialog {{ background: {theme_rgba('panel', 0.97)}; }}"
            f"QLabel {{ color: {text_css(0.88)}; }}"
            f"QListWidget {{ background: {theme_rgba('panel', 0.60)};"
            f"  border: 1px solid {theme_rgba('accent_dark', 0.30)};"
            f"  color: {text_css(0.88)}; outline: none; }}"
            f"QListWidget::item:selected {{ background: {theme_rgba('accent_dark', 0.35)}; }}"
            f"QLineEdit {{ background: {theme_rgba('panel', 0.25)};"
            f"  border: 1px solid {theme_rgba('accent_dark', 0.35)};"
            f"  color: {text_css(1.0)}; padding: 4px 7px; }}"
            f"QPushButton {{ background: {theme_rgba('panel', 0.45)};"
            f"  border: 1px solid {theme_rgba('accent_dark', 0.40)};"
            f"  color: {text_css(0.90)}; padding: 4px 10px; }}"
            f"QPushButton:hover {{ background: {theme_rgba('accent_dark', 0.20)}; }}"
        )
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel(f"Choose a source type for cell {self._cell_num}")
        title.setProperty("isTitle", True)
        title.setStyleSheet(beacon_title_tag_css(font_size=10) if current_color_scheme() == "beacon" else "font-weight: 600; font-size: 12px;")
        root.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # --- Left: catalog list ---
        left = QWidget()
        left.setFixedWidth(180)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        ll.addWidget(QLabel("Source Type"))
        self._catalog_list = QListWidget()
        self._catalog_list.setSpacing(2)
        for entry in _CATALOG:
            item = QListWidgetItem(entry["label"])
            item.setData(Qt.ItemDataRole.UserRole, entry["id"])
            item.setToolTip(entry["subtitle"])
            self._catalog_list.addItem(item)
        self._catalog_list.currentRowChanged.connect(self._on_catalog_row_changed)
        ll.addWidget(self._catalog_list)
        splitter.addWidget(left)

        # --- Right: stacked config pages ---
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_web_page())       # index 0
        self._stack.addWidget(self._build_terminal_page())  # index 1
        self._stack.addWidget(self._build_media_page())     # index 2
        self._stack.addWidget(self._build_widget_page())    # index 3
        splitter.addWidget(self._stack)
        splitter.setSizes([180, 360])
        root.addWidget(splitter, stretch=1)

        # --- Bottom: cancel / confirm ---
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        btn_cancel = QPushButton("[cancel]")
        btn_cancel.clicked.connect(self.reject)
        btn_confirm = QPushButton("[confirm]")
        btn_confirm.setDefault(True)
        btn_confirm.clicked.connect(self._on_confirm)
        bottom.addWidget(btn_cancel)
        bottom.addWidget(btn_confirm)
        root.addLayout(bottom)

        self._catalog_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Config pages
    # ------------------------------------------------------------------

    def _build_web_page(self) -> QWidget:
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(12, 8, 12, 8)
        vl.setSpacing(6)
        vl.addWidget(QLabel("URL"))
        self._web_url = QLineEdit()
        self._web_url.setPlaceholderText("https://example.com")
        vl.addWidget(self._web_url)
        hint = QLabel("The URL will open in a browser tab inside the grid cell.")
        hint.setStyleSheet("font-size: 10px; color: rgba(20,8,8,0.55);")
        hint.setWordWrap(True)
        vl.addWidget(hint)
        vl.addStretch(1)
        return page

    def _build_terminal_page(self) -> QWidget:
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(12, 8, 12, 8)
        vl.setSpacing(6)
        vl.addWidget(QLabel("Terminal"))
        desc = QLabel(
            "A local terminal session will be opened inside this grid cell.\n"
            "The shell defaults to your system shell."
        )
        desc.setWordWrap(True)
        vl.addWidget(desc)
        vl.addWidget(QLabel("Working directory (optional)"))
        self._term_cwd = QLineEdit()
        self._term_cwd.setPlaceholderText("/path/to/directory")
        vl.addWidget(self._term_cwd)
        vl.addStretch(1)
        return page

    def _build_media_page(self) -> QWidget:
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(12, 8, 12, 8)
        vl.setSpacing(6)
        vl.addWidget(QLabel("Media File"))
        row = QHBoxLayout()
        self._media_path = QLineEdit()
        self._media_path.setPlaceholderText("/path/to/file.mp4 or image.png")
        browse_btn = QPushButton("[browse]")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_media)
        row.addWidget(self._media_path, stretch=1)
        row.addWidget(browse_btn)
        vl.addLayout(row)
        hint = QLabel("Supported: images (png, jpg, bmp) and video files (mp4, avi, mov).")
        hint.setStyleSheet("font-size: 10px; color: rgba(20,8,8,0.55);")
        hint.setWordWrap(True)
        vl.addWidget(hint)
        vl.addStretch(1)
        return page

    def _build_widget_page(self) -> QWidget:
        from PyQt6.QtWidgets import QComboBox
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(12, 8, 12, 8)
        vl.setSpacing(6)
        vl.addWidget(QLabel("Widget"))
        self._widget_combo = QComboBox()
        self._widget_combo.setStyleSheet(
            f"QComboBox {{ background: {theme_rgba('panel', 0.25)};"
            f"  border: 1px solid {theme_rgba('accent_dark', 0.35)};"
            f"  color: {text_css(1.0)}; padding: 3px 6px; }}"
        )
        for name in _WIDGETS:
            self._widget_combo.addItem(name)
        vl.addWidget(self._widget_combo)
        hint = QLabel("The selected internal widget will be embedded in the grid cell.")
        hint.setStyleSheet("font-size: 10px; color: rgba(20,8,8,0.55);")
        hint.setWordWrap(True)
        vl.addWidget(hint)
        vl.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_catalog_row_changed(self, row: int) -> None:
        if row >= 0:
            self._stack.setCurrentIndex(row)

    def _browse_media(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Media File",
            "",
            "Media Files (*.png *.jpg *.jpeg *.bmp *.gif *.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if path:
            self._media_path.setText(path)

    def _on_confirm(self) -> None:
        row = self._catalog_list.currentRow()
        source_id = _CATALOG[row]["id"] if 0 <= row < len(_CATALOG) else "web"
        config: dict[str, Any] = {"type": source_id}

        if source_id == "web":
            config["url"] = self._web_url.text().strip()
        elif source_id == "terminal":
            config["cwd"] = self._term_cwd.text().strip()
        elif source_id == "media":
            config["path"] = self._media_path.text().strip()
        elif source_id == "widget":
            config["widget_name"] = self._widget_combo.currentText()

        self.source_confirmed.emit(self._cell_num, config)
        self.accept()
