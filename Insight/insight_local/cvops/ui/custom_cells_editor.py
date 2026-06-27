from __future__ import annotations

import base64
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractButton,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .device_selector import DeviceSelector
from .training_console import TrainingConsoleWidget
from .cvops_theme import cvops_color, set_cvops_stylesheet

WB_FONT_MONO = "JetBrains Mono, IBM Plex Mono, Courier New, monospace"

# Extensions ingested as UTF-8 text; everything else is stored base64 (images, etc.).
_TEXT_ASSET_EXT = {
    "py", "txt", "csv", "tsv", "json", "jsonl", "md", "yaml", "yml",
    "ini", "cfg", "conf", "log", "xml", "html", "htm", "sql", "sh",
}


def _clean_internal_file_name(raw: str, fallback: str = "data.txt") -> str:
    """Return a safe cell-local relative path for an internal project file."""
    text = str(raw or "").strip().replace("\\", "/").lstrip("/")
    parts: list[str] = []
    for part in text.split("/"):
        piece = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in part).strip(" .")
        if not piece or piece in {".", ".."}:
            continue
        parts.append(piece[:80])
    return "/".join(parts)[:220] or fallback


def _format_for_file_name(name: str) -> str:
    return Path(str(name or "")).suffix.lower().lstrip(".") or "text"


class _AssetDropList(QListWidget):
    """Cell-asset list that accepts dropped files from the OS file manager."""

    filesDropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasUrls():
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.toLocalFile()]
            if paths:
                self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


def _ghost(button: QAbstractButton, *, min_width: int = 0) -> QAbstractButton:
    button.setProperty("variant", "ghost")
    if min_width:
        button.setMinimumWidth(min_width)
    button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    style = button.style()
    style.unpolish(button)
    style.polish(button)
    return button


class CustomCellsEditor(QWidget):
    """Reusable draft-cell editor backed by the custom_cells service endpoints."""

    errorRaised = pyqtSignal(str)
    statusChanged = pyqtSignal(str)
    draftSaved = pyqtSignal(str)
    runDraftRequested = pyqtSignal(str, dict)
    scenarioMutated = pyqtSignal(str)

    def __init__(
        self,
        *,
        http_get: Callable[[str], dict[str, Any]],
        http_put: Optional[Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]] = None,
        http_post: Optional[Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("customCellsEditor")
        self._http_get = http_get
        self._http_put = http_put
        self._http_post = http_post
        self._scenario = ""
        self._entry: dict[str, Any] = {}
        self._cells_data: list[dict[str, Any]] = []
        self._prev_list_row = -1
        self._loaded_scenario = ""
        self._assets_open = False
        self._asset_editor_key: Optional[tuple[str, str]] = None

        # Scenario datasets are still part of the saved draft, but the JSON
        # editor was removed from this view. We keep the value internally so
        # collect_put_body() / collect_train_override() stay backward compatible.
        self._scenario_datasets: list[Any] = []
        # Selected accelerator token ("" / "0" / "mps" / "cpu"). UI-only for now —
        # remembered for the run but not yet wired into the train override.
        self._selected_device = ""
        set_cvops_stylesheet(self, self._editor_qss)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(10)

        # --- Top bar: title + SAVE / START MACHINE + machine status dot --------
        toolbar_card = QFrame()
        toolbar_card.setObjectName("cellsNotebookToolbar")
        toolbar_card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        toolbar = QVBoxLayout(toolbar_card)
        toolbar.setContentsMargins(14, 12, 14, 12)
        toolbar.setSpacing(10)

        toolbar_top = QHBoxLayout()
        toolbar_top.setContentsMargins(0, 0, 0, 0)
        toolbar_top.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setContentsMargins(0, 0, 0, 0)
        title_col.setSpacing(4)
        self._title = QLabel("Notebook")
        self._title.setObjectName("cellsNotebookTitle")
        title_col.addWidget(self._title)
        self._meta = QLabel("Draft Python files, internal file/data spaces, notebook runtime controls, and live output.")
        self._meta.setObjectName("cellsNotebookMeta")
        self._meta.setWordWrap(True)
        title_col.addWidget(self._meta)
        toolbar_top.addLayout(title_col, stretch=1)

        self._context_controls = QHBoxLayout()
        self._context_controls.setContentsMargins(0, 0, 0, 0)
        self._context_controls.setSpacing(10)
        toolbar_top.addLayout(self._context_controls, stretch=0)
        toolbar.addLayout(toolbar_top)

        # Machine selector: reuse the shared accelerator picker (Auto / GPU / CPU).
        toolbar_actions = QHBoxLayout()
        toolbar_actions.setContentsMargins(0, 0, 0, 0)
        toolbar_actions.setSpacing(10)
        machine_lbl = QLabel("Machine:")
        machine_lbl.setObjectName("cellsNotebookHint")
        toolbar_actions.addWidget(machine_lbl)
        self._device_selector = DeviceSelector()
        self._device_selector.setToolTip(
            "Choose which accelerator runs this cell when you start the machine. "
            "Auto picks the detected default; pick a GPU to pin to it, or CPU to skip GPU."
        )
        self._device_selector.deviceChanged.connect(self._on_device_changed)
        toolbar_actions.addWidget(self._device_selector)
        toolbar_actions.addStretch(1)

        self._save_btn = QPushButton("Save draft")
        self._save_btn.setObjectName("cellsNotebookGhost")
        self._save_btn.clicked.connect(self.save_draft)
        self._save_btn.setMinimumWidth(104)
        toolbar_actions.addWidget(self._save_btn)

        self._run_btn = QPushButton("Run notebook")
        self._run_btn.setObjectName("cellsNotebookPrimary")
        self._run_btn.clicked.connect(self.run_draft)
        self._run_btn.setMinimumWidth(150)
        toolbar_actions.addWidget(self._run_btn)

        self._machine_dot = QLabel("●")
        self._machine_dot.setObjectName("cellsNotebookMachineDot")
        toolbar_actions.addWidget(self._machine_dot)
        toolbar.addLayout(toolbar_actions)
        outer.addWidget(toolbar_card)

        # --- Body: file catalog -> internal files/data/code workspace ----------
        editor_split = QSplitter(Qt.Orientation.Horizontal)
        editor_split.setChildrenCollapsible(False)
        editor_split.setHandleWidth(3)

        list_pane = QFrame()
        list_pane.setObjectName("cellsNotebookSidebar")
        list_pane.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        list_pane.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        list_lay = QVBoxLayout(list_pane)
        list_lay.setContentsMargins(12, 12, 12, 12)
        list_lay.setSpacing(8)

        self._sidebar_stack = QStackedWidget()
        self._sidebar_stack.setObjectName("cellsNotebookCatalogStack")

        self._file_catalog_page = QWidget()
        self._file_catalog_page.setObjectName("cellsNotebookFileCatalogPage")
        file_catalog_lay = QVBoxLayout(self._file_catalog_page)
        file_catalog_lay.setContentsMargins(0, 0, 0, 0)
        file_catalog_lay.setSpacing(8)

        list_head = QHBoxLayout()
        list_head.setContentsMargins(0, 0, 0, 0)
        list_head.setSpacing(6)
        list_title_col = QVBoxLayout()
        list_title_col.setContentsMargins(0, 0, 0, 0)
        list_title_col.setSpacing(2)
        list_label = QLabel("1. File")
        list_label.setObjectName("cellsNotebookSectionTitle")
        list_title_col.addWidget(list_label)
        list_hint = QLabel("Pick the cell file. Each file owns internal files/data and run code.")
        list_hint.setObjectName("cellsNotebookHint")
        list_hint.setWordWrap(True)
        list_title_col.addWidget(list_hint)
        list_head.addLayout(list_title_col, stretch=1)
        list_head.addStretch(1)
        self._add_cell_btn = QPushButton("Add file")
        self._add_cell_btn.setObjectName("cellsNotebookGhost")
        self._add_cell_btn.clicked.connect(self.add_cell)
        self._remove_cell_btn = QPushButton("Delete")
        self._remove_cell_btn.setObjectName("cellsNotebookGhost")
        self._remove_cell_btn.clicked.connect(self.remove_cell)
        for button in (self._add_cell_btn, self._remove_cell_btn):
            _ghost(button)
            list_head.addWidget(button)
        file_catalog_lay.addLayout(list_head)
        self._cell_list = QListWidget()
        self._cell_list.setObjectName("cellsNotebookFileList")
        self._cell_list.setMinimumWidth(230)
        self._cell_list.setMinimumHeight(160)
        # Double-click a file name to rename it (the inline Name row was removed).
        self._cell_list.itemClicked.connect(lambda _item: self._show_asset_catalog())
        self._cell_list.itemDoubleClicked.connect(self._rename_current_cell)
        self._cell_list.currentRowChanged.connect(self._on_list_row_changed)
        file_catalog_lay.addWidget(self._cell_list, stretch=1)

        # Compatibility shim for older call-sites/tests that set the removed
        # inline name field directly. The visible rename path is still double-click.
        self._cell_name = QLineEdit(self)
        self._cell_name.setVisible(False)
        self._cell_name.textChanged.connect(self._on_hidden_cell_name_changed)

        self._sidebar_stack.addWidget(self._file_catalog_page)
        self._asset_panel = self._build_asset_panel()
        self._sidebar_stack.addWidget(self._asset_panel)
        list_lay.addWidget(self._sidebar_stack, stretch=1)
        editor_split.addWidget(list_pane)
        editor_split.setStretchFactor(0, 1)

        # --- Vertical split: editor on top, live terminal output below --------
        body_split = QSplitter(Qt.Orientation.Vertical)
        body_split.setChildrenCollapsible(False)
        body_split.setHandleWidth(3)
        body_split.addWidget(editor_split)

        # Colab-style terminal: streams this scenario's cell output line by line.
        output_pane = QFrame()
        output_pane.setObjectName("cellsNotebookOutputPane")
        output_pane.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        output_lay = QVBoxLayout(output_pane)
        output_lay.setContentsMargins(12, 12, 12, 12)
        output_lay.setSpacing(8)
        output_title = QLabel("Output")
        output_title.setObjectName("cellsNotebookSectionTitle")
        output_lay.addWidget(output_title)
        output_hint = QLabel("Live draft-cell output appears here while the notebook runs.")
        output_hint.setObjectName("cellsNotebookHint")
        output_hint.setWordWrap(True)
        output_lay.addWidget(output_hint)
        self._console = TrainingConsoleWidget(terminal_only=True)
        self._console.setObjectName("cellsNotebookTerminal")
        self._console.setMinimumHeight(170)
        output_lay.addWidget(self._console, stretch=1)
        body_split.addWidget(output_pane)
        body_split.setStretchFactor(0, 3)
        body_split.setStretchFactor(1, 2)
        body_split.setSizes([560, 240])
        outer.addWidget(body_split, stretch=1)

        self._status = QLabel("")
        self._status.setObjectName("cellsNotebookStatus")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)
        self.clear_scenario("Select a Custom Code scenario to edit draft cells.")

    def install_context_controls(self, *widgets: QWidget) -> None:
        for widget in widgets:
            if widget is None:
                continue
            self._context_controls.addWidget(widget)

    # ------------------------------------------------------------------ #
    # Per-cell asset space
    # ------------------------------------------------------------------ #

    def _build_asset_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("cellsNotebookAssetPanel")
        panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        self._asset_back_btn = QPushButton("Back")
        self._asset_back_btn.setObjectName("cellsNotebookGhost")
        self._asset_back_btn.clicked.connect(self._show_file_catalog)
        self._asset_title = QLabel("2. Internal files, data & code")
        self._asset_title.setObjectName("cellsNotebookSectionTitle")
        head.addWidget(self._asset_title, stretch=1)
        self._asset_new_btn = QPushButton("New text file")
        self._asset_new_btn.setObjectName("cellsNotebookGhost")
        self._asset_new_btn.clicked.connect(self._new_internal_file)
        self._asset_add_btn = QPushButton("Add file")
        self._asset_add_btn.setObjectName("cellsNotebookGhost")
        self._asset_add_btn.clicked.connect(self._add_asset_files)
        self._asset_remove_btn = QPushButton("Remove")
        self._asset_remove_btn.setObjectName("cellsNotebookGhost")
        self._asset_remove_btn.clicked.connect(self._remove_asset)
        for button in (self._asset_back_btn, self._asset_add_btn, self._asset_new_btn, self._asset_remove_btn):
            _ghost(button)
            head.addWidget(button)
        lay.addLayout(head)

        hint = QLabel(
            "Cell-local project space. Open Code to edit run logic, or add/edit internal data files."
        )
        hint.setObjectName("cellsNotebookHint")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self._asset_list = _AssetDropList()
        self._asset_list.setObjectName("cellsNotebookAssetList")
        self._asset_list.setMaximumHeight(120)
        self._asset_list.filesDropped.connect(self._ingest_asset_paths)
        self._asset_list.currentRowChanged.connect(self._on_asset_selection_changed)
        lay.addWidget(self._asset_list)

        self._asset_editor_stack = QStackedWidget()
        self._asset_editor_stack.setObjectName("cellsNotebookAssetEditorStack")

        self._code = QPlainTextEdit()
        self._code.setObjectName("cellsNotebookCodeEditor")
        self._code.setPlaceholderText("def run(ctx, prev):\n    ...")
        self._code.setMinimumHeight(220)
        self._code.textChanged.connect(self._sync_editor_chrome)
        self._asset_editor_stack.addWidget(self._code)

        self._asset_content = QPlainTextEdit()
        self._asset_content.setObjectName("cellsNotebookAssetEditor")
        self._asset_content.setPlaceholderText("Select or create an internal text file.")
        self._asset_content.setMaximumHeight(180)
        self._asset_content.textChanged.connect(self._on_asset_content_changed)
        self._asset_editor_stack.addWidget(self._asset_content)
        lay.addWidget(self._asset_editor_stack, stretch=1)
        return panel

    def _toggle_assets(self) -> None:
        cell = self._current_cell()
        if cell is None:
            return
        if self._assets_open:
            self._show_file_catalog()
        else:
            self._show_asset_catalog()

    def _show_file_catalog(self) -> None:
        self._assets_open = False
        if hasattr(self, "_sidebar_stack"):
            self._sidebar_stack.setCurrentWidget(self._file_catalog_page)
        self._sync_editor_chrome()

    def _show_asset_catalog(self) -> None:
        if self._current_cell() is None:
            return
        self._assets_open = True
        self._refresh_asset_list()
        if hasattr(self, "_sidebar_stack"):
            self._sidebar_stack.setCurrentWidget(self._asset_panel)
        self._sync_editor_chrome()

    def _current_cell(self) -> Optional[dict[str, Any]]:
        row = self._cell_list.currentRow()
        if 0 <= row < len(self._cells_data):
            return self._cells_data[row]
        return None

    @staticmethod
    def _cell_assets(cell: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        assets: list[tuple[str, dict[str, Any]]] = []
        pasted_names: set[str] = set()
        for b in cell.get("pasted_files") or []:
            if isinstance(b, dict) and str(b.get("name") or "").strip():
                pasted_names.add(str(b.get("name") or ""))
                assets.append(("pasted", b))
        for d in cell.get("datasets") or []:
            if not isinstance(d, dict) or not str(d.get("path") or "").strip():
                continue
            # Saved internal files reopen as pasted_files so they can be edited.
            # Hide the backing managed dataset row when the editable file is present.
            if str(d.get("mode") or "") == "managed_copy":
                path_name = str(d.get("path") or "").replace("\\", "/").split("/")[-1]
                if path_name in {Path(name).name for name in pasted_names}:
                    continue
            assets.append(("dataset", d))
        return assets

    def _refresh_asset_list(self, select_key: Optional[tuple[str, str]] = None) -> None:
        current_key = select_key
        if current_key is None:
            item = self._asset_list.currentItem()
            if item is not None:
                data = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(data, tuple) and len(data) == 2:
                    current_key = (str(data[0]), str(data[1]))
        self._asset_list.blockSignals(True)
        self._asset_list.clear()
        cell = self._current_cell()
        if cell is None:
            self._asset_list.blockSignals(False)
            self._show_asset_editor_empty("Select a file first.")
            return
        target_row = -1
        code_entry = QListWidgetItem("Code  [run]")
        code_entry.setData(Qt.ItemDataRole.UserRole, ("code", ""))
        self._asset_list.addItem(code_entry)
        if current_key in (None, ("code", "")):
            target_row = 0
        for source, item in self._cell_assets(cell):
            if source == "pasted":
                name = str(item.get("name") or "asset")
                fmt = str(item.get("format") or _format_for_file_name(name))
                label = f"{name}  [internal {fmt}]"
                key = name
            else:
                name = str(item.get("name") or Path(str(item.get("path") or "")).name or "asset")
                kind = str(item.get("kind") or "").strip()
                label = f"{name}  [linked {kind}]" if kind else f"{name}  [linked]"
                key = str(item.get("path") or "")
            entry = QListWidgetItem(label)
            entry.setData(Qt.ItemDataRole.UserRole, (source, key))
            self._asset_list.addItem(entry)
            if current_key == (source, key):
                target_row = self._asset_list.count() - 1
        self._asset_list.blockSignals(False)
        if self._asset_list.count() > 0:
            self._asset_list.setCurrentRow(target_row if target_row >= 0 else 0)
        else:
            self._show_asset_editor_empty("No internal files yet. Create a text file or add one from disk.")

    def _add_asset_files(self) -> None:
        if self._current_cell() is None:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Add files to this cell", "", "All files (*.*)")
        if paths:
            self._ingest_asset_paths(paths)

    def _ingest_asset_paths(self, paths: list[str]) -> None:
        cell = self._current_cell()
        if cell is None:
            return
        pasted = cell.setdefault("pasted_files", [])
        if not isinstance(pasted, list):
            pasted = []
            cell["pasted_files"] = pasted
        added = 0
        last_name = ""
        for raw in paths:
            blob = self._read_file_as_blob(str(raw))
            if blob is None:
                continue
            # Replace any same-named pending blob so re-adding refreshes content.
            pasted[:] = [b for b in pasted if str(b.get("name") or "") != blob["name"]]
            pasted.append(blob)
            added += 1
            last_name = str(blob.get("name") or "")
        if added:
            self._refresh_asset_list(("pasted", last_name) if last_name else None)
            self._sync_editor_chrome()
            name = str(cell.get("name") or cell.get("id") or "cell")
            self._set_status(f"Added {added} internal file(s) to {name}. Save draft to persist.")

    def _new_internal_file(self) -> None:
        cell = self._current_cell()
        if cell is None:
            return
        raw_name, ok = QInputDialog.getText(
            self,
            "New Internal File",
            "File name:",
            text="data.txt",
        )
        if not ok:
            return
        name = _clean_internal_file_name(raw_name)
        pasted = cell.setdefault("pasted_files", [])
        if not isinstance(pasted, list):
            pasted = []
            cell["pasted_files"] = pasted
        pasted[:] = [b for b in pasted if str(b.get("name") or "") != name]
        pasted.append({"name": name, "content": "", "format": _format_for_file_name(name)})
        self._refresh_asset_list(("pasted", name))
        self._sync_editor_chrome()
        self._set_status(f"Created internal file {name}. Save draft to persist.")

    def _read_file_as_blob(self, path: str) -> Optional[dict[str, Any]]:
        p = Path(path)
        try:
            data = p.read_bytes()
        except Exception as exc:
            self.errorRaised.emit(f"Could not read {p.name}: {exc}")
            return None
        ext = p.suffix.lower().lstrip(".")
        name = _clean_internal_file_name(p.name)
        if ext in _TEXT_ASSET_EXT:
            try:
                return {"name": name, "content": data.decode("utf-8"), "format": ext or "text"}
            except Exception:
                pass
        return {
            "name": name,
            "content": base64.b64encode(data).decode("ascii"),
            "encoding": "base64",
            "format": ext or "binary",
        }

    def _asset_blob(self, name: str) -> Optional[dict[str, Any]]:
        cell = self._current_cell()
        if cell is None:
            return None
        for blob in cell.get("pasted_files") or []:
            if isinstance(blob, dict) and str(blob.get("name") or "") == name:
                return blob
        return None

    def _show_asset_editor_empty(self, message: str) -> None:
        self._asset_editor_key = None
        self._asset_content.blockSignals(True)
        self._asset_content.clear()
        self._asset_content.setPlaceholderText(message)
        self._asset_content.setReadOnly(True)
        self._asset_content.setEnabled(False)
        if hasattr(self, "_asset_editor_stack"):
            self._asset_editor_stack.setCurrentWidget(self._asset_content)
        self._asset_content.blockSignals(False)

    def _on_asset_selection_changed(self, _row: int) -> None:
        item = self._asset_list.currentItem()
        if item is None:
            self._show_asset_editor_empty("No internal file selected.")
            return
        source, key = item.data(Qt.ItemDataRole.UserRole)
        source = str(source or "")
        key = str(key or "")
        self._asset_editor_key = (source, key)
        if source == "code":
            if hasattr(self, "_asset_editor_stack"):
                self._asset_editor_stack.setCurrentWidget(self._code)
            self._code.setEnabled(True)
            return
        self._asset_content.blockSignals(True)
        try:
            if hasattr(self, "_asset_editor_stack"):
                self._asset_editor_stack.setCurrentWidget(self._asset_content)
            if source != "pasted":
                self._asset_content.clear()
                self._asset_content.setPlaceholderText("Linked dataset/file references are read-only here.")
                self._asset_content.setReadOnly(True)
                self._asset_content.setEnabled(False)
                return
            blob = self._asset_blob(key)
            if blob is None:
                self._show_asset_editor_empty("Internal file is unavailable.")
                return
            if str(blob.get("encoding") or "").lower() == "base64":
                self._asset_content.clear()
                self._asset_content.setPlaceholderText("Binary internal file. Replace it with Add file to change its contents.")
                self._asset_content.setReadOnly(True)
                self._asset_content.setEnabled(False)
                return
            self._asset_content.setPlainText(str(blob.get("content") or ""))
            self._asset_content.setPlaceholderText(f"Edit {key}")
            self._asset_content.setReadOnly(False)
            self._asset_content.setEnabled(True)
        finally:
            self._asset_content.blockSignals(False)

    def _on_asset_content_changed(self) -> None:
        key = self._asset_editor_key
        if key is None or key[0] != "pasted":
            return
        blob = self._asset_blob(key[1])
        if blob is None or str(blob.get("encoding") or "").lower() == "base64":
            return
        blob["content"] = self._asset_content.toPlainText()
        blob["format"] = str(blob.get("format") or _format_for_file_name(key[1]))

    def _remove_asset(self) -> None:
        cell = self._current_cell()
        if cell is None:
            return
        item = self._asset_list.currentItem()
        if item is None:
            return
        source, key = item.data(Qt.ItemDataRole.UserRole)
        if source == "code":
            return
        if source == "pasted":
            cell["pasted_files"] = [
                b for b in cell.get("pasted_files") or [] if str(b.get("name") or "") != key
            ]
        else:
            cell["datasets"] = [
                d for d in cell.get("datasets") or [] if str(d.get("path") or "") != key
            ]
        self._refresh_asset_list()
        self._sync_editor_chrome()

    @staticmethod
    def _editor_qss() -> str:
        panel = cvops_color("bg_panel")
        void = cvops_color("bg_void")
        text = cvops_color("text_signal")
        muted = cvops_color("text_iron")
        border = cvops_color("line_light")
        accent = cvops_color("line_bright")
        selected = cvops_color("selection_active")
        selected_text = cvops_color("selection_text")
        return f"""
QWidget#customCellsEditor {{
    background: transparent;
}}
QFrame#cellsNotebookToolbar,
QFrame#cellsNotebookSidebar,
QFrame#cellsNotebookOutputPane,
QFrame#cellsNotebookAssetPanel,
QFrame#cellsNotebookTerminal {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 0px;
}}
QFrame#cellsNotebookAssetPanel {{
    background: {void};
}}
QListWidget#cellsNotebookAssetList {{
    background: {panel};
    color: {text};
    border: 1px solid {border};
    border-radius: 0px;
    selection-background-color: {selected};
    selection-color: {selected_text};
}}
QPlainTextEdit#cellsNotebookAssetEditor,
QPlainTextEdit#cellsNotebookCodeEditor {{
    background: {panel};
    color: {text};
    border: 1px solid {border};
    border-radius: 0px;
    font-family: {WB_FONT_MONO};
    padding: 6px;
    selection-background-color: {selected};
    selection-color: {selected_text};
}}
QListWidget#cellsNotebookAssetList::item {{
    padding: 4px 8px;
}}
QListWidget#cellsNotebookAssetList::item:selected {{
    background: {selected};
    color: {selected_text};
}}
QLabel#cellsNotebookTitle {{
    color: {text};
    font-size: 14px;
    font-weight: 700;
    background: transparent;
    border: none;
}}
QLabel#cellsNotebookSectionTitle {{
    color: {text};
    font-size: 12px;
    font-weight: 700;
    background: transparent;
    border: none;
}}
QLabel#cellsNotebookMeta,
QLabel#cellsNotebookHint,
QLabel#cellsNotebookStatus {{
    color: {muted};
    font-size: 10px;
    background: transparent;
    border: none;
}}
QLabel#cellsNotebookMachineDot {{
    color: {accent};
    font-size: 16px;
    border: none;
    background: transparent;
}}
QPushButton#cellsNotebookPrimary,
QPushButton#cellsNotebookGhost {{
    min-height: 30px;
    padding: 4px 12px;
    font-weight: 600;
}}
QPushButton#cellsNotebookPrimary {{
    background: {accent};
    color: {void};
    border: 1px solid {accent};
}}
QPushButton#cellsNotebookPrimary:hover {{
    color: {text};
}}
QPushButton#cellsNotebookGhost {{
    background: transparent;
    color: {text};
    border: 1px solid {border};
}}
QPushButton#cellsNotebookGhost:hover {{
    border-color: {accent};
    color: {accent};
}}
QListWidget#cellsNotebookFileList {{
    background: {void};
    color: {text};
    border: 1px solid {border};
    border-radius: 0px;
    selection-background-color: {selected};
    selection-color: {selected_text};
}}
QListWidget#cellsNotebookFileList::item {{
    padding: 6px 8px;
    border-radius: 0px;
}}
QListWidget#cellsNotebookFileList::item:selected {{
    background: {selected};
    color: {selected_text};
}}
"""

    def set_scenario(self, scenario: str, entry: Optional[dict[str, Any]] = None) -> None:
        name = str(scenario or "").strip()
        self._scenario = name
        self._entry = dict(entry or {})
        self.setEnabled(bool(name))
        self._title.setText(name if name else "Notebook")
        self._meta.setText(
            "Draft Python files, internal file/data spaces, notebook runtime controls, and live output."
            if name
            else "Select a Custom Code scenario to begin editing notebook files."
        )
        if not name:
            self.clear_scenario("Select a Custom Code scenario to edit draft cells.")
            return
        if self._loaded_scenario != name:
            self.load_from_server(name)
            self._loaded_scenario = name
        self._sync_editor_chrome()

    def clear_scenario(self, message: str = "") -> None:
        self._scenario = ""
        self._entry = {}
        self._loaded_scenario = ""
        self._cells_data = []
        self._prev_list_row = -1
        self._scenario_datasets = []
        self._refresh_list()
        self.setEnabled(False)
        self._title.setText("Notebook")
        self._meta.setText("Select a Custom Code scenario to begin editing notebook files.")
        self._set_status(message)
        self._sync_editor_chrome()

    def load_from_server(self, scenario: Optional[str] = None) -> None:
        name = str(scenario or self._scenario or "").strip()
        self._cells_data = []
        self._prev_list_row = -1
        if not name:
            self._scenario_datasets = []
            self._refresh_list()
            return
        try:
            data = self._http_get(f"/scenarios/{name}/custom_cells")
        except Exception as exc:
            self._scenario_datasets = []
            self._refresh_list()
            msg = f"Unable to load custom cells for {name}: {exc}"
            self._set_status(msg)
            self.errorRaised.emit(msg)
            return
        if not isinstance(data, dict):
            self._scenario_datasets = []
            self._refresh_list()
            return
        for cell in data.get("cells") or []:
            if isinstance(cell, dict):
                self._cells_data.append(dict(cell))
        sd = data.get("scenario_datasets")
        self._scenario_datasets = list(sd) if isinstance(sd, list) else []
        self._refresh_list(select_row=0)
        self._sync_editor_chrome()

    def save_draft(self) -> dict[str, Any]:
        name = self._scenario
        if not name:
            return {}
        if self._http_put is None:
            self._set_status("Save draft requires HTTP PUT from the app shell.")
            return {}
        try:
            data = self._http_put(f"/scenarios/{name}/custom_cells", self.collect_put_body())
        except Exception as exc:
            msg = f"Save draft failed: {exc}"
            self._set_status(msg)
            self.errorRaised.emit(msg)
            return {}
        if isinstance(data, dict):
            self._replace_from_draft(data)
        self._set_status(f"{name}: custom cell draft saved.")
        self.draftSaved.emit(name)
        return data if isinstance(data, dict) else {}

    def run_draft(self) -> None:
        data = self.save_draft()
        if not data or not self._scenario:
            return
        override = self.collect_train_override()
        # Colab-style: clear the terminal and announce the run before kicking it.
        self._console.reset()
        self._console.set_training_active(True)
        device = self._selected_device or "auto"
        self._console.append_line(
            f"[machine] starting {self._scenario} on device={device}", "stdout"
        )
        self.runDraftRequested.emit(self._scenario, override)

    def _on_device_changed(self, token: str) -> None:
        self._selected_device = str(token or "")

    def device(self) -> str:
        """Selected accelerator token ("" / "0" / "mps" / "cpu")."""
        return self._selected_device

    # --- Live console feed -------------------------------------------------
    def append_console_line(self, line: str, stream: str = "stdout") -> None:
        """Append a raw line to the inline terminal."""
        self._console.append_line(line, stream)

    def reset_console(self) -> None:
        self._console.reset()

    def apply_cell_progress(self, payload: dict[str, Any]) -> None:
        """Mirror a cell_progress websocket event into the inline terminal.

        Only events for the currently-loaded scenario are rendered. Mirrors the
        formatting used by the Workbench training console so output reads the same
        in both places.
        """
        if not isinstance(payload, dict):
            return
        scenario = str(payload.get("scenario") or "").strip()
        if not scenario or scenario != self._scenario:
            return
        cell_name = str(payload.get("cell_name") or "").strip() or f"Cell {payload.get('cell_index', '')}"
        status = str(payload.get("cell_status") or "").strip() or "done"
        output = str(payload.get("output") or "").rstrip()

        if status == "running":
            self._console.append_line(f"[cell] running: {cell_name}", "stdout")
            return
        if output:
            for ln in output.splitlines():
                self._console.append_line(f"[{cell_name}] {ln}", "stdout")
        is_err = status == "error"
        self._console.append_line(f"[cell] {status}: {cell_name}", "stderr" if is_err else "stdout")
        if status in {"done", "error", "completed", "cancelled"}:
            self._console.set_training_active(False)

    def collect_put_body(self) -> dict[str, Any]:
        row = self._cell_list.currentRow()
        if row >= 0:
            self._flush_editor_to_row(row)
        scenario_datasets = list(self._scenario_datasets) if isinstance(self._scenario_datasets, list) else []
        out_cells: list[dict[str, Any]] = []
        for cell in self._cells_data:
            copied = dict(cell)
            pasted = copied.get("pasted_files")
            if not isinstance(pasted, list) or not pasted:
                copied.pop("pasted_files", None)
            out_cells.append(copied)
        return {"cells": out_cells, "scenario_datasets": scenario_datasets}

    def collect_train_override(self) -> dict[str, Any]:
        cells_out: list[dict[str, Any]] = []
        for cell in self._cells_data:
            if not isinstance(cell, dict):
                continue
            path = str(cell.get("path") or "").strip()
            if not path:
                continue
            spec: dict[str, Any] = {
                "id": str(cell.get("id") or ""),
                "name": str(cell.get("name") or ""),
                "path": path,
                "entry": str(cell.get("entry") or "run"),
            }
            datasets = cell.get("datasets")
            if isinstance(datasets, list):
                spec["datasets"] = datasets
            cells_out.append(spec)
        scenario_datasets = list(self._scenario_datasets) if isinstance(self._scenario_datasets, list) else []
        return {"cells": cells_out, "datasets": scenario_datasets} if cells_out else {}

    def add_cell(self) -> None:
        current = self._cell_list.currentRow()
        if current >= 0:
            self._flush_editor_to_row(current)
        self._cells_data.append(self._default_cell())
        self._refresh_list(select_row=len(self._cells_data) - 1)
        self._show_asset_catalog()

    def remove_cell(self) -> None:
        row = self._cell_list.currentRow()
        if row < 0 or row >= len(self._cells_data):
            return
        del self._cells_data[row]
        self._prev_list_row = -1
        self._refresh_list(select_row=max(0, row - 1))

    def _rename_current_cell(self, *_args: Any) -> None:
        row = self._cell_list.currentRow()
        if row < 0 or row >= len(self._cells_data):
            return
        cell = self._cells_data[row]
        current = str(cell.get("name") or cell.get("id") or f"cell_{row}")
        new_name, ok = QInputDialog.getText(self, "Rename File", "File name:", text=current)
        if not ok:
            return
        new_name = str(new_name or "").strip()
        if not new_name:
            return
        cell["name"] = new_name
        item = self._cell_list.item(row)
        if item is not None:
            item.setText(new_name)
        self._sync_editor_chrome()

    def _on_hidden_cell_name_changed(self, text: str) -> None:
        row = self._cell_list.currentRow()
        if row < 0 or row >= len(self._cells_data):
            return
        new_name = str(text or "").strip()
        if not new_name:
            return
        self._cells_data[row]["name"] = new_name
        item = self._cell_list.item(row)
        if item is not None:
            item.setText(new_name)
        self._sync_editor_chrome()

    def set_status(self, text: str) -> None:
        self._set_status(text)

    def _replace_from_draft(self, data: dict[str, Any]) -> None:
        selected = max(0, self._cell_list.currentRow())
        self._cells_data = [dict(cell) for cell in data.get("cells") or [] if isinstance(cell, dict)]
        sd = data.get("scenario_datasets")
        self._scenario_datasets = list(sd) if isinstance(sd, list) else []
        self._refresh_list(select_row=selected)

    def _flush_editor_to_row(self, row: int) -> None:
        if row < 0 or row >= len(self._cells_data):
            return
        cell = self._cells_data[row]
        cell["code"] = self._code.toPlainText()

    def _on_list_row_changed(self, row: int) -> None:
        previous = self._prev_list_row
        if previous >= 0:
            self._flush_editor_to_row(previous)
        self._prev_list_row = row
        if row < 0 or row >= len(self._cells_data):
            self._code.clear()
            self._sync_editor_chrome()
            return
        cell = self._cells_data[row]
        current_name = str(cell.get("name") or cell.get("id") or f"cell_{row}")
        self._cell_name.blockSignals(True)
        self._cell_name.setText(current_name)
        self._cell_name.blockSignals(False)
        self._code.setPlainText(str(cell.get("code") or ""))
        if self._assets_open:
            self._refresh_asset_list()
        self._sync_editor_chrome()

    def _refresh_list(self, *, select_row: int = 0) -> None:
        self._cell_list.blockSignals(True)
        self._cell_list.clear()
        for index, cell in enumerate(self._cells_data):
            label = str(cell.get("name") or cell.get("id") or f"cell_{index}")
            self._cell_list.addItem(label)
        if self._cells_data:
            self._cell_list.setCurrentRow(min(select_row, len(self._cells_data) - 1))
        self._cell_list.blockSignals(False)
        self._prev_list_row = self._cell_list.currentRow()
        if self._cells_data and self._prev_list_row >= 0:
            self._on_list_row_changed(self._prev_list_row)
        elif not self._cells_data:
            self._code.clear()
        self._sync_editor_chrome()

    @staticmethod
    def _default_cell() -> dict[str, Any]:
        cell_id = uuid.uuid4().hex[:10]
        code = (
            "def run(ctx, prev):\n"
            "    print('datasets', ctx.datasets)\n"
            "    print('active_cell', ctx.active_cell)\n"
            "    return {'data': {'ok': True}}\n"
        )
        return {
            "id": cell_id,
            "name": f"cell_{cell_id}",
            "entry": "run",
            "code": code,
            "datasets": [],
            "pasted_files": [],
        }

    def _set_status(self, text: str) -> None:
        self._status.setText(str(text or ""))
        self.statusChanged.emit(str(text or ""))

    def _sync_editor_chrome(self) -> None:
        if not self._scenario:
            self._assets_open = False
            if hasattr(self, "_sidebar_stack"):
                self._sidebar_stack.setCurrentWidget(self._file_catalog_page)
            self._remove_cell_btn.setEnabled(False)
            return

        row = self._cell_list.currentRow()
        if row < 0 or row >= len(self._cells_data):
            self._assets_open = False
            if hasattr(self, "_sidebar_stack"):
                self._sidebar_stack.setCurrentWidget(self._file_catalog_page)
            self._remove_cell_btn.setEnabled(False)
            return

        cell = self._cells_data[row]
        name = str(cell.get("name") or cell.get("id") or f"cell_{row}").strip() or f"cell_{row}"
        asset_count = len(self._cell_assets(cell))
        if hasattr(self, "_asset_title"):
            self._asset_title.setText(f"2. Internal files, data & code: {name}")
        if hasattr(self, "_sidebar_stack"):
            self._sidebar_stack.setCurrentWidget(
                self._asset_panel if self._assets_open else self._file_catalog_page
            )
        code = self._code.toPlainText() if self._code.isEnabled() else str(cell.get("code") or "")
        line_count = len(code.splitlines()) if code else 0
        if hasattr(self, "_asset_title"):
            self._asset_title.setToolTip(
                f"{asset_count} internal file(s), Code has {line_count} line(s), "
                f"entry: {str(cell.get('entry') or 'run')}"
            )
        self._remove_cell_btn.setEnabled(True)
