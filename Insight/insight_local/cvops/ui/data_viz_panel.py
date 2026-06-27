from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtQml import QJSEngine, QJSValue
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from .csv_table_editor import CsvTableEditorDialog, CsvVisualizationWidget


_MAX_ROWS = 100_000
_AUTOLOAD_MAX_ROWS = 20_000

_TEMPLATE_IDS = ["distribution", "correlation", "scatter", "categorical", "missing", "stats"]
_DEFAULT_ACTIVE_TEMPLATE_IDS = ["distribution", "correlation", "scatter"]
_TEMPLATE_LABELS = {
    "distribution": "DISTRIBUTION",
    "correlation": "CORRELATION",
    "scatter": "SCATTER",
    "categorical": "CATEGORICAL",
    "missing": "MISSING DATA",
    "stats": "SUMMARY TABLE",
}

_DEFAULT_CELL_CODE = """// Custom visualization cell
// Available: fingerprint (full investigation result)
// Return JSX or a data array for charting

// Example: top 5 features by standard deviation
const ranked = fingerprint.distributions
  .filter(d => d.std != null)
  .sort((a, b) => b.std - a.std)
  .slice(0, 8);

return ranked.map(d => ({
  name: d.name,
  std: d.std,
  mean: d.mean,
}));"""


class _CsvDropZone(QFrame):
    fileDropped = pyqtSignal(str)
    browseRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("dataVizImportDropZone")
        self.setStyleSheet(
            "QFrame#dataVizImportDropZone {"
            " border: 1px dashed rgba(90,104,98,0.72);"
            " background: rgba(10,14,12,0.90);"
            " border-radius: 12px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)
        self._title = QLabel("Drop CSV here or click to browse")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 12px; color: rgba(232,237,233,0.88);")
        self._sub = QLabel("Auto-detects archetype and runs the investigation fingerprint pipeline")
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub.setWordWrap(True)
        self._sub.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 10px; color: rgba(147,161,161,0.78);")
        layout.addStretch(1)
        layout.addWidget(self._title)
        layout.addWidget(self._sub)
        layout.addStretch(1)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            event.acceptProposedAction()
            self._set_hover(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._set_hover(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[override]
        self._set_hover(False)
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            event.ignore()
            return
        for url in mime.urls():
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in {".csv", ".tsv"}:
                self.fileDropped.emit(str(path))
                event.acceptProposedAction()
                return
        event.ignore()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.browseRequested.emit()
        super().mousePressEvent(event)

    def _set_hover(self, active: bool) -> None:
        if active:
            self.setStyleSheet(
                "QFrame#dataVizImportDropZone {"
                " border: 1px dashed rgba(122,232,96,0.92);"
                " background: rgba(122,232,96,0.10);"
                " border-radius: 12px;"
                "}"
            )
        else:
            self.setStyleSheet(
                "QFrame#dataVizImportDropZone {"
                " border: 1px dashed rgba(90,104,98,0.72);"
                " background: rgba(10,14,12,0.90);"
                " border-radius: 12px;"
                "}"
            )


class _CellSpacePanel(QWidget):
    closeRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._fingerprint: dict[str, Any] = {}
        self._engine = QJSEngine()

        self.setMinimumWidth(420)
        self.setMaximumWidth(420)
        self.setStyleSheet(
            "background: rgba(10,14,12,0.98); border-left: 1px solid rgba(90,104,98,0.62);"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(10, 8, 8, 8)
        header.setSpacing(8)
        tag = QLabel("CELL SPACE")
        tag.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 11px; color: rgba(122,232,96,0.98); font-weight: 700;")
        header.addWidget(tag)
        header.addStretch(1)
        self._run_btn = QPushButton("RUN [Cmd+Enter]")
        self._run_btn.setStyleSheet(
            "QPushButton {"
            " font-family: 'JetBrains Mono'; font-size: 10px;"
            " color: rgba(122,232,96,0.98);"
            " border: 1px solid rgba(122,232,96,0.45);"
            " background: rgba(122,232,96,0.10);"
            " padding: 4px 10px;"
            "}"
        )
        self._run_btn.clicked.connect(self.run_cell)
        header.addWidget(self._run_btn)
        close_btn = QPushButton("x")
        close_btn.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 12px;")
        close_btn.clicked.connect(self.closeRequested.emit)
        header.addWidget(close_btn)
        outer.addLayout(header)

        hint = QLabel("// fingerprint.distributions, .corrPairs, .issues, .raw, .numCols, .catCols")
        hint.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 9px; color: rgba(147,161,161,0.82); padding: 2px 10px 6px;")
        outer.addWidget(hint)

        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(2)
        outer.addWidget(split, stretch=1)

        self._editor = QPlainTextEdit()
        self._editor.setStyleSheet(
            "background: rgba(5,8,7,0.96); color: rgba(232,237,233,0.95);"
            "font-family: 'JetBrains Mono'; font-size: 12px;"
            "border-top: 1px solid rgba(90,104,98,0.62); border-bottom: 1px solid rgba(90,104,98,0.62);"
        )
        self._editor.setPlainText(_DEFAULT_CELL_CODE)
        split.addWidget(self._editor)

        output_wrap = QWidget()
        output_layout = QVBoxLayout(output_wrap)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(0)
        output_label = QLabel("OUTPUT")
        output_label.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 9px; color: rgba(147,161,161,0.82); padding: 4px 10px;")
        output_layout.addWidget(output_label)

        self._out_stack = QStackedWidget()
        output_layout.addWidget(self._out_stack, stretch=1)

        self._out_text = QTextEdit()
        self._out_text.setReadOnly(True)
        self._out_text.setStyleSheet(
            "background: rgba(17,24,21,0.92); color: rgba(232,237,233,0.88);"
            "font-family: 'JetBrains Mono'; font-size: 11px;"
            "border: 0;"
        )
        self._out_stack.addWidget(self._out_text)

        chart_host = QWidget()
        chart_layout = QVBoxLayout(chart_host)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)
        self._out_fig = Figure(figsize=(4.2, 2.4), facecolor="#050807")
        self._out_canvas = FigureCanvasQTAgg(self._out_fig)
        chart_layout.addWidget(self._out_canvas)
        self._out_stack.addWidget(chart_host)
        split.addWidget(output_wrap)
        split.setSizes([380, 240])

        shortcut = QShortcut(QKeySequence("Ctrl+Return"), self._editor)
        shortcut.activated.connect(self.run_cell)
        shortcut2 = QShortcut(QKeySequence("Meta+Return"), self._editor)
        shortcut2.activated.connect(self.run_cell)

    def set_fingerprint(self, fingerprint: dict[str, Any]) -> None:
        self._fingerprint = dict(fingerprint or {})

    def run_cell(self) -> None:
        code = str(self._editor.toPlainText() or "").strip()
        if not code:
            self._render_raw("Empty cell")
            return
        payload = json.dumps(self._fingerprint or {}, ensure_ascii=False)
        script = f"const fingerprint = {payload};\n(function(){{\n{code}\n}})();"
        result = self._engine.evaluate(script)
        if result.isError():
            line = result.property("lineNumber").toInt() if result.property("lineNumber").isNumber() else 0
            self._render_raw(f"{result.toString()} (line {line})")
            return
        py_value = self._qjs_to_py(result)
        if isinstance(py_value, list) and py_value and isinstance(py_value[0], dict):
            if self._render_chart(py_value):
                return
        self._render_raw(py_value)

    def _qjs_to_py(self, value: QJSValue) -> Any:
        try:
            variant = value.toVariant()
            if variant is not None:
                return variant
        except Exception:
            pass

        if value.isNull() or value.isUndefined():
            return None
        if value.isBool():
            return bool(value.toBool())
        if value.isNumber():
            return float(value.toNumber())
        if value.isString():
            return str(value.toString())
        if value.isArray():
            length = int(value.property("length").toInt())
            return [self._qjs_to_py(value.property(i)) for i in range(length)]
        if value.isObject():
            keys = value.property("__keys").toVariant()
            if not isinstance(keys, list):
                probe = self._engine.evaluate("Object.keys")
                if probe.isCallable():
                    try:
                        keys_value = probe.call([value])
                        keys = keys_value.toVariant()
                    except Exception:
                        keys = []
            result: dict[str, Any] = {}
            for key in keys or []:
                skey = str(key)
                result[skey] = self._qjs_to_py(value.property(skey))
            return result
        return str(value.toString())

    def _render_chart(self, rows: list[dict[str, Any]]) -> bool:
        if not rows:
            return False
        sample = rows[0]
        label_key = None
        for key, val in sample.items():
            if isinstance(val, str):
                label_key = key
                break
        numeric_keys = [key for key, val in sample.items() if isinstance(val, (int, float))]
        if not numeric_keys:
            return False

        labels = [str(item.get(label_key, i + 1)) for i, item in enumerate(rows)] if label_key else [str(i + 1) for i in range(len(rows))]
        n = len(rows)
        self._out_fig.clear()
        ax = self._out_fig.add_subplot(111)
        ax.set_facecolor("#0a0e0c")
        self._out_fig.patch.set_facecolor("#050807")
        for spine in ax.spines.values():
            spine.set_color("#3f4c46")
        ax.tick_params(colors="#a8b4ac", labelsize=8)
        ax.grid(True, color="#2d3833", linestyle="--", linewidth=0.5, alpha=0.6)

        width = 0.8 / max(1, len(numeric_keys))
        x_base = list(range(n))
        palette = ["#5BE872", "#4D9FFF", "#FFB547", "#22D3EE", "#FF4D4D", "#A78BFA"]
        for idx, key in enumerate(numeric_keys):
            values = [float(item.get(key) or 0.0) for item in rows]
            xs = [x + (idx * width) for x in x_base]
            ax.bar(xs, values, width=width, label=key, color=palette[idx % len(palette)], alpha=0.78)

        if n <= 24:
            center_offset = (len(numeric_keys) - 1) * width / 2.0
            ax.set_xticks([x + center_offset for x in x_base])
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        else:
            ax.set_xticks([])
        if len(numeric_keys) > 1:
            ax.legend(fontsize=7, facecolor="#0a0e0c", edgecolor="#3f4c46", labelcolor="#d6ddd8")
        ax.set_title("Cell Space Output", color="#d6ddd8", fontsize=11)
        self._out_canvas.draw_idle()
        self._out_stack.setCurrentIndex(1)
        return True

    def _render_raw(self, value: Any) -> None:
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, indent=2, ensure_ascii=False)
            except Exception:
                text = str(value)
        self._out_text.setPlainText(text)
        self._out_stack.setCurrentIndex(0)


class DataVizPanel(QWidget):
    """Inline local data visualizer with explicit IMPORT -> INVESTIGATE flow."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("dashboardNativePanel")
        self.setMinimumWidth(0)
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
        self.setSizePolicy(policy)
        self._current_path: Optional[Path] = None
        self._current_max_rows = _MAX_ROWS
        self._fingerprint: dict[str, Any] = {}
        self._headers: list[str] = []
        self._data_rows: list[list[str]] = []
        self._cell_space_open: bool = False

        self._template_buttons: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self._mode_import_btn = QPushButton("IMPORT")
        self._mode_import_btn.clicked.connect(lambda: self._set_mode("import"))
        self._mode_investigate_btn = QPushButton("INVESTIGATE")
        self._mode_investigate_btn.clicked.connect(lambda: self._set_mode("investigate"))
        mode_row.addWidget(self._mode_import_btn)
        mode_row.addWidget(self._mode_investigate_btn)
        mode_row.addStretch(1)
        mode_tag = QLabel("cvLayer // Dataset Investigator")
        mode_tag.setStyleSheet("font-size: 9px; color: rgba(147,161,161,0.8);")
        mode_row.addWidget(mode_tag)
        layout.addLayout(mode_row)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, stretch=1)

        self._import_page = self._build_import_page()
        self._investigate_page = self._build_investigate_page()
        self._stack.addWidget(self._import_page)
        self._stack.addWidget(self._investigate_page)

        self._set_mode("import")

    # ------------------------------------------------------------------
    # Public API (called by Window/Catalog)
    # ------------------------------------------------------------------

    def set_scenario_csv(self, csv_rel: str, root_dir: str) -> None:
        csv_rel = str(csv_rel or "").strip()
        if not csv_rel:
            self.clear()
            return
        p = Path(csv_rel)
        if not p.is_absolute():
            p = Path(root_dir) / p
        self.set_csv_path(p, max_rows=_AUTOLOAD_MAX_ROWS)

    def set_csv_path(self, path: Path, *, max_rows: int = _MAX_ROWS) -> None:
        self._current_path = Path(path).resolve()
        self._current_max_rows = int(max_rows)
        self._reload_btn.setEnabled(True)
        self._analyze_btn.setEnabled(True)
        self._editor_btn.setEnabled(True)
        self._load_and_refresh()
        self._set_mode("investigate")

    def set_data_source_path(self, path: Path, *, max_rows: int = _AUTOLOAD_MAX_ROWS) -> None:
        source = Path(path).expanduser()
        csv_path = self._resolve_csv_candidate(source)
        if csv_path is None:
            self.clear()
            self._set_banner(f"No CSV data source found for: {source}", error=False)
            return
        self.set_csv_path(csv_path, max_rows=max_rows)

    def clear(self) -> None:
        self._current_path = None
        self._fingerprint = {}
        self._headers = []
        self._data_rows = []
        self._reload_btn.setEnabled(False)
        self._analyze_btn.setEnabled(False)
        self._editor_btn.setEnabled(False)
        self._set_banner("")
        self._stats_bar.setText("No dataset loaded")
        self._issues.clear()
        self._issues.setVisible(False)
        self._viz.refresh_from([], [])
        self._cell_space.set_fingerprint({})
        self._refresh_sidebar()
        self._set_mode("import")

    # ------------------------------------------------------------------
    # UI builders
    # ------------------------------------------------------------------

    def _build_import_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(10)
        center_layout.addSpacing(18)

        title = QLabel("Import your data")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 34px; font-weight: 300; color: rgba(232,237,233,0.96);")
        center_layout.addWidget(title)

        sub = QLabel("CSV files supported. Drag and drop or click to browse.")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 11px; color: rgba(147,161,161,0.80);")
        center_layout.addWidget(sub)

        self._drop_zone = _CsvDropZone()
        self._drop_zone.setMinimumSize(460, 220)
        self._drop_zone.setMaximumWidth(560)
        self._drop_zone.fileDropped.connect(self._on_import_drop)
        self._drop_zone.browseRequested.connect(self._open_csv_dialog)

        drop_wrap = QHBoxLayout()
        drop_wrap.addStretch(1)
        drop_wrap.addWidget(self._drop_zone)
        drop_wrap.addStretch(1)
        center_layout.addLayout(drop_wrap)

        self._import_error = QLabel("")
        self._import_error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._import_error.setWordWrap(True)
        self._import_error.setVisible(False)
        self._import_error.setStyleSheet(
            "font-family: 'JetBrains Mono'; font-size: 11px; color: rgba(220,50,47,0.95);"
            "padding: 8px 12px;"
        )
        center_layout.addWidget(self._import_error)

        demo_btn = QPushButton("or load demo dataset")
        demo_btn.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 11px; padding: 6px 16px;")
        demo_btn.clicked.connect(self._load_demo_dataset)
        demo_wrap = QHBoxLayout()
        demo_wrap.addStretch(1)
        demo_wrap.addWidget(demo_btn)
        demo_wrap.addStretch(1)
        center_layout.addLayout(demo_wrap)

        center_layout.addStretch(1)
        outer.addWidget(center, stretch=1)
        return page

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().minimumSizeHint()
        return QSize(0, hint.height())

    def _build_investigate_page(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        button_row.setContentsMargins(0, 0, 0, 0)

        self._banner = QLabel("")
        self._banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._banner.setStyleSheet("font-size: 10px; color: #dc322f;")
        self._banner.setWordWrap(False)
        self._banner.setVisible(False)
        button_row.addWidget(self._banner, stretch=1)
        button_row.addStretch(1)

        open_btn = QPushButton("Open CSV…")
        open_btn.clicked.connect(self._open_csv_dialog)
        button_row.addWidget(open_btn)

        self._reload_btn = QPushButton("Reload")
        self._reload_btn.setEnabled(False)
        self._reload_btn.clicked.connect(self._reload)
        button_row.addWidget(self._reload_btn)

        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(self._analyze_current_dataset)
        button_row.addWidget(self._analyze_btn)

        self._editor_btn = QPushButton("Full Editor…")
        self._editor_btn.setEnabled(False)
        self._editor_btn.clicked.connect(self._open_full_editor)
        button_row.addWidget(self._editor_btn)

        outer.addLayout(button_row)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(True)
        split.setHandleWidth(2)
        outer.addWidget(split, stretch=1)
        self._investigate_split = split

        # Left sidebar (fixed 260)
        left = QWidget()
        left.setMinimumWidth(260)
        left.setMaximumWidth(260)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        self._sidebar_file = QLabel("no dataset")
        self._sidebar_file.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 11px; color: rgba(232,237,233,0.92);")
        left_layout.addWidget(self._sidebar_file)

        health_wrap = QVBoxLayout()
        health_wrap.setSpacing(3)
        hlabel = QLabel("HEALTH")
        hlabel.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 9px; color: rgba(147,161,161,0.82);")
        self._sidebar_health = QLabel("100%")
        self._sidebar_health.setStyleSheet(
            "font-family: 'JetBrains Mono'; font-size: 12px; color: rgba(122,232,96,0.96);"
            "border: 1px solid rgba(122,232,96,0.45); background: rgba(122,232,96,0.10); padding: 3px 8px;"
        )
        health_wrap.addWidget(hlabel)
        health_wrap.addWidget(self._sidebar_health)
        left_layout.addLayout(health_wrap)

        self._sidebar_shape = QLabel("0 rows x 0 cols")
        self._sidebar_shape.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 11px; color: rgba(232,237,233,0.85);")
        left_layout.addWidget(self._sidebar_shape)

        self._sidebar_types = QLabel("0 numeric / 0 categorical")
        self._sidebar_types.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 10px; color: rgba(147,161,161,0.82);")
        left_layout.addWidget(self._sidebar_types)

        ff_label = QLabel("FEATURE FOCUS")
        ff_label.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 9px; color: rgba(147,161,161,0.82);")
        left_layout.addWidget(ff_label)
        self._feature_focus = QComboBox()
        self._feature_focus.currentTextChanged.connect(self._on_feature_focus_changed)
        self._feature_focus.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 11px;")
        left_layout.addWidget(self._feature_focus)

        self._sidebar_issue_label = QLabel("ISSUES (0)")
        self._sidebar_issue_label.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 9px; color: rgba(147,161,161,0.82);")
        left_layout.addWidget(self._sidebar_issue_label)

        self._sidebar_issue_wrap = QWidget()
        self._sidebar_issue_list = QVBoxLayout(self._sidebar_issue_wrap)
        self._sidebar_issue_list.setContentsMargins(0, 0, 0, 0)
        self._sidebar_issue_list.setSpacing(4)
        left_layout.addWidget(self._sidebar_issue_wrap)

        tmpl_label = QLabel("TEMPLATES")
        tmpl_label.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 9px; color: rgba(147,161,161,0.82);")
        left_layout.addWidget(tmpl_label)

        for template_id in _TEMPLATE_IDS:
            btn = QPushButton(_TEMPLATE_LABELS[template_id])
            btn.setCheckable(True)
            btn.setChecked(template_id in _DEFAULT_ACTIVE_TEMPLATE_IDS)
            btn.clicked.connect(self._on_template_toggle)
            btn.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 10px; text-align: left; padding: 5px 8px;")
            self._template_buttons[template_id] = btn
            left_layout.addWidget(btn)

        self._cell_toggle = QPushButton("CELL SPACE")
        self._cell_toggle.setCheckable(True)
        self._cell_toggle.clicked.connect(lambda checked: self._set_cell_space_visible(bool(checked)))
        self._cell_toggle.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 10px; text-align: left; padding: 6px 8px;")
        left_layout.addWidget(self._cell_toggle)

        left_layout.addStretch(1)
        split.addWidget(left)

        # Center visualizer
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        self._stats_bar = QLabel("No dataset loaded")
        self._stats_bar.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._stats_bar.setWordWrap(True)
        self._stats_bar.setStyleSheet(
            "font-size: 10px; color: rgba(168,180,172,0.95);"
            "padding: 6px 8px; border-radius: 3px;"
            "background: rgba(10,14,12,0.88); border: 1px solid rgba(90,104,98,0.62);"
        )
        center_layout.addWidget(self._stats_bar)

        self._issues = QLabel("")
        self._issues.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._issues.setWordWrap(True)
        self._issues.setVisible(False)
        self._issues.setStyleSheet(
            "font-size: 10px; color: rgba(220,50,47,0.95);"
            "padding: 5px 8px; border-radius: 3px;"
            "background: rgba(220,50,47,0.08); border: 1px solid rgba(220,50,47,0.25);"
        )
        center_layout.addWidget(self._issues)

        self._viz = CsvVisualizationWidget()
        self._viz.setMinimumHeight(460)
        self._viz.set_controls_visible(False)
        self._viz.set_deck_templates(_DEFAULT_ACTIVE_TEMPLATE_IDS)
        self._viz.cellSpaceToggleRequested.connect(self._set_cell_space_visible)
        center_layout.addWidget(self._viz, stretch=1)

        split.addWidget(center)

        # Right Cell Space panel
        self._cell_space = _CellSpacePanel()
        self._cell_space.closeRequested.connect(lambda: self._set_cell_space_visible(False))
        split.addWidget(self._cell_space)
        self._cell_space.setVisible(False)

        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setCollapsible(0, True)
        split.setCollapsible(1, False)
        split.setCollapsible(2, True)
        split.setSizes([260, 1120, 0])

        return page

    # ------------------------------------------------------------------
    # Internal flow
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str) -> None:
        is_import = mode == "import"
        self._stack.setCurrentWidget(self._import_page if is_import else self._investigate_page)
        self._mode_import_btn.setStyleSheet(
            "font-family: 'JetBrains Mono'; font-size: 10px; padding: 6px 12px;"
            + ("color: rgba(122,232,96,0.96); border: 1px solid rgba(122,232,96,0.45); background: rgba(122,232,96,0.10);" if is_import else "")
        )
        enabled_investigate = bool(self._headers)
        self._mode_investigate_btn.setEnabled(enabled_investigate)
        self._mode_investigate_btn.setStyleSheet(
            "font-family: 'JetBrains Mono'; font-size: 10px; padding: 6px 12px;"
            + ("color: rgba(122,232,96,0.96); border: 1px solid rgba(122,232,96,0.45); background: rgba(122,232,96,0.10);" if (not is_import and enabled_investigate) else "")
        )

    def _on_import_drop(self, path: str) -> None:
        self.set_csv_path(Path(path))

    def _on_template_toggle(self) -> None:
        active = [key for key, btn in self._template_buttons.items() if btn.isChecked()]
        if not active:
            # keep at least one template active
            first = _TEMPLATE_IDS[0]
            self._template_buttons[first].setChecked(True)
            active = [first]
        self._viz.set_deck_templates(active)

    def _on_feature_focus_changed(self, name: str) -> None:
        value = str(name or "").strip()
        if value:
            self._viz.set_feature_focus(value)

    def _set_cell_space_visible(self, visible: bool) -> None:
        target = bool(visible)
        self._cell_space_open = target
        self._cell_space.setVisible(target)
        self._cell_toggle.blockSignals(True)
        self._cell_toggle.setChecked(target)
        self._cell_toggle.blockSignals(False)
        self._viz.set_cell_space_checked(target)

        if target:
            self._investigate_split.setSizes([260, 840, 420])
        else:
            self._investigate_split.setSizes([260, 1200, 0])

    def _set_banner(self, message: str, *, error: bool = True) -> None:
        if not message:
            self._banner.clear()
            self._banner.setVisible(False)
            self._import_error.clear()
            self._import_error.setVisible(False)
            return
        color = "#dc322f" if error else "rgba(147,161,161,0.85)"
        self._banner.setStyleSheet(f"font-size: 10px; color: {color};")
        self._banner.setText(message)
        self._banner.setVisible(True)
        self._import_error.setStyleSheet(f"font-family: 'JetBrains Mono'; font-size: 11px; color: {color}; padding: 8px 12px;")
        self._import_error.setText(message)
        self._import_error.setVisible(True)

    def _resolve_csv_candidate(self, source: Path) -> Optional[Path]:
        try:
            source = source.resolve()
        except OSError:
            source = source.absolute()
        if source.is_file():
            return source if source.suffix.lower() in {".csv", ".tsv"} else None
        if not source.is_dir():
            return None
        direct = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}]
        if direct:
            preferred_names = ("data.csv", "dataset.csv", "train.csv")
            for name in preferred_names:
                match = next((p for p in direct if p.name.lower() == name), None)
                if match is not None:
                    return match
            return sorted(direct, key=lambda p: p.name.lower())[0]
        try:
            nested = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".tsv"}]
        except OSError:
            nested = []
        return sorted(nested, key=lambda p: (len(p.relative_to(source).parts), p.as_posix().lower()))[0] if nested else None

    def _load_and_refresh(self) -> None:
        if not self._current_path:
            return
        if not self._current_path.is_file():
            self._set_banner(f"[ERROR] File not found: {self._current_path}")
            self._viz.refresh_from([], [])
            return

        rows: list[list[str]] = []
        truncated = False
        try:
            with self._current_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.reader(handle)
                for i, row in enumerate(reader):
                    if i >= self._current_max_rows:
                        truncated = True
                        break
                    rows.append([str(c) for c in row])
        except Exception as exc:
            self._set_banner(f"[ERROR] Could not read file: {exc}")
            self._viz.refresh_from([], [])
            return

        if not rows:
            self._set_banner(f"{self._current_path.name} — empty file", error=False)
            self._viz.refresh_from([], [], source=self._current_path.name)
            self._headers = []
            self._data_rows = []
            self._fingerprint = {}
            self._refresh_sidebar()
            return

        headers = [str(h or "").strip() for h in rows[0]]
        data_rows = rows[1:]
        self._load_from_rows(headers, data_rows, source_name=self._current_path.name, truncated=truncated)

    def _load_from_rows(self, headers: list[str], data_rows: list[list[str]], *, source_name: str, truncated: bool = False) -> None:
        self._headers = list(headers)
        self._data_rows = [list(row) for row in data_rows]
        source = source_name
        if truncated:
            source += f" (first {self._current_max_rows:,})"

        self._fingerprint = self._build_simple_fingerprint(self._headers, self._data_rows)
        self._viz.refresh_from(self._headers, self._data_rows, source=source)
        self._viz.set_deck_templates([key for key, btn in self._template_buttons.items() if btn.isChecked()])

        self._set_fingerprint_summary(self._fingerprint, truncated=truncated)
        self._cell_space.set_fingerprint(self._fingerprint)
        self._refresh_sidebar()
        self._set_banner("")
        self._set_mode("investigate")

    def _reload(self) -> None:
        if self._current_path:
            self._load_and_refresh()

    def _open_csv_dialog(self) -> None:
        start = str(self._current_path.parent) if self._current_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV for Visualization", start,
            "CSV / TSV Files (*.csv *.tsv);;All Files (*)"
        )
        if path:
            self.set_csv_path(Path(path))

    def _open_full_editor(self) -> None:
        dlg = CsvTableEditorDialog(csv_path=self._current_path, parent=self)
        dlg.exec()
        if self._current_path and self._current_path.is_file():
            self._load_and_refresh()

    def _analyze_current_dataset(self) -> None:
        if not self._current_path or not self._current_path.is_file():
            return
        self._analyze_btn.setEnabled(False)
        self._set_banner("Analyzing dataset fingerprint locally…", error=False)
        try:
            raw = self._run_fingerprint(self._current_path)
            if raw:
                self._fingerprint = self._normalize_external_fingerprint(raw, headers=self._headers, data_rows=self._data_rows)
                self._set_fingerprint_summary(self._fingerprint)
                self._cell_space.set_fingerprint(self._fingerprint)
                self._refresh_sidebar()
                self._set_banner("")
            else:
                self._set_banner("Advanced fingerprint unavailable; using built-in profile.", error=False)
        finally:
            self._analyze_btn.setEnabled(True)

    def _load_demo_dataset(self) -> None:
        random.seed(42)
        headers = ["Car ID", "Brand", "Model", "Year", "Engine Size", "Fuel Type", "Transmission", "Mileage", "Doors", "Price"]
        brands = ["Toyota", "Ford", "BMW", "Honda", "Mercedes", "Tesla", "Audi"]
        fuels = ["Gasoline", "Diesel", "Hybrid", "Electric"]
        transmissions = ["Automatic", "Manual"]
        rows: list[list[str]] = []
        for i in range(2500):
            year = random.randint(2000, 2023)
            engine = round(random.uniform(1.2, 5.2), 1)
            mileage = random.randint(15, 300000)
            price = max(1800, int(random.gauss(26000, 18000) + (year - 2000) * 850 - mileage * 0.03 + engine * 1300))
            rows.append([
                str(i + 1),
                random.choice(brands),
                f"Model-{random.randint(1, 24)}",
                str(year),
                str(engine),
                random.choice(fuels),
                random.choice(transmissions),
                str(mileage),
                str(random.choice([2, 3, 4, 5])),
                str(price),
            ])

        self._current_path = Path("demo_dataset.csv")
        self._current_max_rows = len(rows)
        self._reload_btn.setEnabled(True)
        self._analyze_btn.setEnabled(True)
        self._editor_btn.setEnabled(False)
        self._load_from_rows(headers, rows, source_name="demo_dataset.csv", truncated=False)

    def _run_fingerprint(self, path: Path) -> dict[str, Any] | None:
        try:
            root = Path(__file__).resolve().parents[4] / "mlops" / "forecasting" / "visualizationPrimitves"
            init_file = root / "__init__.py"
            if not init_file.is_file():
                return None
            if "dataset_investigator" not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    "dataset_investigator",
                    init_file,
                    submodule_search_locations=[str(root)],
                )
                if spec is None or spec.loader is None:
                    return None
                module = importlib.util.module_from_spec(spec)
                sys.modules["dataset_investigator"] = module
                spec.loader.exec_module(module)
            else:
                module = sys.modules["dataset_investigator"]
            investigate = getattr(module, "investigate", None)
            if investigate is None:
                return None
            fp = investigate(str(path), max_samples=50000, embedding_dim=2, random_state=42)
            if hasattr(fp, "to_dict"):
                return dict(fp.to_dict())
            if isinstance(fp, dict):
                return fp
            return None
        except Exception:
            return None

    def _normalize_external_fingerprint(self, payload: dict[str, Any], *, headers: list[str], data_rows: list[list[str]]) -> dict[str, Any]:
        out = self._build_simple_fingerprint(headers, data_rows)
        out["archetype"] = str(payload.get("archetype") or out.get("archetype") or "TABULAR")
        out["quality_score"] = float(payload.get("quality_score") or out.get("quality_score") or 0.0)
        if isinstance(payload.get("issues"), list):
            out["issues"] = payload.get("issues") or out["issues"]
        if isinstance(payload.get("quality_issues"), list) and not out.get("issues"):
            issues = []
            for item in payload.get("quality_issues") or []:
                if isinstance(item, dict):
                    issues.append(
                        {
                            "severity": str(item.get("severity") or "info").lower(),
                            "category": str(item.get("category") or "quality"),
                            "message": str(item.get("message") or "").strip(),
                        }
                    )
            out["issues"] = issues
        for key, src in (("corrPairs", "corrPairs"), ("corrPairs", "corr_pairs"), ("distributions", "distributions"), ("raw", "raw")):
            value = payload.get(src)
            if value is not None:
                out[key] = value
        return out

    def _build_simple_fingerprint(self, headers: list[str], rows: list[list[str]]) -> dict[str, Any]:
        n_rows = len(rows)
        n_cols = len(headers)
        raw_rows = [
            {headers[i]: (row[i] if i < len(row) else "") for i in range(n_cols)}
            for row in rows[:5000]
        ]

        missing_tokens = {"", "na", "n/a", "null", "none", "nan", "?"}

        def _is_missing(value: str) -> bool:
            return str(value or "").strip().lower() in missing_tokens

        def _parse_number(value: str) -> Optional[float]:
            text = str(value or "").strip().replace(",", "")
            if _is_missing(text):
                return None
            try:
                parsed = float(text)
                if not math.isfinite(parsed):
                    return None
                return parsed
            except Exception:
                return None

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        distributions: list[dict[str, Any]] = []
        cat_distributions: list[dict[str, Any]] = []

        col_values: dict[str, list[str]] = {}
        for idx, name in enumerate(headers):
            vals = [row[idx] if idx < len(row) else "" for row in rows]
            col_values[name] = vals

        for name in headers:
            vals = col_values[name]
            non_missing = [v for v in vals if not _is_missing(v)]
            nums = [n for n in (_parse_number(v) for v in non_missing) if n is not None]
            numeric_ratio = (len(nums) / len(non_missing)) if non_missing else 0.0
            missing = max(0, len(vals) - len(non_missing))
            if len(nums) >= 5 and numeric_ratio >= 0.60:
                numeric_cols.append(name)
                sorted_vals = sorted(nums)
                mean = sum(sorted_vals) / max(1, len(sorted_vals))
                std = math.sqrt(sum((x - mean) ** 2 for x in sorted_vals) / max(1, len(sorted_vals)))
                def _q(p: float) -> float:
                    if not sorted_vals:
                        return 0.0
                    pos = (len(sorted_vals) - 1) * p
                    base = int(math.floor(pos))
                    rest = pos - base
                    nxt = sorted_vals[min(len(sorted_vals) - 1, base + 1)]
                    return sorted_vals[base] + rest * (nxt - sorted_vals[base])
                if std > 0:
                    zvals = [(x - mean) / std for x in sorted_vals]
                    skew = sum(z ** 3 for z in zvals) / max(1, len(zvals))
                    kurt = sum(z ** 4 for z in zvals) / max(1, len(zvals)) - 3.0
                else:
                    skew = 0.0
                    kurt = 0.0
                distributions.append(
                    {
                        "name": name,
                        "dtype": "float64",
                        "count": n_rows,
                        "missing": missing,
                        "missing_pct": round((missing / max(1, n_rows)) * 100.0, 2),
                        "unique": len({round(v, 8) for v in sorted_vals}),
                        "mean": round(mean, 3),
                        "std": round(std, 3),
                        "min": round(sorted_vals[0], 6),
                        "q25": round(_q(0.25), 6),
                        "median": round(_q(0.50), 6),
                        "q75": round(_q(0.75), 6),
                        "max": round(sorted_vals[-1], 6),
                        "skewness": round(skew, 3),
                        "kurtosis": round(kurt, 3),
                    }
                )
            else:
                categorical_cols.append(name)
                ctr = Counter(str(v).strip() for v in non_missing if str(v).strip())
                den = max(1, len(non_missing))
                top = [
                    {"value": k, "count": int(c), "pct": round((c / den) * 100.0, 2)}
                    for k, c in ctr.most_common(12)
                ]
                cat_distributions.append(
                    {
                        "name": name,
                        "dtype": "object",
                        "count": n_rows,
                        "missing": missing,
                        "missing_pct": round((missing / max(1, n_rows)) * 100.0, 2),
                        "unique": len(ctr),
                        "top_values": top,
                    }
                )

        corr_pairs: list[dict[str, Any]] = []
        for i, left in enumerate(numeric_cols[:15]):
            for right in numeric_cols[i + 1:15]:
                xs: list[float] = []
                ys: list[float] = []
                left_vals = col_values[left]
                right_vals = col_values[right]
                for lv, rv in zip(left_vals, right_vals):
                    x = _parse_number(lv)
                    y = _parse_number(rv)
                    if x is None or y is None:
                        continue
                    xs.append(x)
                    ys.append(y)
                if len(xs) < 10:
                    continue
                mx = sum(xs) / len(xs)
                my = sum(ys) / len(ys)
                sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / len(xs))
                sy = math.sqrt(sum((y - my) ** 2 for y in ys) / len(ys))
                if sx <= 0 or sy <= 0:
                    continue
                corr = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) * sx * sy)
                if abs(corr) >= 0.3:
                    corr_pairs.append(
                        {
                            "feature_1": left,
                            "feature_2": right,
                            "correlation": round(corr, 4),
                        }
                    )
        corr_pairs.sort(key=lambda item: abs(float(item.get("correlation") or 0.0)), reverse=True)

        issues: list[dict[str, Any]] = []
        for item in distributions + cat_distributions:
            missing_pct = float(item.get("missing_pct") or 0.0)
            if missing_pct > 50:
                issues.append({"severity": "critical", "category": "missing", "message": f"'{item['name']}' has {missing_pct:.2f}% missing"})
            elif missing_pct > 10:
                issues.append({"severity": "warning", "category": "missing", "message": f"'{item['name']}' has {missing_pct:.2f}% missing"})

        try:
            dupes = n_rows - len({tuple(row) for row in rows})
        except Exception:
            dupes = 0
        if dupes > 0:
            ratio = dupes / max(1, n_rows)
            issues.append(
                {
                    "severity": "critical" if ratio > 0.05 else "warning",
                    "category": "duplicate",
                    "message": f"{dupes} duplicate rows ({ratio * 100.0:.1f}%)",
                }
            )

        for pair in corr_pairs:
            corr = float(pair.get("correlation") or 0.0)
            if abs(corr) > 0.95:
                issues.append(
                    {
                        "severity": "warning",
                        "category": "leakage",
                        "message": f"'{pair['feature_1']}' and '{pair['feature_2']}' have r={corr:.3f}",
                    }
                )

        embedding: list[dict[str, float]] = []
        if len(numeric_cols) >= 2:
            a = numeric_cols[0]
            b = numeric_cols[1]
            for row in rows:
                i = headers.index(a)
                j = headers.index(b)
                x = _parse_number(row[i] if i < len(row) else "")
                y = _parse_number(row[j] if j < len(row) else "")
                if x is None or y is None:
                    continue
                embedding.append({"x": x, "y": y})
                if len(embedding) >= 2000:
                    break

        quality = 1.0
        for issue in issues:
            sev = str(issue.get("severity") or "info")
            if sev == "critical":
                quality -= 0.15
            elif sev == "warning":
                quality -= 0.05
            else:
                quality -= 0.01
        quality = max(0.0, min(1.0, quality))

        return {
            "archetype": "TABULAR",
            "n_samples": n_rows,
            "n_features": n_cols,
            "numeric_features": len(numeric_cols),
            "categorical_features": len(categorical_cols),
            "distributions": distributions,
            "catDistributions": cat_distributions,
            "corrPairs": corr_pairs,
            "issues": issues,
            "quality_score": round(quality, 3),
            "embedding": embedding,
            "numCols": numeric_cols,
            "catCols": categorical_cols,
            "raw": raw_rows,
        }

    def _set_fingerprint_summary(self, fingerprint: dict[str, Any], *, truncated: bool = False) -> None:
        if not fingerprint:
            self._stats_bar.setText("Rows/columns loaded. Fingerprint unavailable.")
            self._issues.clear()
            self._issues.setVisible(False)
            return

        score = float(fingerprint.get("quality_score") or 0.0)
        archetype = str(fingerprint.get("archetype") or "TABULAR")
        n_samples = int(fingerprint.get("n_samples") or len(self._data_rows))
        n_features = int(fingerprint.get("n_features") or len(self._headers))
        trunc_note = " · preview only" if truncated else ""
        self._stats_bar.setText(
            f"{archetype} · {n_samples:,} rows · {n_features} cols · quality {(score * 100):.1f}%{trunc_note}"
        )

        issues = fingerprint.get("issues") or []
        preview: list[str] = []
        if isinstance(issues, list):
            for item in issues[:5]:
                if isinstance(item, dict):
                    sev = str(item.get("severity") or "info").upper()
                    msg = str(item.get("message") or "").strip()
                    if msg:
                        preview.append(f"[{sev}] {msg}")
        if preview:
            self._issues.setText("\n".join(preview))
            self._issues.setVisible(True)
        else:
            self._issues.clear()
            self._issues.setVisible(False)

    def _refresh_sidebar(self) -> None:
        fp = self._fingerprint or {}
        source_name = self._current_path.name if self._current_path else "no dataset"
        self._sidebar_file.setText(source_name)
        n_rows = int(fp.get("n_samples") or len(self._data_rows))
        n_cols = int(fp.get("n_features") or len(self._headers))
        num = int(fp.get("numeric_features") or len(fp.get("numCols") or []))
        cat = int(fp.get("categorical_features") or len(fp.get("catCols") or []))
        score = float(fp.get("quality_score") or 0.0)

        self._sidebar_shape.setText(f"{n_rows:,} x {n_cols}")
        self._sidebar_types.setText(f"{num} numeric / {cat} categorical")

        pct = max(0, min(100, int(round(score * 100))))
        if score >= 0.8:
            tone = "rgba(122,232,96,0.96)"
            bg = "rgba(122,232,96,0.10)"
        elif score >= 0.5:
            tone = "rgba(245,180,70,0.98)"
            bg = "rgba(245,180,70,0.12)"
        else:
            tone = "rgba(220,50,47,0.98)"
            bg = "rgba(220,50,47,0.12)"
        self._sidebar_health.setText(f"{pct}%")
        self._sidebar_health.setStyleSheet(
            f"font-family: 'JetBrains Mono'; font-size: 12px; color: {tone};"
            f"border: 1px solid {tone}; background: {bg}; padding: 3px 8px;"
        )

        num_cols = [str(item) for item in (fp.get("numCols") or []) if str(item).strip()]
        current = self._feature_focus.currentText().strip()
        self._feature_focus.blockSignals(True)
        self._feature_focus.clear()
        if num_cols:
            self._feature_focus.addItems(num_cols)
            if current in num_cols:
                self._feature_focus.setCurrentText(current)
            else:
                self._feature_focus.setCurrentIndex(0)
        self._feature_focus.setEnabled(bool(num_cols))
        self._feature_focus.blockSignals(False)
        if num_cols:
            self._viz.set_feature_focus(self._feature_focus.currentText().strip())

        self._render_sidebar_issues(fp.get("issues") if isinstance(fp.get("issues"), list) else [])

    def _render_sidebar_issues(self, issues: list[Any]) -> None:
        while self._sidebar_issue_list.count():
            item = self._sidebar_issue_list.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        safe_issues = [item for item in issues if isinstance(item, dict)]
        self._sidebar_issue_label.setText(f"ISSUES ({len(safe_issues)})")
        if not safe_issues:
            row = QLabel("No quality issues detected")
            row.setWordWrap(True)
            row.setStyleSheet(
                "font-family: 'JetBrains Mono'; font-size: 10px; color: rgba(122,232,96,0.92);"
                "padding: 6px 8px; border-radius: 0 3px 3px 12px; background: rgba(10,14,12,0.88);"
                "border-left: 1px solid rgba(122,232,96,0.85);"
            )
            self._sidebar_issue_list.addWidget(row)
            return

        for issue in safe_issues[:6]:
            severity = str(issue.get("severity") or "info").lower()
            message = str(issue.get("message") or "").strip()
            if not message:
                continue
            if severity == "critical":
                tone = "rgba(220,50,47,0.98)"
                bg = "rgba(220,50,47,0.10)"
            elif severity == "warning":
                tone = "rgba(245,180,70,0.98)"
                bg = "rgba(245,180,70,0.10)"
            else:
                tone = "rgba(77,159,255,0.98)"
                bg = "rgba(77,159,255,0.10)"
            row = QLabel(message)
            row.setWordWrap(True)
            row.setToolTip(message)
            row.setStyleSheet(
                "font-family: 'JetBrains Mono'; font-size: 10px; color: rgba(232,237,233,0.88);"
                f"padding: 6px 8px; border-radius: 0 3px 3px 12px; background: {bg};"
                f"border-left: 1px solid {tone};"
            )
            self._sidebar_issue_list.addWidget(row)
