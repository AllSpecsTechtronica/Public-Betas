from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .collapsible_section import CollapsibleSection


# Per-field display metadata. step/decimals override the generic defaults derived
# from the schema min/max when a field benefits from a different resolution
# (e.g. learning-rate sliders want more decimals than augmentation knobs).
_FIELD_META: dict[str, dict[str, Any]] = {
    # schedule
    "epochs":       {"label": "Epochs",            "tip": "Total training epochs"},
    "imgsz":        {"label": "Image size",        "tip": "Must be a multiple of 32; guard may clamp"},
    "batch":        {"label": "Batch",             "tip": "-1 = auto; guard may clamp"},
    "workers":      {"label": "DataLoader workers","tip": "Guard may clamp; 0 = main thread only"},
    "patience":     {"label": "Patience",          "tip": "Early-stop patience in epochs"},
    "close_mosaic": {"label": "Close mosaic",      "tip": "Disable mosaic for last N epochs"},
    "save_period":  {"label": "Save period",       "tip": "Checkpoint every N epochs; -1 = end only"},
    # quality stop
    "quality_stop_enabled": {"label": "Enable quality stop", "tip": "Stop YOLO training once validation quality is stable enough"},
    "quality_stop_attempt_mode": {"label": "Attempt mode (probe)", "tip": "One-shot feasibility probe: collapse min/consecutive epochs to 1, force rapid-clear and regression guards on. Use to answer 'can this dataset clear threshold at all?'"},
    "quality_stop_max_time_seconds": {"label": "Max time (seconds)", "tip": "Wall-clock budget; 0 disables. Run exits with verdict if exceeded"},
    "quality_stop_metric":  {"label": "Quality metric",      "tip": "Validation metric used for quality-target stopping"},
    "quality_stop_threshold": {"label": "Quality threshold", "tip": "Stop target, e.g. 0.90 = 90%", "decimals": 3, "step": 0.01},
    "quality_stop_min_epochs": {"label": "Minimum epochs",   "tip": "Do not stop before this many epochs have completed (ignored in attempt mode)"},
    "quality_stop_consecutive_epochs": {"label": "Consecutive epochs", "tip": "Required qualifying epochs before stopping (forced to 1 in attempt mode)"},
    # optimizer
    "optimizer":    {"label": "Optimizer",         "tip": "auto = Ultralytics picks per dataset size"},
    "lr0":          {"label": "lr0",               "tip": "Initial learning rate", "decimals": 6, "step": 0.0001},
    "lrf":          {"label": "lrf",               "tip": "Final lr multiplier",   "decimals": 6, "step": 0.0001},
    "momentum":     {"label": "Momentum",          "decimals": 4, "step": 0.001},
    "weight_decay": {"label": "Weight decay",      "decimals": 6, "step": 0.0001},
    "warmup_epochs":   {"label": "Warmup epochs",  "decimals": 2, "step": 0.1},
    "warmup_momentum": {"label": "Warmup momentum","decimals": 4, "step": 0.01},
    "warmup_bias_lr":  {"label": "Warmup bias lr", "decimals": 5, "step": 0.001},
    "cos_lr":       {"label": "Cosine LR schedule"},
    "amp":          {"label": "Automatic mixed precision"},
    # regularization
    "dropout":         {"label": "Dropout",         "decimals": 3, "step": 0.01},
    "label_smoothing": {"label": "Label smoothing", "decimals": 3, "step": 0.01},
    # augmentation
    "hsv_h":      {"label": "HSV-H",        "decimals": 3, "step": 0.005},
    "hsv_s":      {"label": "HSV-S",        "decimals": 3, "step": 0.01},
    "hsv_v":      {"label": "HSV-V",        "decimals": 3, "step": 0.01},
    "degrees":    {"label": "Rotation deg", "decimals": 1, "step": 1.0},
    "translate":  {"label": "Translate",    "decimals": 3, "step": 0.01},
    "scale":      {"label": "Scale",        "decimals": 3, "step": 0.01},
    "shear":      {"label": "Shear deg",    "decimals": 2, "step": 0.5},
    "perspective":{"label": "Perspective",  "decimals": 6, "step": 0.00005},
    "fliplr":     {"label": "Flip LR prob", "decimals": 2, "step": 0.05},
    "flipud":     {"label": "Flip UD prob", "decimals": 2, "step": 0.05},
    "mosaic":     {"label": "Mosaic",       "decimals": 2, "step": 0.05},
    "mixup":      {"label": "Mixup",        "decimals": 2, "step": 0.05},
    "copy_paste": {"label": "Copy-paste",   "decimals": 2, "step": 0.05},
    "erasing":    {"label": "Erasing",      "decimals": 2, "step": 0.05},
    # reproducibility
    "seed":          {"label": "Seed"},
    "deterministic": {"label": "Deterministic (cudnn.deterministic=True)"},
}


_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Schedule", (
        "epochs", "imgsz", "batch", "workers", "patience", "close_mosaic", "save_period",
    )),
    ("Quality Stop", (
        "quality_stop_enabled", "quality_stop_attempt_mode",
        "quality_stop_metric", "quality_stop_threshold",
        "quality_stop_max_time_seconds",
        "quality_stop_min_epochs", "quality_stop_consecutive_epochs",
    )),
    ("Optimizer", (
        "optimizer", "lr0", "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
        "cos_lr", "amp",
    )),
    ("Regularization", (
        "dropout", "label_smoothing",
    )),
    ("Augmentation", (
        "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "shear", "perspective",
        "fliplr", "flipud",
        "mosaic", "mixup", "copy_paste", "erasing",
    )),
    ("Reproducibility", (
        "seed", "deterministic",
    )),
)


class _FieldEditor:
    """Wraps a single hp editor widget + its label + dirty indicator.

    Owns its own coercion so the panel can collect values without per-type
    branching at collection time.
    """

    def __init__(
        self,
        key: str,
        kind: str,
        validator: Any,
        meta: dict[str, Any],
        on_changed: Callable[[], None],
    ) -> None:
        self.key = key
        self.kind = kind
        self._validator = validator
        self._saved_value: Any = None
        self._saved_present: bool = False
        self._edited: bool = False
        self._on_changed = on_changed
        label_text = str(meta.get("label") or key)
        self.label = QLabel(label_text + ":")
        self.label.setToolTip(str(meta.get("tip") or ""))
        self.dirty_mark = QLabel("")
        self.dirty_mark.setFixedWidth(8)
        self.dirty_mark.setStyleSheet("color: rgba(203, 130, 28, 0.95); font-weight: 700;")

        def _user_edited() -> None:
            self._edited = True
            on_changed()

        self.widget: QWidget
        if kind == "int":
            lo, hi = validator if isinstance(validator, tuple) else (-(10**9), 10**9)
            w = QSpinBox()
            w.setRange(int(lo), int(hi))
            w.setSingleStep(1)
            w.valueChanged.connect(lambda _v: _user_edited())
            self.widget = w
        elif kind == "float":
            lo, hi = validator if isinstance(validator, tuple) else (-1e9, 1e9)
            decimals = int(meta.get("decimals") or 4)
            step = float(meta.get("step") or (10 ** -decimals))
            w = QDoubleSpinBox()
            w.setRange(float(lo), float(hi))
            w.setDecimals(decimals)
            w.setSingleStep(step)
            w.valueChanged.connect(lambda _v: _user_edited())
            self.widget = w
        elif kind == "bool":
            w = QCheckBox()
            w.stateChanged.connect(lambda _v: _user_edited())
            self.widget = w
        elif kind == "str_choices":
            w = QComboBox()
            choices = list(validator) if isinstance(validator, tuple) else []
            for ch in choices:
                w.addItem(str(ch), str(ch))
            w.currentIndexChanged.connect(lambda _v: _user_edited())
            self.widget = w
        else:  # pragma: no cover - defensive
            w = QLabel("[UNSUPPORTED]")
            self.widget = w

    def set_value(self, value: Any, *, mark_saved: bool, saved_present: bool = True) -> None:
        if self.kind == "int":
            try:
                self.widget.blockSignals(True)
                self.widget.setValue(int(value) if value is not None else self.widget.minimum())
            finally:
                self.widget.blockSignals(False)
        elif self.kind == "float":
            try:
                self.widget.blockSignals(True)
                self.widget.setValue(float(value) if value is not None else 0.0)
            finally:
                self.widget.blockSignals(False)
        elif self.kind == "bool":
            try:
                self.widget.blockSignals(True)
                self.widget.setChecked(bool(value))
            finally:
                self.widget.blockSignals(False)
        elif self.kind == "str_choices":
            try:
                self.widget.blockSignals(True)
                idx = self.widget.findData(str(value)) if value is not None else -1
                if idx < 0:
                    idx = 0
                self.widget.setCurrentIndex(idx)
            finally:
                self.widget.blockSignals(False)
        if mark_saved:
            self._saved_value = self.get_value()
            self._saved_present = bool(saved_present)
            self._edited = False
            self.dirty_mark.setText("")

    def get_value(self) -> Any:
        if self.kind == "int":
            return int(self.widget.value())
        if self.kind == "float":
            return float(self.widget.value())
        if self.kind == "bool":
            return bool(self.widget.isChecked())
        if self.kind == "str_choices":
            return str(self.widget.currentData() or "")
        return None

    def is_dirty(self) -> bool:
        if not self._saved_present:
            return self._edited
        return self.get_value() != self._saved_value

    def refresh_dirty_mark(self) -> None:
        self.dirty_mark.setText("*" if self.is_dirty() else "")

    def is_set(self) -> bool:
        """Whether this field was present in the loaded hyperparams dict."""
        return self._saved_present


class HyperparamSuitePanel(QFrame):
    """Grouped editor covering Ultralytics training hyperparameters.

    Payload flows:
      load(hyperparams, schema) -> render current values
      savePressed(payload)       -> emitted when user clicks Save
      resetPressed()             -> emitted when user clicks Reset to saved
    """

    savePressed = pyqtSignal(dict)
    resetPressed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 5, 6, 6)
        outer.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Hyperparameters")
        title.setProperty("isTitle", True)
        title.setStyleSheet("font-weight: 600; border: none;")
        header.addWidget(title, stretch=0)
        header.addStretch(1)
        self._dirty_counter = QLabel("")
        self._dirty_counter.setStyleSheet(
            "border: none; font-size: 10px; color: rgba(203, 130, 28, 0.95);"
        )
        header.addWidget(self._dirty_counter)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._emit_save)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Reset hyperparameters to the saved scenario values.")
        self._reset_btn.clicked.connect(self.resetPressed.emit)
        header.addWidget(self._save_btn)
        header.addWidget(self._reset_btn)
        outer.addLayout(header)

        self._fields: dict[str, _FieldEditor] = {}
        self._sections: list[CollapsibleSection] = []
        self._schema: dict[str, dict[str, Any]] = {}

        # Render a section per tuple in _SECTIONS. Fields whose keys aren't
        # in the schema (older server) get skipped silently so the UI degrades.
        for section_title, field_keys in _SECTIONS:
            section = CollapsibleSection(section_title, expanded=(section_title == "Schedule"))
            body = section.body_layout()
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(3)
            body.addLayout(grid)
            self._sections.append(section)
            outer.addWidget(section)
            # Grid cell indexes are filled dynamically below in _populate_section.
            section._grid = grid  # type: ignore[attr-defined]
            section._keys = field_keys  # type: ignore[attr-defined]

        foot = QHBoxLayout()
        foot.setSpacing(6)
        self._defaults_btn = QPushButton("Clear all")
        self._defaults_btn.setToolTip(
            "Send an empty update with reset=true — scenario reverts to Ultralytics defaults "
            "(only guard_profile is preserved)."
        )
        self._defaults_btn.clicked.connect(self._emit_clear_all)
        foot.addStretch(1)
        foot.addWidget(self._defaults_btn)
        outer.addLayout(foot)

        self._save_btn.setEnabled(False)
        self._reset_btn.setEnabled(False)

    def load(self, hyperparams: dict[str, Any], schema: dict[str, dict[str, Any]]) -> None:
        """Populate the editor from scenario hyperparams + server-provided schema."""
        self._schema = dict(schema or {})
        self._fields.clear()
        # Clear previous rows in each section's grid (we re-render so the order
        # matches the section field list even if schema added/removed keys).
        for section in self._sections:
            grid: QGridLayout = section._grid  # type: ignore[attr-defined]
            while grid.count():
                item = grid.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.hide()
                    w.deleteLater()
            section.setVisible(self._populate_section(section, hyperparams) > 0)
        self._refresh_dirty_state()

    def current_values(self, *, dirty_only: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, field in self._fields.items():
            if dirty_only and not field.is_dirty():
                continue
            if dirty_only and not field.is_set() and not field.is_dirty():
                continue
            out[key] = field.get_value()
        return out

    def _populate_section(self, section: CollapsibleSection, hyperparams: dict[str, Any]) -> int:
        grid: QGridLayout = section._grid  # type: ignore[attr-defined]
        keys: tuple[str, ...] = section._keys  # type: ignore[attr-defined]
        row = 0
        for key in keys:
            schema_entry = self._schema.get(key)
            if not isinstance(schema_entry, dict):
                continue
            kind = str(schema_entry.get("kind") or "")
            if kind not in {"int", "float", "bool", "str_choices"}:
                continue
            # Re-assemble validator from the serialized schema shape.
            if kind == "str_choices":
                validator: Any = tuple(schema_entry.get("choices") or ())
            elif "min" in schema_entry and "max" in schema_entry:
                validator = (schema_entry["min"], schema_entry["max"])
            else:
                validator = None
            meta = _FIELD_META.get(key, {"label": key})
            field = _FieldEditor(key, kind, validator, meta, self._refresh_dirty_state)
            saved_present = key in hyperparams
            value = hyperparams.get(key)
            field.set_value(value, mark_saved=True, saved_present=saved_present)
            grid.addWidget(field.dirty_mark, row, 0, alignment=Qt.AlignmentFlag.AlignTop)
            grid.addWidget(field.label, row, 1, alignment=Qt.AlignmentFlag.AlignTop)
            grid.addWidget(field.widget, row, 2)
            grid.setColumnStretch(2, 1)
            self._fields[key] = field
            row += 1
        return row

    def _refresh_dirty_state(self) -> None:
        dirty = 0
        for field in self._fields.values():
            field.refresh_dirty_mark()
            if field.is_dirty():
                dirty += 1
        if dirty:
            self._dirty_counter.setText(f"{dirty} unsaved")
        else:
            self._dirty_counter.setText("")
        self._save_btn.setEnabled(dirty > 0)
        self._reset_btn.setEnabled(dirty > 0)

    def _emit_save(self) -> None:
        payload = self.current_values(dirty_only=True)
        if not payload:
            return
        self.savePressed.emit(payload)

    def _emit_clear_all(self) -> None:
        # Explicit marker so the controller knows to POST reset=true with empty updates.
        self.savePressed.emit({"__reset_all__": True})
