from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    COLOR_SCHEME_CHOICES,
    DEFAULT_CONFIDENCE,
    DEFAULT_FPS,
    DEFAULT_IMG_SIZE,
    DEFAULT_IOU,
    DEFAULT_MAX_DET,
    DEFAULT_MODEL_PATH,
    LOCKED_INFERENCE_IMAGE_SIZE,
    NEW_TRACK_SECONDS,
    OLLAMA_VISION_MODELS,
    PERSISTENT_SECONDS,
    PREVIEW_QUALITY,
    RECOGNITION_AUTO,
    INSIGHT_ANTHROPIC_MODEL,
    INSIGHT_OLLAMA_MODEL,
    INSIGHT_OPENAI_MODEL,
    RECOGNITION_THRESHOLD,
    RECOGNITION_TOP_K,
    TRACK_STALE_FRAMES,
    TRACK_STALE_SECONDS,
    list_model_catalog_names,
    normalize_image_size,
)
from ..privacy import PrivacyStatus, detect_privacy_status
from .theme import contrast_text_hex, current_color_scheme, text_css, theme_rgba
from .neo_button import ParallelogramButton


def _section_style() -> str:
    return (
        f"color: {theme_rgba('accent_dark', 1.0)}; font-size: 9px; font-weight: 700; "
        f"letter-spacing: 2px; padding: 4px 8px; "
        f"background: {theme_rgba('accent_dark', 0.13)}; "
        f"border: 1px dotted {theme_rgba('accent_dark', 0.38)}; border-radius: 0px;"
    )


def _label_style() -> str:
    return (
        f"color: {theme_rgba('accent_dark', 1.0)}; font-size: 10px; font-weight: 600; "
        f"background: {theme_rgba('accent_dark', 0.10)}; "
        f"border: 1px dotted {theme_rgba('accent_dark', 0.30)}; "
        "border-radius: 0px; padding: 2px 6px;"
    )


def _value_style() -> str:
    return (
        f"color: {theme_rgba('accent_dark', 0.92)}; font-size: 10px; font-weight: 600; "
        "border: none; min-width: 38px;"
    )


def _slider_style() -> str:
    return (
        f"QSlider::groove:horizontal {{ height: 2px; background: {theme_rgba('accent_dark', 0.24)}; }}"
        f"QSlider::handle:horizontal {{ width: 8px; height: 8px; margin: -3px 0; "
        f"background: {theme_rgba('accent_dark', 0.82)}; }}"
        f"QSlider::sub-page:horizontal {{ background: {theme_rgba('panel', 0.34)}; }}"
    )


def _sep_style() -> str:
    return f"background: {theme_rgba('accent_dark', 0.18)}; border: none;"


def _toggle_off_style() -> str:
    return (
        f"QPushButton {{ border: 1px solid {theme_rgba('accent_dark', 0.36)}; "
        f"background: {theme_rgba('panel', 0.72)}; "
        f"color: {theme_rgba('accent_dark', 0.72)}; padding: 3px 10px; font-size: 9px; letter-spacing: 1px; }}"
        f"QPushButton:hover {{ color: {theme_rgba('accent_dark', 0.90)}; border-color: {theme_rgba('accent_dark', 0.50)}; }}"
    )


def _toggle_on_style() -> str:
    on_text = text_css(0.98) if current_color_scheme() == "fire" else contrast_text_hex("accent_dark")
    return (
        f"QPushButton {{ border: 1px solid {theme_rgba('pressed', 0.58)}; "
        f"background: {theme_rgba('accent_dark', 0.88)}; "
        f"color: {on_text}; padding: 3px 10px; font-size: 9px; letter-spacing: 1px; }}"
        f"QPushButton:hover {{ color: {on_text}; border-color: {theme_rgba('pressed', 0.72)}; }}"
    )


def _save_button_style() -> str:
    on_text = text_css(0.98) if current_color_scheme() == "fire" else contrast_text_hex("accent_dark")
    return (
        f"QPushButton {{ border: 1px solid {theme_rgba('pressed', 0.58)}; "
        f"background: {theme_rgba('accent_dark', 0.88)}; "
        f"color: {on_text}; padding: 8px 16px; font-size: 10px; "
        "font-weight: 700; letter-spacing: 2px; }"
        f"QPushButton:hover {{ color: {on_text}; "
        f"border-color: {theme_rgba('pressed', 0.72)}; }}"
        f"QPushButton:disabled {{ background: {theme_rgba('accent_dark', 0.42)}; "
        f"color: {theme_rgba('accent_dark', 0.44)}; }}"
    )


def _combo_style() -> str:
    return (
        f"QComboBox {{ background: {theme_rgba('input_fill', 0.92)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.34)}; "
        f"color: {theme_rgba('accent_dark', 0.94)}; padding: 4px 8px; font-size: 10px; }}"
        f"QComboBox:hover {{ background: {theme_rgba('hover', 0.68)}; "
        f"border-color: {theme_rgba('accent_dark', 0.48)}; }}"
        f"QComboBox:focus {{ background: {theme_rgba('input_fill', 0.96)}; "
        f"border: 1px solid {theme_rgba('pressed', 0.62)}; }}"
        f"QComboBox::drop-down {{ border: none; width: 18px; background: {theme_rgba('accent_dark', 0.24)}; }}"
        f"QComboBox QAbstractItemView {{ background: {theme_rgba('input_list', 0.96)}; "
        f"color: {theme_rgba('accent_dark', 0.94)}; border: 1px solid {theme_rgba('accent_dark', 0.36)}; }}"
    )


def _line_edit_style() -> str:
    return (
        f"QLineEdit {{ background: {theme_rgba('input_fill', 0.92)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.34)}; "
        f"color: {theme_rgba('accent_dark', 0.94)}; padding: 4px 8px; font-size: 10px; }}"
        f"QLineEdit:hover {{ background: {theme_rgba('hover', 0.68)}; "
        f"border-color: {theme_rgba('accent_dark', 0.48)}; }}"
        f"QLineEdit:focus {{ background: {theme_rgba('input_fill', 0.96)}; "
        f"border: 1px solid {theme_rgba('pressed', 0.62)}; }}"
    )


def _privacy_safe_style() -> str:
    return f"color: {theme_rgba('accent_dark', 0.86)}; font-size: 10px; font-weight: 700; padding: 4px 0;"


def _privacy_warn_style() -> str:
    return f"color: {theme_rgba('privacy_warn', 0.98)}; font-size: 10px; font-weight: 700; padding: 4px 0;"


def _session_metrics_style() -> str:
    return (
        f"color: {theme_rgba('accent_dark', 0.92)}; font-size: 10px; "
        "font-family: 'JetBrains Mono', 'IBM Plex Mono', 'Menlo', monospace; "
        f"padding: 8px 10px; line-height: 1.45; "
        f"background: {theme_rgba('panel', 0.55)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.24)}; border-radius: 0px;"
    )


def _supervisor_log_style() -> str:
    return (
        f"QTextEdit {{ background: {theme_rgba('accent_dark', 0.42)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.28)}; "
        "color: rgba(20,8,8,0.86); font-size: 10px; }"
    )


def _privacy_guide_style() -> str:
    return (
        f"QTextEdit {{ background: {theme_rgba('accent_dark', 0.32)}; "
        f"border: 1px solid {theme_rgba('accent_dark', 0.24)}; "
        "color: rgba(20,8,8,0.86); font-size: 10px; }"
    )

_OPENAI_MODELS = [
    "gpt-5.4-mini",
    "gpt-4.1-mini",
    "gpt-4.1",
]
_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-latest",
    "claude-3-7-sonnet-latest",
]


class SettingsPanel(QWidget):
    """Minimal transparent settings panel for the data column."""

    ai_provider_changed = pyqtSignal(str)
    ai_model_changed = pyqtSignal(str, str)
    bbox_style_changed = pyqtSignal(str)
    panel_background_style_changed = pyqtSignal(str)
    edge_background_changed = pyqtSignal(bool)
    edge_brightness_changed = pyqtSignal(float)
    heat_tags_changed = pyqtSignal(bool)
    labels_only_changed = pyqtSignal(bool)
    thermal_mode_changed = pyqtSignal(str)
    button_caption_mode_changed = pyqtSignal(str)
    fps_changed = pyqtSignal(int)
    color_scheme_changed = pyqtSignal(str)
    hotkey_peek_changed = pyqtSignal(object)  # emits set[str]

    def __init__(
        self,
        send_message: Callable[[dict[str, Any]], None],
        storage_paths: dict[str, Path | str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._send = send_message
        self._storage_paths = {
            label: Path(path).expanduser().resolve()
            for label, path in (storage_paths or {}).items()
        }
        self.setStyleSheet("background: transparent;")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.viewport().setStyleSheet("background: transparent;")

        inner = QWidget()
        inner.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(inner)
        self._layout.setContentsMargins(10, 4, 10, 10)
        self._layout.setSpacing(2)

        self._sliders: dict[str, tuple[QSlider, QLabel, Callable[[int], str], int]] = {}
        self._combos: dict[str, QComboBox] = {}
        self._toggles: dict[str, QPushButton] = {}
        self._separators: list[QFrame] = []
        self._section_labels: list[QLabel] = []
        self._row_labels: list[QLabel] = []
        self._hotkey_peek_options: set[str] = {"roi", "sidebar"}
        self._hotkey_peek_btns: dict[str, QPushButton] = {}

        self._add_section("LIVE SESSION")
        self._session_metrics = QLabel(
            "Mode and telemetry appear here once the scene is online "
            "(same info that used to show on the video corners)."
        )
        self._session_metrics.setWordWrap(True)
        self._session_metrics.setStyleSheet(_session_metrics_style())
        self._layout.addWidget(self._session_metrics)

        self._add_separator()

        # -- Detection --
        self._add_section("DETECTION")
        model_names = list_model_catalog_names() or [DEFAULT_MODEL_PATH.name]
        default_model = DEFAULT_MODEL_PATH.name if DEFAULT_MODEL_PATH.name in model_names else model_names[0]
        self._add_combo("detector_model", "Detector Model", model_names, default_model)
        self._add_slider("confidence", "Confidence", DEFAULT_CONFIDENCE, 0.05, 0.95, 100, self._fmt_pct)
        self._add_slider("iou", "IOU Threshold", DEFAULT_IOU, 0.05, 0.95, 100, self._fmt_pct)
        self._add_slider(
            "image_size",
            "Inference Size",
            DEFAULT_IMG_SIZE,
            LOCKED_INFERENCE_IMAGE_SIZE,
            LOCKED_INFERENCE_IMAGE_SIZE,
            1,
            self._fmt_int,
            step=1,
        )
        self._add_slider("max_det", "Max Detections", DEFAULT_MAX_DET, 10, 300, 1, self._fmt_int, step=10)

        self._add_separator()

        # -- Tracking --
        self._add_section("TRACKING")
        self._add_slider("stale_seconds", "Stale Timeout", TRACK_STALE_SECONDS, 0.4, 5.0, 10, self._fmt_sec)
        self._add_slider("stale_frames", "Stale Frames", TRACK_STALE_FRAMES, 2, 30, 1, self._fmt_int)
        self._add_slider("new_track_sec", "New Track Time", NEW_TRACK_SECONDS, 0.5, 5.0, 10, self._fmt_sec)
        self._add_slider("persistent_sec", "Persistent Time", PERSISTENT_SECONDS, 1.0, 15.0, 10, self._fmt_sec)

        self._add_separator()

        # -- Display --
        self._add_section("DISPLAY")
        self._add_combo("color_scheme", "Color Scheme", list(COLOR_SCHEME_CHOICES), "default")
        self._add_combo("button_caption_mode", "Button Labels", ["both", "title", "icon"], "both")
        self._add_combo("bbox_style", "BBox Style", ["square", "diamond", "circle"], "square")
        self._add_combo("panel_background_style", "Panel Background", ["hexagons", "squares"], "hexagons")
        self._add_toggle("labels_only", "Labels Only", False)
        self._add_toggle("edge_background", "Edge Background", True)
        self._add_slider("edge_brightness", "Edge Strength", 15, 5, 40, 1, self._fmt_int)
        self._add_toggle("heat_tags", "Heatmap Tags", False)
        self._add_combo("thermal_mode", "Thermal Mode", ["edge", "clouds", "edge+clouds"], "edge")
        self._add_slider("max_cards", "Max Cards", 4, 3, 6, 1, self._fmt_int)
        self._add_slider("preview_quality", "JPEG Quality", PREVIEW_QUALITY, 30, 100, 1, self._fmt_int)
        self._add_slider("fps", "Target FPS", DEFAULT_FPS, 0, 240, 1, self._fmt_fps)

        self._add_separator()

        # -- Recognition --
        self._add_section("RECOGNITION")
        self._add_slider("recog_threshold", "Match Threshold", RECOGNITION_THRESHOLD, 0.30, 0.99, 100, self._fmt_pct)
        self._add_slider("recog_top_k", "Top-K", RECOGNITION_TOP_K, 1, 20, 1, self._fmt_int)
        self._add_toggle("recog_auto", "Auto-Recognize", RECOGNITION_AUTO)

        self._add_separator()

        # -- Hotkey Peek --
        self._add_section("HOTKEY PEEK")
        self._build_hotkey_peek_section()

        self._add_separator()

        # -- AI Provider --
        self._add_section("AI PROVIDER")
        self._build_ai_controls()

        self._add_separator()

        # -- Privacy --
        self._add_section("PRIVACY")
        self._build_privacy_controls()

        self._add_separator()

        # -- Supervisor --
        self._add_section("SYSTEM HEALTH")
        self._supervisor_summary = QLabel("Supervisor telemetry will appear here.")
        self._supervisor_summary.setWordWrap(True)
        self._supervisor_summary.setStyleSheet("color: rgba(20,8,8,0.78); font-size: 10px; padding: 4px 0;")
        self._layout.addWidget(self._supervisor_summary)
        self._supervisor_log = QTextEdit()
        self._supervisor_log.setReadOnly(True)
        self._supervisor_log.setMinimumHeight(220)
        self._supervisor_log.setStyleSheet(_supervisor_log_style())
        self._layout.addWidget(self._supervisor_log)

        self._layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._save_button = ParallelogramButton("SAVE SETTINGS", variant="primary")
        self._save_button.setFixedHeight(36)
        self._save_button.clicked.connect(self._on_save_clicked)
        self._save_feedback_timer = QTimer(self)
        self._save_feedback_timer.setSingleShot(True)
        self._save_feedback_timer.timeout.connect(self._reset_save_button)
        outer.addWidget(self._save_button)

        self.refresh_privacy_status()
        image_size_slider = self._sliders.get("image_size")
        if image_size_slider is not None:
            image_size_slider[0].setEnabled(False)

    # ---- builders ----

    def _add_section(self, title: str) -> None:
        lbl = QLabel(title)
        lbl.setStyleSheet(_section_style())
        self._section_labels.append(lbl)
        self._layout.addWidget(lbl)

    def _add_separator(self) -> None:
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(_sep_style())
        self._separators.append(sep)
        self._layout.addWidget(sep)

    def _add_slider(
        self,
        key: str,
        label: str,
        default: float | int,
        lo: float | int,
        hi: float | int,
        scale: int,
        fmt: Callable[[int, int], str],
        step: int = 1,
    ) -> None:
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 2, 0, 2)
        rl.setSpacing(6)

        name = QLabel(label)
        name.setStyleSheet(_label_style())
        name.setFixedWidth(100)
        self._row_labels.append(name)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setStyleSheet(_slider_style())
        int_lo = int(lo * scale)
        int_hi = int(hi * scale)
        slider.setRange(int_lo, int_hi)
        slider.setSingleStep(step)
        slider.setValue(int(default * scale))

        val = QLabel(fmt(int(default * scale), scale))
        val.setStyleSheet(_value_style())
        val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        slider.valueChanged.connect(lambda v, s=scale, f=fmt, vl=val, k=key: self._on_slider(k, v, s, f, vl))

        rl.addWidget(name)
        rl.addWidget(slider, stretch=1)
        rl.addWidget(val)

        self._sliders[key] = (slider, val, fmt, scale)
        self._layout.addWidget(row)

    def _add_toggle(self, key: str, label: str, default: bool) -> None:
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 4, 0, 4)
        rl.setSpacing(6)

        name = QLabel(label)
        name.setStyleSheet(_label_style())
        name.setFixedWidth(100)
        self._row_labels.append(name)

        btn = ParallelogramButton("ON" if default else "OFF", variant="ghost")
        btn.setFixedWidth(60)
        btn.setCheckable(True)
        btn.setChecked(default)

        def _toggle(checked: bool, b: ParallelogramButton = btn, k: str = key) -> None:
            b.setText("ON" if checked else "OFF")
            self._emit_toggle(k, checked)

        btn.toggled.connect(_toggle)

        rl.addWidget(name)
        rl.addStretch(1)
        rl.addWidget(btn)
        self._toggles[key] = btn
        self._layout.addWidget(row)

    def _add_combo(self, key: str, label: str, options: list[str], default: str) -> None:
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 4, 0, 4)
        rl.setSpacing(6)

        name = QLabel(label)
        name.setStyleSheet(_label_style())
        name.setFixedWidth(100)
        self._row_labels.append(name)

        combo = QComboBox()
        combo.setStyleSheet(_combo_style())
        combo.addItems(options)
        if default in options:
            combo.setCurrentText(default)
        combo.currentTextChanged.connect(lambda v, k=key: self._emit_combo(k, v))

        rl.addWidget(name)
        rl.addWidget(combo, stretch=1)
        self._combos[key] = combo
        self._layout.addWidget(row)

    def _build_hotkey_peek_section(self) -> None:
        hint = QLabel("Hold Tab to peek selected features (only when both are off)")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme_rgba('accent_dark', 0.55)}; font-size: 9px; padding: 2px 0 6px 0; border: none;")
        self._layout.addWidget(hint)

        _PEEK_OPTIONS = [
            ("roi",     "ROI"),
            ("sidebar", "Sidebar"),
            ("active",  "Active"),
            ("thermal", "Thermal"),
        ]
        row = QWidget()
        row.setStyleSheet("background: transparent;")
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 4)
        rl.setSpacing(4)
        for key, label in _PEEK_OPTIONS:
            btn = ParallelogramButton(label, variant="ghost")
            btn.setCheckable(True)
            btn.setChecked(key in self._hotkey_peek_options)
            btn.clicked.connect(lambda _c, k=key: self._on_hotkey_peek_toggled(k))
            self._hotkey_peek_btns[key] = btn
            rl.addWidget(btn)
        rl.addStretch(1)
        self._layout.addWidget(row)

    def _on_hotkey_peek_toggled(self, key: str) -> None:
        btn = self._hotkey_peek_btns.get(key)
        if btn is None:
            return
        if btn.isChecked():
            self._hotkey_peek_options.add(key)
        else:
            self._hotkey_peek_options.discard(key)
        if isinstance(btn, ParallelogramButton):
            btn.update()
        else:
            btn.setStyleSheet(_toggle_on_style() if btn.isChecked() else _toggle_off_style())
        self.hotkey_peek_changed.emit(set(self._hotkey_peek_options))

    def _build_ai_controls(self) -> None:
        provider_row = QWidget()
        provider_layout = QHBoxLayout(provider_row)
        provider_layout.setContentsMargins(0, 4, 0, 4)
        provider_layout.setSpacing(6)

        provider_label = QLabel("Provider")
        provider_label.setStyleSheet(_label_style())
        self._row_labels.append(provider_label)
        provider_label.setFixedWidth(100)

        self._ai_provider_combo = QComboBox()
        self._ai_provider_combo.setStyleSheet(_combo_style())
        self._ai_provider_combo.addItems(["auto", "ollama", "openai", "anthropic"])
        self._ai_provider_combo.setCurrentText("auto")
        self._ai_provider_combo.currentTextChanged.connect(self._on_ai_provider_changed)

        provider_layout.addWidget(provider_label)
        provider_layout.addWidget(self._ai_provider_combo, stretch=1)
        self._layout.addWidget(provider_row)

        self._ai_model_row = QWidget()
        model_layout = QHBoxLayout(self._ai_model_row)
        model_layout.setContentsMargins(0, 4, 0, 4)
        model_layout.setSpacing(6)

        model_label = QLabel("Model")
        model_label.setStyleSheet(_label_style())
        self._row_labels.append(model_label)
        model_label.setFixedWidth(100)

        self._ai_model_combo = QComboBox()
        self._ai_model_combo.setStyleSheet(_combo_style())
        self._ai_model_combo.setEditable(False)
        self._ai_model_combo.currentTextChanged.connect(self._emit_current_ai_model)

        self._ai_model_edit = QLineEdit()
        self._ai_model_edit.setStyleSheet(_line_edit_style())
        self._ai_model_edit.setPlaceholderText("e.g. llava:latest")
        self._ai_model_edit.textChanged.connect(self._emit_current_ai_model)

        model_layout.addWidget(model_label)
        model_layout.addWidget(self._ai_model_combo, stretch=1)
        model_layout.addWidget(self._ai_model_edit, stretch=1)
        self._layout.addWidget(self._ai_model_row)

        self._ai_models: dict[str, str] = {
            "ollama": INSIGHT_OLLAMA_MODEL,
            "openai": INSIGHT_OPENAI_MODEL,
            "anthropic": INSIGHT_ANTHROPIC_MODEL,
        }
        self._sync_ai_model_controls("auto")

    def _build_privacy_controls(self) -> None:
        self._privacy_summary = QLabel("Checking whether Insight storage sits inside cloud-synced folders.")
        self._privacy_summary.setWordWrap(True)
        self._privacy_summary.setStyleSheet(_privacy_safe_style())
        self._layout.addWidget(self._privacy_summary)

        self._privacy_status = QLabel("")
        self._privacy_status.setWordWrap(True)
        self._privacy_status.setStyleSheet("color: rgba(20,8,8,0.78); font-size: 10px; padding: 2px 0 6px 0;")
        self._layout.addWidget(self._privacy_status)

        self._privacy_guide = QTextEdit()
        self._privacy_guide.setReadOnly(True)
        self._privacy_guide.setMinimumHeight(170)
        self._privacy_guide.setStyleSheet(_privacy_guide_style())
        self._layout.addWidget(self._privacy_guide)

    def _on_ai_provider_changed(self, provider: str) -> None:
        self._sync_ai_model_controls(provider)
        self.ai_provider_changed.emit(provider)
        self._emit_current_ai_model()

    def _sync_ai_model_controls(self, provider: str) -> None:
        provider = provider if provider in {"auto", "ollama", "openai", "anthropic"} else "auto"
        if provider == "openai":
            self._ai_model_combo.blockSignals(True)
            self._ai_model_combo.clear()
            self._ai_model_combo.addItems(_OPENAI_MODELS)
            self._ai_model_combo.setCurrentText(self._ai_models.get("openai", INSIGHT_OPENAI_MODEL))
            self._ai_model_combo.blockSignals(False)
            self._ai_model_combo.setEditable(False)
            self._ai_model_combo.setStyleSheet(_combo_style())
            self._ai_model_combo.show()
            self._ai_model_edit.hide()
            self._ai_model_row.show()
            return
        if provider == "anthropic":
            self._ai_model_combo.blockSignals(True)
            self._ai_model_combo.clear()
            self._ai_model_combo.addItems(_ANTHROPIC_MODELS)
            self._ai_model_combo.setCurrentText(self._ai_models.get("anthropic", INSIGHT_ANTHROPIC_MODEL))
            self._ai_model_combo.blockSignals(False)
            self._ai_model_combo.setEditable(False)
            self._ai_model_combo.setStyleSheet(_combo_style())
            self._ai_model_combo.show()
            self._ai_model_edit.hide()
            self._ai_model_row.show()
            return
        if provider == "ollama":
            current = self._ai_models.get("ollama", INSIGHT_OLLAMA_MODEL)
            self._ai_model_combo.blockSignals(True)
            self._ai_model_combo.setEditable(True)
            self._ai_model_combo.clear()
            self._ai_model_combo.addItems(list(OLLAMA_VISION_MODELS))
            if current and self._ai_model_combo.findText(current) < 0:
                self._ai_model_combo.addItem(current)
            self._ai_model_combo.setCurrentText(current)
            if self._ai_model_combo.lineEdit() is not None:
                self._ai_model_combo.lineEdit().setPlaceholderText("e.g. llava:latest")
                self._ai_model_combo.lineEdit().setStyleSheet(_line_edit_style())
            self._ai_model_combo.blockSignals(False)
            self._ai_model_combo.setStyleSheet(_combo_style())
            self._ai_model_combo.show()
            self._ai_model_edit.hide()
            self._ai_model_row.show()
            return
        self._ai_model_row.hide()

    def _emit_current_ai_model(self, *_args: Any) -> None:
        provider = self._ai_provider_combo.currentText()
        if provider == "openai":
            model = self._ai_model_combo.currentText().strip() or INSIGHT_OPENAI_MODEL
            self._ai_models["openai"] = model
            self.ai_model_changed.emit("openai", model)
        elif provider == "anthropic":
            model = self._ai_model_combo.currentText().strip() or INSIGHT_ANTHROPIC_MODEL
            self._ai_models["anthropic"] = model
            self.ai_model_changed.emit("anthropic", model)
        elif provider == "ollama":
            model = self._ai_model_combo.currentText().strip() or INSIGHT_OLLAMA_MODEL
            self._ai_models["ollama"] = model
            self.ai_model_changed.emit("ollama", model)

    def set_ai_selection(self, provider: str, models: dict[str, str]) -> None:
        for key in ("ollama", "openai", "anthropic"):
            value = str(models.get(key, "") or "").strip()
            if value:
                self._ai_models[key] = value
        provider_value = provider if provider in {"auto", "ollama", "openai", "anthropic"} else "auto"
        self._ai_provider_combo.blockSignals(True)
        self._ai_provider_combo.setCurrentText(provider_value)
        self._ai_provider_combo.blockSignals(False)
        self._sync_ai_model_controls(provider_value)

    # ---- formatters ----

    @staticmethod
    def _fmt_pct(v: int, scale: int) -> str:
        return f"{v / scale * 100:.0f}%"

    @staticmethod
    def _fmt_sec(v: int, scale: int) -> str:
        return f"{v / scale:.1f}s"

    @staticmethod
    def _fmt_sec_int(v: int, _scale: int) -> str:
        return f"{v}s"

    @staticmethod
    def _fmt_int(v: int, _scale: int) -> str:
        return str(v)

    @staticmethod
    def _fmt_fps(v: int, _scale: int) -> str:
        return "Uncapped" if v <= 0 else str(v)

    # ---- emitters ----

    def _on_slider(self, key: str, value: int, scale: int, fmt: Callable, val_label: QLabel) -> None:
        real = value / scale if scale > 1 else value
        if key == "image_size":
            real = normalize_image_size(real)
            slider_data = self._sliders.get(key)
            if slider_data is not None:
                slider = slider_data[0]
                if slider.value() != int(real):
                    slider.blockSignals(True)
                    slider.setValue(int(real))
                    slider.blockSignals(False)
            value = int(real * scale) if scale > 1 else int(real)
        val_label.setText(fmt(value, scale))
        if key == "fps":
            self.fps_changed.emit(int(real))
        elif key == "edge_brightness":
            self.edge_brightness_changed.emit(float(real) / 100.0)
            return  # don't forward to session settings
        self._emit_setting(key, real)

    def _emit_setting(self, key: str, value: Any) -> None:
        self._send({"type": "update_settings", "settings": {key: value}})

    def _on_save_clicked(self) -> None:
        self._send({"type": "save_settings"})
        self._save_button.setText("[SAVED]")
        self._save_feedback_timer.start(1400)

    def _reset_save_button(self) -> None:
        self._save_button.setText("SAVE SETTINGS")

    def set_slider_value(self, key: str, value: float | int) -> None:
        slider_data = self._sliders.get(key)
        if slider_data is None:
            return
        slider, label, fmt, scale = slider_data
        if key == "image_size":
            value = normalize_image_size(value)
        raw_value = int(round(float(value) * scale))
        raw_value = max(slider.minimum(), min(slider.maximum(), raw_value))
        slider.blockSignals(True)
        slider.setValue(raw_value)
        slider.blockSignals(False)
        label.setText(fmt(raw_value, scale))

    def set_combo_value(self, key: str, value: str) -> None:
        combo = self._combos.get(key)
        if combo is None:
            return
        text = str(value or "").strip()
        if text and combo.findText(text) < 0:
            combo.addItem(text)
        combo.blockSignals(True)
        if text:
            combo.setCurrentText(text)
        combo.blockSignals(False)

    def set_hotkey_peek(self, options: set[str]) -> None:
        self._hotkey_peek_options = set(options)
        for key, btn in self._hotkey_peek_btns.items():
            btn.blockSignals(True)
            btn.setChecked(key in self._hotkey_peek_options)
            if isinstance(btn, ParallelogramButton):
                btn.update()
            else:
                btn.setStyleSheet(_toggle_on_style() if btn.isChecked() else _toggle_off_style())
            btn.blockSignals(False)

    def set_toggle_value(self, key: str, value: bool) -> None:
        btn = self._toggles.get(key)
        if btn is None:
            return
        state = bool(value)
        btn.blockSignals(True)
        if isinstance(btn, ParallelogramButton):
            btn.setChecked(state)
            btn.setText("ON" if state else "OFF")
            btn.update()
        else:
            btn._state = state
            btn.setText("ON" if state else "OFF")
            btn.setStyleSheet(_toggle_on_style() if state else _toggle_off_style())
        btn.blockSignals(False)

    def _emit_toggle(self, key: str, state: bool) -> None:
        if key == "recog_auto":
            self._send({"type": "set_recognition_auto", "enabled": state})
        elif key == "edge_background":
            self.edge_background_changed.emit(state)
        elif key == "heat_tags":
            self.heat_tags_changed.emit(state)
        elif key == "labels_only":
            self.labels_only_changed.emit(state)
        else:
            self._emit_setting(key, state)

    def _emit_combo(self, key: str, value: str) -> None:
        if key == "ai_provider":
            self.ai_provider_changed.emit(value)
        elif key == "bbox_style":
            self.bbox_style_changed.emit(value)
        elif key == "panel_background_style":
            self.panel_background_style_changed.emit(value)
        elif key == "thermal_mode":
            self.thermal_mode_changed.emit(value)
        elif key == "button_caption_mode":
            self.button_caption_mode_changed.emit(value)
        elif key == "color_scheme":
            self.color_scheme_changed.emit(value)
        else:
            self._emit_setting(key, value)

    def refresh_theme_styles(self) -> None:
        for lbl in self._section_labels:
            lbl.setStyleSheet(_section_style())
        for lbl in self._row_labels:
            lbl.setStyleSheet(_label_style())
        for sep in self._separators:
            sep.setStyleSheet(_sep_style())
        for _key, (slider, val, _fmt, _scale) in self._sliders.items():
            slider.setStyleSheet(_slider_style())
            val.setStyleSheet(_value_style())
        for _key, btn in self._toggles.items():
            if isinstance(btn, ParallelogramButton):
                btn.update()
            else:
                btn.setStyleSheet(_toggle_on_style() if btn._state else _toggle_off_style())
        for _key, combo in self._combos.items():
            combo.setStyleSheet(_combo_style())
        for key, btn in self._hotkey_peek_btns.items():
            if isinstance(btn, ParallelogramButton):
                btn.update()
            else:
                btn.setStyleSheet(_toggle_on_style() if key in self._hotkey_peek_options else _toggle_off_style())
        if hasattr(self, "_ai_provider_combo"):
            self._ai_provider_combo.setStyleSheet(_combo_style())
        if hasattr(self, "_ai_model_combo"):
            self._ai_model_combo.setStyleSheet(_combo_style())
            if self._ai_model_combo.lineEdit() is not None:
                self._ai_model_combo.lineEdit().setStyleSheet(_line_edit_style())
        if hasattr(self, "_ai_model_edit"):
            self._ai_model_edit.setStyleSheet(_line_edit_style())
        self._supervisor_log.setStyleSheet(_supervisor_log_style())
        self._privacy_guide.setStyleSheet(_privacy_guide_style())
        if hasattr(self, "_save_button") and not isinstance(self._save_button, ParallelogramButton):
            self._save_button.setStyleSheet(_save_button_style())
        if hasattr(self, "_session_metrics"):
            self._session_metrics.setStyleSheet(_session_metrics_style())

    def set_live_session_metrics(self, entries: list[tuple[str, str]]) -> None:
        """Runtime HUD line items (mode, view, model, latency, etc.) for the Live Session block."""
        if not entries:
            self._session_metrics.setText("—")
            return
        self._session_metrics.setText("\n".join(f"{k}  {v}" for k, v in entries))

    def refresh_privacy_status(self, storage_paths: dict[str, Path | str] | None = None) -> None:
        if storage_paths is not None:
            self._storage_paths = {
                label: Path(path).expanduser().resolve()
                for label, path in storage_paths.items()
            }
        status = detect_privacy_status(self._storage_paths)
        self._apply_privacy_status(status)

    def _apply_privacy_status(self, status: PrivacyStatus) -> None:
        if status.protected:
            self._privacy_summary.setText(
                "Current Insight storage does not appear to be inside iCloud Drive or Microsoft OneDrive."
            )
            self._privacy_summary.setStyleSheet(_privacy_safe_style())
        else:
            self._privacy_summary.setText(
                "Warning: some Insight data paths appear to be inside iCloud Drive or Microsoft OneDrive. "
                "Those services may upload copies to their servers."
            )
            self._privacy_summary.setStyleSheet(_privacy_warn_style())

        lines: list[str] = []
        for provider in status.providers:
            state = "ENABLED / DETECTED" if provider.detected else "NOT DETECTED"
            lines.append(f"{provider.name}: {state}")
            if provider.detected and provider.matched_paths:
                lines.append(f"Current Insight paths inside {provider.name}: {', '.join(provider.matched_paths)}")
            elif provider.detected:
                lines.append(f"Current Insight paths inside {provider.name}: no")
            else:
                lines.append(f"Current Insight paths inside {provider.name}: no")
        for label, path in self._storage_paths.items():
            lines.append(f"{label}: {path}")
        self._privacy_status.setText("\n".join(lines))

        guide_lines = [
            "Privacy note:",
            "If iCloud Drive or Microsoft OneDrive sync these folders, local gallery images, history, and state files may be copied to their servers.",
            "",
            "How to reduce that risk:",
            "1. Move the Insight project, gallery, and state folders outside any iCloud Drive or OneDrive location.",
            "2. Restart Insight after moving the folders so the app uses the new local-only paths.",
            "",
            "iCloud Drive:",
            "System Settings -> Apple ID -> iCloud -> iCloud Drive.",
            "Turn off iCloud Drive or move Insight folders out of iCloud Drive locations such as Desktop, Documents, or iCloud Drive itself.",
            "If the folder is already in iCloud Drive, drag it to a normal local projects folder or another local disk.",
            "",
            "Microsoft OneDrive:",
            "OneDrive menu bar icon -> Settings -> Sync and backup.",
            "Turn off folder backup, or open Choose folders / Manage backup and stop syncing the folder that contains Insight data.",
            "If Insight is inside a OneDrive folder, move it to a normal local folder outside OneDrive.",
        ]
        self._privacy_guide.setPlainText("\n".join(guide_lines))

    def apply_supervisor_state(self, summary: str, lines: list[str]) -> None:
        self._supervisor_summary.setText(summary or "Supervisor telemetry unavailable.")
        self._supervisor_log.setPlainText("\n".join(lines))
