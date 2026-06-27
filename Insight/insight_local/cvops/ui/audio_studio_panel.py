"""Audio Studio — dedicated dataset-readiness panel for audio_recognition scenarios.

Replaces the generic DatasetPanel inside the "Dataset Readiness" card whenever the
selected scenario's backbone is `audio_recognition`. The layout is editor-first:
a multi-region waveform sits at the centre, draft regions live in a table directly
beneath it, and the committed-clip ledger from /database/{slug} sits at the bottom.

The panel exposes the same public surface as DatasetPanel so catalog_panel.py can
treat it as a drop-in:
    .set_scenario(scenario, dataset_folder, backbone_type, backbone_config)
    .reload_library_list()
    .reload()
    .refresh_responsive_layout()
    signals: datasetChanged(str), errorRaised(str)
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid
from array import array
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QProcess, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .audio_timeline import AudioTimeline, _fmt_ms
from .cvops_theme import cvops_mapped_qcolor, cvops_qcolor


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm"}
_REGION_PALETTE = (
    QColor(133, 153, 0, 110),   # solarized green
    QColor(38, 139, 210, 110),  # blue
    QColor(181, 137, 0, 110),   # yellow
    QColor(211, 54, 130, 110),  # magenta
    QColor(42, 161, 152, 110),  # cyan
    QColor(108, 113, 196, 110), # violet
    QColor(203, 75, 22, 110),   # orange
)


def _human_size(value: Any) -> str:
    try:
        n = float(value)
    except Exception:
        return ""
    if n <= 0:
        return ""
    units = ("B", "KB", "MB", "GB", "TB")
    idx = 0
    while n >= 1024 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(n)} {units[idx]}"
    return f"{n:.1f} {units[idx]}"


# ----------------------------------------------------------------------------
# Multi-region timeline — extends AudioTimeline with persistent labelled regions.
# ----------------------------------------------------------------------------


class MultiRegionTimeline(AudioTimeline):
    """AudioTimeline with persistent, labelled regions painted as colored bands.

    Drag-selection (Shift+drag) still emits selection_changed. A click that lands
    inside an existing region emits region_clicked(region_id) instead of seeking,
    so the row can be re-loaded into the editor. Otherwise behaviour is inherited.
    """

    region_clicked = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._regions: list[dict[str, Any]] = []  # {id,start_ms,end_ms,label,split,color}
        self._active_region_id: str = ""
        self._committed_regions: list[dict[str, Any]] = []  # read-only ghost bands

    # -- region management --------------------------------------------------

    def set_regions(self, regions: list[dict[str, Any]]) -> None:
        self._regions = [dict(r) for r in regions]
        self.update()

    def set_committed_regions(self, regions: list[dict[str, Any]]) -> None:
        self._committed_regions = [dict(r) for r in regions]
        self.update()

    def set_active_region(self, region_id: str) -> None:
        self._active_region_id = str(region_id or "")
        self.update()

    # -- interaction --------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ):
            ms = self._ms_from_x(event.position().x())
            hit = self._region_at_ms(ms)
            if hit:
                self._active_region_id = str(hit.get("id") or "")
                self.region_clicked.emit(self._active_region_id)
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def _region_at_ms(self, ms: int) -> dict[str, Any]:
        for region in self._regions:
            try:
                s = int(region.get("start_ms") or 0)
                e = int(region.get("end_ms") or 0)
            except Exception:
                continue
            if s <= ms <= e:
                return region
        return {}

    # -- paint --------------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self._duration_ms <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = self.rect()
        bottom_h = self._scaled_px(self._BASE_BOTTOM_H, minimum=12)
        track_top = rect.top() + self._scaled_px(5, minimum=3)
        track_bottom = rect.bottom() - bottom_h
        track_h = max(4, track_bottom - track_top)
        width = max(1, rect.width())

        # Committed (immutable) — thin strip along the top of the track.
        strip_h = max(3, track_h // 8)
        for region in self._committed_regions:
            try:
                s = int(region.get("start_ms") or 0)
                e = int(region.get("end_ms") or 0)
            except Exception:
                continue
            if e <= s:
                continue
            sx = max(0, self._x_from_ms(s))
            ex = min(width, self._x_from_ms(e))
            if ex <= sx:
                continue
            color = cvops_qcolor("accent_select", 80)
            painter.fillRect(sx, track_top, ex - sx, strip_h, color)

        # Drafts — full-height translucent bands with label badges.
        for region in self._regions:
            try:
                s = int(region.get("start_ms") or 0)
                e = int(region.get("end_ms") or 0)
            except Exception:
                continue
            if e <= s:
                continue
            sx = max(0, self._x_from_ms(s))
            ex = min(width, self._x_from_ms(e))
            if ex <= sx:
                continue
            color = region.get("color")
            if not isinstance(color, QColor):
                color = cvops_qcolor("accent_active", 110)
            else:
                color = cvops_mapped_qcolor(color)
            painter.fillRect(sx, track_top, ex - sx, track_h, color)

            is_active = str(region.get("id") or "") == self._active_region_id
            border_color = QColor(color)
            border_color.setAlpha(220 if is_active else 160)
            pen = QPen(border_color, 2 if is_active else 1)
            painter.setPen(pen)
            painter.drawRect(sx, track_top, max(1, ex - sx - 1), track_h - 1)

            label = str(region.get("label") or "").strip() or "(unlabeled)"
            split = str(region.get("split") or "train")
            badge = f"{label} - {split}"
            painter.setPen(QPen(cvops_qcolor("text_bright"), 1))
            painter.drawText(sx + 4, track_top + 13, badge)
        painter.end()


# ----------------------------------------------------------------------------
# Audio Studio panel
# ----------------------------------------------------------------------------


class AudioStudioPanel(QWidget):
    """Audio-only dataset-readiness panel: source bin, multi-region editor, ledger."""

    datasetChanged = pyqtSignal(str)
    errorRaised = pyqtSignal(str)

    def __init__(
        self,
        base_url: str,
        http_get: Callable[[str], dict[str, Any]],
        http_post: Callable[[str, Optional[dict[str, Any]]], dict[str, Any]],
        http_delete: Callable[[str], dict[str, Any]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._http_get = http_get
        self._http_post = http_post
        self._http_delete = http_delete

        self._scenario_name: str = ""
        self._dataset_folder_hint: str = ""
        self._backbone_config: dict[str, Any] = {}

        self._audio_assets: list[dict[str, Any]] = []
        self._library_entries: list[tuple[str, str]] = []
        self._committed_items: list[dict[str, Any]] = []

        # Drafts are keyed by source path so switching assets preserves work.
        self._drafts_by_path: dict[str, list[dict[str, Any]]] = {}
        self._current_source_path: str = ""
        self._active_region_id: str = ""

        # ffmpeg waveform decode
        self._wf_process: Optional[QProcess] = None
        self._wf_buffer = bytearray()
        self._wf_token: int = 0

        # Region-bounded playback: when set, _on_position_changed pauses the
        # main player as soon as the cursor reaches this ms.
        self._region_play_end_ms: Optional[int] = None

        # Denoise preview: cleaned region rendered to a temp wav, played by a
        # separate QMediaPlayer so the main waveform/cursor stays untouched.
        self._denoise_preview_paths: list[str] = []
        self._denoise_active: bool = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        root.addWidget(self._build_header())
        root.addWidget(self._build_main_splitter(), stretch=1)

        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.6);")
        root.addWidget(self._status)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("audioStudioHeader")
        header.setStyleSheet(
            "QFrame#audioStudioHeader { background: rgba(133,153,0,0.06);"
            " border: 1px solid rgba(133,153,0,0.18); border-radius: 6px; }"
        )
        h = QHBoxLayout(header)
        h.setContentsMargins(10, 6, 10, 6)
        h.setSpacing(12)

        self._scenario_label = QLabel("Scenario: -")
        self._scenario_label.setStyleSheet("font-weight: 600;")
        h.addWidget(self._scenario_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: rgba(120,120,120,0.25);")
        h.addWidget(sep)

        self._dataset_combo = QComboBox()
        self._dataset_combo.setMinimumWidth(220)
        self._dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)
        h.addWidget(QLabel("AudioFolder:"))
        h.addWidget(self._dataset_combo)

        self._reload_btn = QPushButton("Reload")
        self._reload_btn.clicked.connect(self._on_reload_clicked)
        h.addWidget(self._reload_btn)

        self._import_btn = QPushButton("Import Folder...")
        self._import_btn.setToolTip(
            "Copy a complete local AudioFolder dataset (split/class/audio) into assets/ml_audio."
        )
        self._import_btn.clicked.connect(self._on_import_folder_clicked)
        h.addWidget(self._import_btn)

        h.addStretch(1)

        self._counts_label = QLabel("0 clips | 0 classes | train 0 | val 0")
        self._counts_label.setStyleSheet("color: rgba(133,153,0,0.7);")
        h.addWidget(self._counts_label)

        return header

    def _build_main_splitter(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(3)

        # ----- Top: source bin + editor canvas, side-by-side ----------------
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setChildrenCollapsible(False)
        top.setHandleWidth(3)
        source_bin = self._build_source_bin()
        source_bin.setMinimumWidth(28)
        source_bin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        editor = self._build_editor()
        editor.setMinimumWidth(28)
        editor.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        top.addWidget(source_bin)
        top.addWidget(editor)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 4)

        splitter.addWidget(top)
        splitter.addWidget(self._build_region_table())
        splitter.addWidget(self._build_ledger())
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)
        return splitter

    def _build_source_bin(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        title = QLabel("Audio Source Assets")
        title.setStyleSheet("font-size: 11px; font-weight: 600;")
        v.addWidget(title)

        self._source_search = QLineEdit()
        self._source_search.setPlaceholderText("Filter source files...")
        self._source_search.setClearButtonEnabled(True)
        self._source_search.textChanged.connect(self._refilter_source_list)
        v.addWidget(self._source_search)

        self._source_list = QListWidget()
        self._source_list.setAlternatingRowColors(True)
        self._source_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._source_list.itemSelectionChanged.connect(self._on_source_selection_changed)
        v.addWidget(self._source_list, stretch=1)

        meta = QLabel("Select an asset to load it into the editor.")
        meta.setWordWrap(True)
        meta.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.6);")
        v.addWidget(meta)
        self._source_meta = meta
        return wrap

    def _build_editor(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        self._editor_title = QLabel("Editor: (no clip loaded)")
        self._editor_title.setStyleSheet("font-weight: 600;")
        title_row.addWidget(self._editor_title, stretch=1)
        v.addLayout(title_row)

        # Transport row
        transport = QHBoxLayout()
        transport.setSpacing(6)
        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(64)
        self._play_btn.clicked.connect(self._toggle_play)
        transport.addWidget(self._play_btn)
        self._time_label = QLabel("--:--")
        self._time_label.setFixedWidth(46)
        transport.addWidget(self._time_label)
        self._dur_label = QLabel("/ --:--")
        transport.addWidget(self._dur_label)
        transport.addStretch(1)
        self._zoom_hint = QLabel("Shift+drag = region | Ctrl+wheel = zoom | click region to select")
        self._zoom_hint.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.55);")
        transport.addWidget(self._zoom_hint)
        v.addLayout(transport)

        # Multi-region timeline (the editor canvas)
        self._timeline = MultiRegionTimeline(self)
        self._timeline.setMinimumHeight(140)
        self._timeline.seek_requested.connect(self._on_seek_requested)
        self._timeline.selection_changed.connect(self._on_selection_changed)
        self._timeline.selection_cleared.connect(self._on_selection_cleared)
        self._timeline.region_clicked.connect(self._on_region_clicked)
        v.addWidget(self._timeline, stretch=1)

        # Region label / split / actions row
        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(QLabel("Class:"))
        self._region_label_edit = QLineEdit()
        self._region_label_edit.setPlaceholderText("e.g. alarm, speech, engine")
        self._region_label_edit.editingFinished.connect(self._on_region_field_edited)
        controls.addWidget(self._region_label_edit, stretch=1)
        controls.addWidget(QLabel("Split:"))
        self._region_split_combo = QComboBox()
        self._region_split_combo.addItem("train", "train")
        self._region_split_combo.addItem("val", "val")
        self._region_split_combo.currentIndexChanged.connect(self._on_region_field_edited)
        controls.addWidget(self._region_split_combo)
        v.addLayout(controls)

        # Region playback row — the active draft can be auditioned raw or as
        # the cached denoised version (auto-rendered on selection).
        playback = QHBoxLayout()
        playback.setSpacing(6)
        self._play_region_btn = QPushButton("Play Region")
        self._play_region_btn.setToolTip("Play the active draft region from start to end on the main player.")
        self._play_region_btn.clicked.connect(self._play_active_region)
        playback.addWidget(self._play_region_btn)
        self._play_denoised_btn = QPushButton("Play Denoised")
        self._play_denoised_btn.setToolTip(
            "Play the cached denoised version of the active region on a side channel."
        )
        self._play_denoised_btn.clicked.connect(self._play_denoised_region)
        playback.addWidget(self._play_denoised_btn)
        self._stop_region_btn = QPushButton("Stop")
        self._stop_region_btn.setToolTip("Stop region or denoised playback.")
        self._stop_region_btn.clicked.connect(self._stop_all_playback)
        playback.addWidget(self._stop_region_btn)
        playback.addStretch(1)
        v.addLayout(playback)

        # Region info data card — shows the active draft's metadata + the
        # most recent /audio/analyze metrics for that region.
        self._region_info_card = self._build_region_info_card()
        v.addWidget(self._region_info_card)

        # Action buttons
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._add_region_btn = QPushButton("Add Region")
        self._add_region_btn.setToolTip("Capture the current Shift+drag selection as a draft region.")
        self._add_region_btn.clicked.connect(self._add_region_from_selection)
        actions.addWidget(self._add_region_btn)
        self._commit_region_btn = QPushButton("Commit to Training")
        self._commit_region_btn.setToolTip("Send the active draft to /audio/collect_clip and add it to the dataset.")
        self._commit_region_btn.clicked.connect(self._commit_active_region)
        actions.addWidget(self._commit_region_btn)
        self._commit_all_btn = QPushButton("Commit All")
        self._commit_all_btn.setToolTip("Commit every draft region for this clip.")
        self._commit_all_btn.clicked.connect(self._commit_all_regions)
        actions.addWidget(self._commit_all_btn)
        self._copy_region_btn = QPushButton("Export Clip...")
        self._copy_region_btn.setToolTip("Save the active region as a standalone WAV file.")
        self._copy_region_btn.clicked.connect(self._export_active_region)
        actions.addWidget(self._copy_region_btn)
        self._analyze_region_btn = QPushButton("Analyze")
        self._analyze_region_btn.clicked.connect(self._analyze_active_region)
        actions.addWidget(self._analyze_region_btn)
        actions.addStretch(1)
        v.addLayout(actions)
        return wrap

    def _build_region_info_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("audioRegionInfoCard")
        card.setStyleSheet(
            "QFrame#audioRegionInfoCard {"
            " background: rgba(38,139,210,0.06);"
            " border: 1px solid rgba(38,139,210,0.22);"
            " border-radius: 6px; }"
            " QLabel[role='caption'] { font-size: 10px; color: rgba(200,200,200,0.55); }"
            " QLabel[role='value'] { font-size: 11px; font-weight: 600; }"
        )
        grid = QGridLayout(card)
        grid.setContentsMargins(10, 6, 10, 6)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)

        self._info_value_labels: dict[str, QLabel] = {}

        def add_field(col: int, row: int, key: str, caption: str) -> None:
            cap = QLabel(caption)
            cap.setProperty("role", "caption")
            val = QLabel("-")
            val.setProperty("role", "value")
            grid.addWidget(cap, row * 2, col)
            grid.addWidget(val, row * 2 + 1, col)
            self._info_value_labels[key] = val

        # Row 1 — region identity
        add_field(0, 0, "class", "Class")
        add_field(1, 0, "split", "Split")
        add_field(2, 0, "start", "Start")
        add_field(3, 0, "end", "End")
        add_field(4, 0, "duration", "Duration")
        add_field(5, 0, "source", "Source")

        # Row 2 — analyze-derived metrics (shown after running Analyze)
        add_field(0, 1, "duration_s", "Sec")
        add_field(1, 1, "sample_rate", "SR")
        add_field(2, 1, "rms", "RMS")
        add_field(3, 1, "peak", "Peak")
        add_field(4, 1, "snr_db", "SNR (dB)")
        add_field(5, 1, "silence", "Silence")

        for col in range(6):
            grid.setColumnStretch(col, 1)
        self._render_region_info_card()
        return card

    def _render_region_info_card(self) -> None:
        if not hasattr(self, "_info_value_labels"):
            return
        region = self._active_draft()
        if not region:
            for key, label in self._info_value_labels.items():
                label.setText("-")
            return
        start = int(region.get("start_ms") or 0)
        end = int(region.get("end_ms") or 0)
        source = Path(self._current_source_path).name if self._current_source_path else "-"
        self._info_value_labels["class"].setText(str(region.get("label") or "(unlabeled)"))
        self._info_value_labels["split"].setText(str(region.get("split") or "train"))
        self._info_value_labels["start"].setText(_fmt_ms(start))
        self._info_value_labels["end"].setText(_fmt_ms(end))
        self._info_value_labels["duration"].setText(_fmt_ms(max(0, end - start)))
        self._info_value_labels["source"].setText(source)

        metrics = region.get("metrics") if isinstance(region.get("metrics"), dict) else {}

        def _fmt_metric(key: str, fmt: str, default: str = "-") -> str:
            try:
                value = float(metrics.get(key)) if metrics.get(key) is not None else None
            except Exception:
                return default
            if value is None:
                return default
            return fmt.format(value)

        self._info_value_labels["duration_s"].setText(_fmt_metric("duration_s", "{:.2f}s"))
        sr_val = metrics.get("sample_rate") if metrics else None
        self._info_value_labels["sample_rate"].setText(f"{int(sr_val)} Hz" if sr_val else "-")
        self._info_value_labels["rms"].setText(_fmt_metric("rms", "{:.4f}"))
        self._info_value_labels["peak"].setText(_fmt_metric("peak", "{:.4f}"))
        self._info_value_labels["snr_db"].setText(_fmt_metric("snr_db", "{:.1f}"))
        self._info_value_labels["silence"].setText(_fmt_metric("silence_ratio", "{:.0%}"))

    def _build_region_table(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        title_row = QHBoxLayout()
        title = QLabel("Draft Regions (this clip)")
        title.setStyleSheet("font-weight: 600;")
        title_row.addWidget(title, stretch=1)
        self._clear_drafts_btn = QPushButton("Clear Drafts")
        self._clear_drafts_btn.clicked.connect(self._clear_drafts)
        title_row.addWidget(self._clear_drafts_btn)
        v.addLayout(title_row)

        self._region_columns = (
            ("class", "Class"),
            ("split", "Split"),
            ("start", "Start"),
            ("end", "End"),
            ("duration", "Duration"),
            ("rms", "RMS"),
            ("snr", "SNR (dB)"),
            ("silence", "Silence"),
            ("denoised", "Denoised"),
        )
        self._region_table = QTableWidget(0, len(self._region_columns))
        self._region_table.setHorizontalHeaderLabels([h for _, h in self._region_columns])
        self._region_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._region_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._region_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._region_table.verticalHeader().setVisible(False)
        self._region_table.itemSelectionChanged.connect(self._on_region_table_selection)
        header = self._region_table.horizontalHeader()
        for col_idx in range(len(self._region_columns)):
            mode = (
                QHeaderView.ResizeMode.Stretch
                if col_idx == 0
                else QHeaderView.ResizeMode.ResizeToContents
            )
            header.setSectionResizeMode(col_idx, mode)
        v.addWidget(self._region_table)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._remove_region_btn = QPushButton("Remove Selected")
        self._remove_region_btn.clicked.connect(self._remove_selected_region)
        row.addWidget(self._remove_region_btn)
        row.addStretch(1)
        v.addLayout(row)
        return wrap

    def _build_ledger(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        title_row = QHBoxLayout()
        title = QLabel("Committed Training Clips")
        title.setStyleSheet("font-weight: 600;")
        title_row.addWidget(title, stretch=1)
        self._ledger_search = QLineEdit()
        self._ledger_search.setPlaceholderText("Filter by class, split, or filename...")
        self._ledger_search.setClearButtonEnabled(True)
        self._ledger_search.textChanged.connect(self._render_ledger)
        title_row.addWidget(self._ledger_search, stretch=1)
        v.addLayout(title_row)

        self._ledger_table = QTableWidget(0, 5)
        self._ledger_table.setHorizontalHeaderLabels(
            ["File", "Split", "Class", "Size", "Relative path"]
        )
        self._ledger_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._ledger_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._ledger_table.verticalHeader().setVisible(False)
        ledger_h = self._ledger_table.horizontalHeader()
        ledger_h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        ledger_h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        ledger_h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        ledger_h.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        ledger_h.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        v.addWidget(self._ledger_table)
        return wrap

    # ------------------------------------------------------------------
    # Public API used by catalog_panel
    # ------------------------------------------------------------------

    def set_scenario(
        self,
        scenario: str,
        dataset_folder: str = "",
        backbone_type: str = "",
        backbone_config: Optional[dict[str, Any]] = None,
    ) -> None:
        # backbone_type is part of the shared signature; this panel is only ever
        # mounted for audio_recognition so we just record what we got.
        _ = backbone_type
        self._scenario_name = str(scenario or "").strip()
        self._dataset_folder_hint = str(dataset_folder or "").strip()
        self._backbone_config = dict(backbone_config or {})
        label = self._scenario_name or "-"
        self._scenario_label.setText(f"Scenario: {label}")
        self._load_audio_assets()
        self.reload_library_list()

    def reload_library_list(self) -> None:
        try:
            payload = self._http_get("/database")
        except Exception as exc:
            self._raise_error(f"Database list failed: {exc}")
            return
        names = list(payload.get("datasets") or []) if isinstance(payload, dict) else []
        categories = payload.get("categories") if isinstance(payload, dict) else {}
        if not isinstance(categories, dict):
            categories = {}
        entries: list[tuple[str, str]] = []
        for n in names:
            value = str(n or "").strip()
            if not value:
                continue
            if str(categories.get(value) or "") != "audio":
                continue
            entries.append((value, value))
        self._library_entries = entries

        preferred = self._dataset_folder_hint or str(self._dataset_combo.currentData() or "")
        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()
        self._dataset_combo.addItem("(select an AudioFolder dataset)", "")
        for label, value in entries:
            self._dataset_combo.addItem(label, value)
        target_idx = self._dataset_combo.findData(preferred) if preferred else -1
        if target_idx >= 0:
            self._dataset_combo.setCurrentIndex(target_idx)
        else:
            self._dataset_combo.setCurrentIndex(0)
        self._dataset_combo.blockSignals(False)
        self.reload()

    def reload(self) -> None:
        slug = str(self._dataset_combo.currentData() or "").strip()
        if not slug:
            self._committed_items = []
            self._render_ledger()
            self._sync_committed_regions()
            self._counts_label.setText("0 clips | 0 classes | train 0 | val 0")
            self._status.setText(
                "No AudioFolder dataset selected. Add or import a dataset under assets/ml_audio."
            )
            self._sync_action_state()
            return
        enc = urllib.parse.quote(slug, safe="")
        try:
            payload = self._http_get(f"/database/{enc}")
        except Exception as exc:
            self._raise_error(f"Dataset list failed for '{slug}': {exc}")
            return
        if not isinstance(payload, dict):
            payload = {}
        items = payload.get("audio_files") or []
        if not isinstance(items, list):
            items = []
        self._committed_items = [dict(x) for x in items if isinstance(x, dict)]
        classes = payload.get("classes") if isinstance(payload, dict) else []
        n_classes = len(classes) if isinstance(classes, list) else 0
        split_counts = payload.get("split_counts") if isinstance(payload, dict) else {}
        if not isinstance(split_counts, dict):
            split_counts = {}
        train_n = int(split_counts.get("train") or 0)
        val_n = int(split_counts.get("val") or 0)
        total = len(self._committed_items)
        self._counts_label.setText(
            f"{total} clips | {n_classes} classes | train {train_n} | val {val_n}"
        )
        shown_classes = ", ".join(str(c) for c in (classes or [])[:6] if str(c)) if isinstance(classes, list) else ""
        suffix = f" Classes: {shown_classes}." if shown_classes else ""
        self._status.setText(f"[audio] {slug} - {total} committed clip(s).{suffix}")
        if shown_classes and not str(self._region_label_edit.text() or "").strip():
            try:
                first = next(iter(c for c in (classes or []) if str(c).strip()), "")
                if first:
                    self._region_label_edit.setText(str(first))
            except Exception:
                pass
        self._render_ledger()
        self._sync_committed_regions()
        self._sync_action_state()

    def refresh_responsive_layout(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Source bin
    # ------------------------------------------------------------------

    def _load_audio_assets(self) -> None:
        try:
            payload = self._http_get("/audio/assets")
        except Exception as exc:
            self._audio_assets = []
            self._source_list.clear()
            err = QListWidgetItem(f"Audio asset list failed: {exc}")
            err.setFlags(err.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._source_list.addItem(err)
            self._sync_action_state()
            return
        items = payload.get("items") if isinstance(payload, dict) else []
        if not isinstance(items, list):
            items = []
        self._audio_assets = [dict(x) for x in items if isinstance(x, dict)]
        self._refilter_source_list()
        if self._source_list.count() and self._source_list.currentRow() < 0:
            for i in range(self._source_list.count()):
                item = self._source_list.item(i)
                data = item.data(Qt.ItemDataRole.UserRole) if item is not None else {}
                if isinstance(data, dict):
                    self._source_list.setCurrentRow(i)
                    break

    def _refilter_source_list(self) -> None:
        previous = ""
        current = self._source_list.currentItem()
        if current is not None:
            data = current.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                previous = str(data.get("path") or "")
        query = str(self._source_search.text() or "").strip().lower()
        self._source_list.blockSignals(True)
        self._source_list.clear()
        if not self._audio_assets:
            placeholder = QListWidgetItem("No audio or video assets found under assets/ml_audio.")
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._source_list.addItem(placeholder)
            self._source_list.blockSignals(False)
            self._sync_action_state()
            return
        for entry in self._audio_assets:
            rel = str(entry.get("relative_path") or entry.get("name") or "").strip()
            if not rel:
                continue
            if query and query not in rel.lower():
                cls = str(entry.get("classification_label") or "").lower()
                if query not in cls:
                    continue
            label = rel
            cls = str(entry.get("classification_label") or "").strip()
            if cls:
                label += f"  [{cls}]"
            size = _human_size(entry.get("size"))
            if size:
                label += f"  {size}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(str(entry.get("path") or rel))
            self._source_list.addItem(item)
        if previous:
            for i in range(self._source_list.count()):
                item = self._source_list.item(i)
                data = item.data(Qt.ItemDataRole.UserRole) if item is not None else {}
                if isinstance(data, dict) and str(data.get("path") or "") == previous:
                    self._source_list.setCurrentRow(i)
                    break
        self._source_list.blockSignals(False)
        self._sync_action_state()

    def _current_source_entry(self) -> dict[str, Any]:
        item = self._source_list.currentItem()
        if item is None:
            return {}
        data = item.data(Qt.ItemDataRole.UserRole)
        return dict(data) if isinstance(data, dict) else {}

    def _on_source_selection_changed(self) -> None:
        entry = self._current_source_entry()
        path = str(entry.get("path") or "").strip()
        if not path:
            self._unload_player()
            self._editor_title.setText("Editor: (no clip loaded)")
            self._source_meta.setText("Select an asset to load it into the editor.")
            self._timeline.set_regions([])
            self._sync_committed_regions()
            self._render_region_info_card()
            self._sync_action_state()
            return
        rel = str(entry.get("relative_path") or entry.get("name") or path)
        self._editor_title.setText(f"Editor: {rel}")
        size = _human_size(entry.get("size"))
        meta_parts = [rel]
        if size:
            meta_parts.append(size)
        cls = str(entry.get("classification_label") or "").strip()
        if cls:
            meta_parts.append(f"label: {cls}")
        self._source_meta.setText("  -  ".join(meta_parts))
        self._current_source_path = path
        self._load_player(path)
        self._timeline.set_regions(self._drafts_by_path.get(path, []))
        self._sync_committed_regions()
        self._render_region_table()
        # Auto-fill the class label from the asset's existing classification when
        # the user hasn't typed one yet.
        if cls and not str(self._region_label_edit.text() or "").strip():
            self._region_label_edit.setText(cls)
        self._render_region_info_card()
        self._sync_action_state()

    # ------------------------------------------------------------------
    # Player + ffmpeg waveform decode
    # ------------------------------------------------------------------

    def _ensure_player(self) -> QMediaPlayer:
        if not hasattr(self, "_player"):
            self._player = QMediaPlayer(self)
            self._audio_out = QAudioOutput(self)
            self._player.setAudioOutput(self._audio_out)
            self._player.positionChanged.connect(self._on_position_changed)
            self._player.durationChanged.connect(self._on_duration_changed)
            self._player.playbackStateChanged.connect(self._on_playback_state_changed)
        return self._player

    def _load_player(self, path: str) -> None:
        player = self._ensure_player()
        self._stop_wf_process()
        player.stop()
        player.setSource(QUrl.fromLocalFile(path))
        self._timeline.reset()
        self._play_btn.setText("Play")
        self._time_label.setText("--:--")
        self._dur_label.setText("/ --:--")
        self._start_wf_decode(Path(path))

    def _unload_player(self) -> None:
        if hasattr(self, "_player"):
            self._stop_wf_process()
            self._player.stop()
            self._player.setSource(QUrl())
        self._timeline.reset()
        self._play_btn.setText("Play")
        self._time_label.setText("--:--")
        self._dur_label.setText("/ --:--")
        self._current_source_path = ""

    def _toggle_play(self) -> None:
        player = self._ensure_player()
        if player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            player.pause()
        else:
            player.play()

    def _on_seek_requested(self, ms: int) -> None:
        player = self._ensure_player()
        player.setPosition(int(ms))
        self._timeline.set_cursor(int(ms))

    def _on_position_changed(self, ms: int) -> None:
        self._timeline.set_cursor(int(ms))
        self._time_label.setText(_fmt_ms(int(ms)))
        if self._region_play_end_ms is not None and ms >= self._region_play_end_ms:
            try:
                self._ensure_player().pause()
            except Exception:
                pass
            self._region_play_end_ms = None

    def _on_duration_changed(self, ms: int) -> None:
        self._timeline.set_duration(int(ms))
        self._dur_label.setText(f"/ {_fmt_ms(int(ms))}")

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("Pause")
        else:
            self._play_btn.setText("Play")

    def _stop_wf_process(self) -> None:
        self._wf_token += 1
        self._wf_buffer.clear()
        proc = self._wf_process
        self._wf_process = None
        if proc is not None:
            if proc.state() != QProcess.ProcessState.NotRunning:
                proc.kill()
            proc.deleteLater()

    def _start_wf_decode(self, path: Path) -> None:
        self._wf_token += 1
        token = self._wf_token
        self._wf_buffer.clear()
        self._timeline.set_analyzing()
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._timeline.set_unavailable("ffmpeg not found - waveform unavailable.")
            return
        proc = QProcess(self)
        self._wf_process = proc
        proc.setProgram(ffmpeg)
        proc.setArguments([
            "-v", "error",
            "-i", str(path),
            "-vn", "-ac", "1", "-ar", "80", "-f", "f32le", "pipe:1",
        ])
        proc.readyReadStandardOutput.connect(lambda _t=token: self._wf_stdout(_t))
        proc.errorOccurred.connect(lambda _e, _t=token: self._wf_error(_t))
        proc.finished.connect(lambda _ec, _es, _t=token: self._wf_finished(_t))
        proc.start()

    def _wf_stdout(self, token: int) -> None:
        if token != self._wf_token or self._wf_process is None:
            return
        self._wf_buffer.extend(bytes(self._wf_process.readAllStandardOutput()))

    def _wf_error(self, token: int) -> None:
        if token != self._wf_token:
            return
        self._timeline.set_unavailable("Waveform decode failed.")

    def _wf_finished(self, token: int) -> None:
        if token != self._wf_token:
            return
        if self._wf_process is not None:
            self._wf_buffer.extend(bytes(self._wf_process.readAllStandardOutput()))
            self._wf_process.deleteLater()
        self._wf_process = None
        raw = bytes(self._wf_buffer)
        self._wf_buffer.clear()
        usable = len(raw) - (len(raw) % 4)
        if usable <= 0:
            self._timeline.set_muted()
            return
        samples: array = array("f")
        try:
            samples.frombytes(raw[:usable])
        except Exception:
            self._timeline.set_unavailable("Waveform decode failed.")
            return
        if not samples:
            self._timeline.set_muted()
            return
        bucket_size = max(1, len(samples) // 2400)
        levels: list[float] = []
        peak = 0.0
        for idx in range(0, len(samples), bucket_size):
            bucket = samples[idx: idx + bucket_size]
            if not bucket:
                continue
            rms = math.sqrt(sum(float(v) * float(v) for v in bucket) / len(bucket))
            levels.append(rms)
            if rms > peak:
                peak = rms
        if peak <= 0.0001:
            self._timeline.set_muted()
            return
        self._timeline.set_levels([min(1.0, v / peak) for v in levels])

    # ------------------------------------------------------------------
    # Region drafts
    # ------------------------------------------------------------------

    def _on_selection_changed(self, _start: int, _end: int) -> None:
        self._sync_action_state()

    def _on_selection_cleared(self) -> None:
        self._sync_action_state()

    def _add_region_from_selection(self) -> None:
        path = self._current_source_path
        if not path:
            self._status.setText("Select an audio asset before adding a region.")
            return
        s = self._timeline._sel_start_ms
        e = self._timeline._sel_end_ms
        if s is None or e is None or e <= s:
            self._status.setText("Shift+drag a selection on the waveform first.")
            return
        drafts = self._drafts_by_path.setdefault(path, [])
        region_id = uuid.uuid4().hex[:8]
        color = _REGION_PALETTE[len(drafts) % len(_REGION_PALETTE)]
        label = str(self._region_label_edit.text() or "").strip()
        split = str(self._region_split_combo.currentData() or "train")
        drafts.append({
            "id": region_id,
            "start_ms": int(s),
            "end_ms": int(e),
            "label": label,
            "split": split,
            "color": color,
        })
        self._active_region_id = region_id
        self._timeline.clear_selection()
        self._timeline.set_regions(drafts)
        self._timeline.set_active_region(region_id)
        self._render_region_table()
        self._render_region_info_card()
        self._sync_action_state()
        self._ensure_region_processed(drafts[-1])

    def _drafts_for_current(self) -> list[dict[str, Any]]:
        return self._drafts_by_path.get(self._current_source_path, [])

    def _active_draft(self) -> dict[str, Any]:
        if not self._active_region_id:
            return {}
        for region in self._drafts_for_current():
            if str(region.get("id") or "") == self._active_region_id:
                return region
        return {}

    def _on_region_clicked(self, region_id: str) -> None:
        self._active_region_id = str(region_id or "")
        region = self._active_draft()
        if region:
            self._region_label_edit.blockSignals(True)
            self._region_label_edit.setText(str(region.get("label") or ""))
            self._region_label_edit.blockSignals(False)
            split = str(region.get("split") or "train")
            idx = self._region_split_combo.findData(split)
            if idx >= 0:
                self._region_split_combo.blockSignals(True)
                self._region_split_combo.setCurrentIndex(idx)
                self._region_split_combo.blockSignals(False)
            self._select_region_table_row(region_id)
        self._timeline.set_active_region(self._active_region_id)
        self._render_region_info_card()
        self._sync_action_state()
        if region:
            self._ensure_region_processed(region)

    def _on_region_field_edited(self) -> None:
        region = self._active_draft()
        if not region:
            return
        region["label"] = str(self._region_label_edit.text() or "").strip()
        region["split"] = str(self._region_split_combo.currentData() or "train")
        self._timeline.set_regions(self._drafts_for_current())
        self._timeline.set_active_region(self._active_region_id)
        self._render_region_table()
        self._render_region_info_card()

    def _render_region_table(self) -> None:
        drafts = self._drafts_for_current()
        self._region_table.blockSignals(True)
        self._region_table.setRowCount(len(drafts))
        for row, region in enumerate(drafts):
            cells = self._region_row_cells(region)
            for col, (key, _) in enumerate(self._region_columns):
                text = cells.get(key, "-")
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, region.get("id"))
                tip = cells.get(f"{key}_tooltip") or ""
                if tip:
                    item.setToolTip(tip)
                self._region_table.setItem(row, col, item)
        self._region_table.blockSignals(False)
        self._select_region_table_row(self._active_region_id)

    def _region_row_cells(self, region: dict[str, Any]) -> dict[str, str]:
        start = int(region.get("start_ms") or 0)
        end = int(region.get("end_ms") or 0)
        duration = max(0, end - start)
        cells: dict[str, str] = {
            "class": str(region.get("label") or "(unlabeled)"),
            "split": str(region.get("split") or "train"),
            "start": _fmt_ms(start),
            "end": _fmt_ms(end),
            "duration": _fmt_ms(duration),
        }

        if region.get("_processing_analyze"):
            cells["rms"] = cells["snr"] = cells["silence"] = "..."
        else:
            metrics = region.get("metrics") if isinstance(region.get("metrics"), dict) else {}

            def _num(key: str) -> Optional[float]:
                try:
                    raw = metrics.get(key) if metrics else None
                except Exception:
                    return None
                if raw is None:
                    return None
                try:
                    return float(raw)
                except Exception:
                    return None

            rms_v = _num("rms")
            snr_v = _num("snr_db")
            sil_v = _num("silence_ratio")
            cells["rms"] = f"{rms_v:.4f}" if rms_v is not None else "-"
            cells["snr"] = f"{snr_v:.1f}" if snr_v is not None else "-"
            cells["silence"] = f"{sil_v:.0%}" if sil_v is not None else "-"

        if region.get("_processing_denoise"):
            cells["denoised"] = "..."
        else:
            cleaned = str(region.get("denoised_path") or "").strip()
            if cleaned:
                cells["denoised"] = Path(cleaned).name
                cells["denoised_tooltip"] = cleaned
            else:
                cells["denoised"] = "-"
        return cells

    def _select_region_table_row(self, region_id: str) -> None:
        if not region_id:
            self._region_table.clearSelection()
            return
        self._region_table.blockSignals(True)
        for row in range(self._region_table.rowCount()):
            item = self._region_table.item(row, 0)
            rid = str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""
            if rid == str(region_id):
                self._region_table.selectRow(row)
                break
        self._region_table.blockSignals(False)

    def _on_region_table_selection(self) -> None:
        rows = self._region_table.selectionModel().selectedRows() if self._region_table.selectionModel() else []
        if not rows:
            return
        item = self._region_table.item(rows[0].row(), 0)
        rid = str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""
        if rid:
            self._on_region_clicked(rid)

    def _remove_selected_region(self) -> None:
        rid = self._active_region_id
        if not rid:
            return
        drafts = self._drafts_for_current()
        self._drafts_by_path[self._current_source_path] = [r for r in drafts if str(r.get("id") or "") != rid]
        self._active_region_id = ""
        self._timeline.set_regions(self._drafts_for_current())
        self._timeline.set_active_region("")
        self._render_region_table()
        self._render_region_info_card()
        self._sync_action_state()

    def _clear_drafts(self) -> None:
        if not self._current_source_path:
            return
        if self._drafts_for_current():
            confirm = QMessageBox.question(
                self,
                "Clear drafts",
                "Remove all draft regions for this clip? Committed clips are not affected.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
        self._drafts_by_path[self._current_source_path] = []
        self._active_region_id = ""
        self._timeline.set_regions([])
        self._timeline.set_active_region("")
        self._render_region_table()
        self._render_region_info_card()
        self._sync_action_state()

    # ------------------------------------------------------------------
    # Commit / Analyze / Export
    # ------------------------------------------------------------------

    def _selected_dataset_slug(self) -> str:
        return str(self._dataset_combo.currentData() or "").strip()

    def _play_active_region(self) -> None:
        region = self._active_draft()
        if not region or not self._current_source_path:
            self._status.setText("Select a draft region first.")
            return
        start = int(region.get("start_ms") or 0)
        end = int(region.get("end_ms") or 0)
        if end <= start:
            self._status.setText("Region has no duration.")
            return
        player = self._ensure_player()
        self._region_play_end_ms = end
        player.setPosition(start)
        player.play()
        self._status.setText(
            f"Playing region [{_fmt_ms(start)}-{_fmt_ms(end)}] from {Path(self._current_source_path).name}..."
        )

    def _stop_region_playback(self) -> None:
        self._region_play_end_ms = None
        if hasattr(self, "_player"):
            try:
                self._player.pause()
            except Exception:
                pass

    def _stop_all_playback(self) -> None:
        self._stop_region_playback()
        self._stop_denoise_preview()
        self._sync_action_state()

    def _ensure_preview_player(self) -> QMediaPlayer:
        if not hasattr(self, "_preview_player"):
            self._preview_player = QMediaPlayer(self)
            self._preview_audio_out = QAudioOutput(self)
            self._preview_player.setAudioOutput(self._preview_audio_out)
            self._preview_player.playbackStateChanged.connect(self._on_preview_state_changed)
        return self._preview_player

    def _on_preview_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.StoppedState and self._denoise_active:
            self._denoise_active = False
            self._status.setText("Denoise preview finished.")
            self._sync_action_state()

    def _ensure_region_processed(self, region: dict[str, Any], *, force: bool = False) -> None:
        """Run /audio/analyze + /audio/copy_clip+/audio/clean for *region*, cache the
        results onto the region dict, and refresh the table cells. Idempotent: skips
        steps whose results are already cached unless force=True.
        """
        if not region or not self._current_source_path:
            return
        start = int(region.get("start_ms") or 0)
        end = int(region.get("end_ms") or 0)
        if end <= start:
            return
        rid = str(region.get("id") or "")
        app = QApplication.instance()

        # ---- Analyze ------------------------------------------------------
        if force or not isinstance(region.get("metrics"), dict) or not region.get("metrics"):
            region["_processing_analyze"] = True
            self._render_region_table()
            if app is not None:
                app.processEvents()
            try:
                payload = self._http_json_direct(
                    "POST",
                    "/audio/analyze",
                    {
                        "path": self._current_source_path,
                        "start_ms": start,
                        "end_ms": end,
                    },
                    timeout=180.0,
                )
                metrics = payload.get("metrics") if isinstance(payload, dict) else {}
                region["metrics"] = dict(metrics) if isinstance(metrics, dict) else {}
            except Exception as exc:
                self._raise_error(f"Analyze failed for region {rid}: {exc}")
                region["metrics"] = {}
            region["_processing_analyze"] = False
            self._render_region_table()
            self._render_region_info_card()

        # ---- Denoise (extract -> /audio/clean) ----------------------------
        cached = str(region.get("denoised_path") or "")
        if cached and not force and Path(cached).is_file():
            return
        region["_processing_denoise"] = True
        self._render_region_table()
        if app is not None:
            app.processEvents()
        try:
            fd_raw, raw_path = tempfile.mkstemp(prefix="cvops_region_", suffix=".wav")
            os.close(fd_raw)
        except Exception as exc:
            region["_processing_denoise"] = False
            self._render_region_table()
            self._raise_error(f"Could not create temp file: {exc}")
            return
        self._denoise_preview_paths.append(raw_path)
        try:
            self._http_json_direct(
                "POST",
                "/audio/copy_clip",
                {
                    "source_path": self._current_source_path,
                    "dest_path": raw_path,
                    "start_ms": start,
                    "end_ms": end,
                },
                timeout=120.0,
            )
        except Exception as exc:
            region["_processing_denoise"] = False
            self._render_region_table()
            self._raise_error(f"Region extract failed for {rid}: {exc}")
            return
        try:
            payload = self._http_json_direct(
                "POST",
                "/audio/clean",
                {
                    "path": raw_path,
                    "noise_reduce": True,
                    "trim_silence": False,
                    "normalize": True,
                    "noise_reduction_strength": 0.7,
                },
                timeout=180.0,
            )
        except Exception as exc:
            region["_processing_denoise"] = False
            self._render_region_table()
            self._raise_error(f"Denoise failed for region {rid}: {exc}")
            return
        cleaned = str(payload.get("cleaned_path") or "") if isinstance(payload, dict) else ""
        if cleaned and Path(cleaned).is_file():
            region["denoised_path"] = cleaned
            self._denoise_preview_paths.append(cleaned)
        else:
            self._raise_error(f"Denoise produced no output file for region {rid}.")
        region["_processing_denoise"] = False
        self._render_region_table()
        self._sync_action_state()

    def _play_denoised_region(self) -> None:
        region = self._active_draft()
        if not region:
            self._status.setText("Select a draft region first.")
            return
        cleaned = str(region.get("denoised_path") or "")
        if not cleaned or not Path(cleaned).is_file():
            # Trigger processing on demand if it hasn't run yet (e.g. user
            # hit Play Denoised before background processing finished).
            self._ensure_region_processed(region)
            cleaned = str(region.get("denoised_path") or "")
        if not cleaned or not Path(cleaned).is_file():
            return
        preview = self._ensure_preview_player()
        preview.stop()
        preview.setSource(QUrl.fromLocalFile(cleaned))
        self._denoise_active = True
        preview.play()
        self._status.setText(f"Playing denoised region: {Path(cleaned).name}")
        self._sync_action_state()

    def _stop_denoise_preview(self) -> None:
        if hasattr(self, "_preview_player"):
            try:
                self._preview_player.stop()
            except Exception:
                pass
        self._denoise_active = False
        self._sync_action_state()

    def _cleanup_denoise_temps(self) -> None:
        for path in list(self._denoise_preview_paths):
            try:
                if Path(path).exists():
                    os.remove(path)
            except Exception:
                pass
        self._denoise_preview_paths.clear()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._cleanup_denoise_temps()
        super().closeEvent(event)

    def _commit_active_region(self) -> None:
        region = self._active_draft()
        if not region:
            self._status.setText("Select a draft region first.")
            return
        self._commit_region(region)

    def _commit_all_regions(self) -> None:
        drafts = list(self._drafts_for_current())
        if not drafts:
            self._status.setText("No draft regions to commit.")
            return
        for region in drafts:
            if not self._commit_region(region, refresh=False):
                break
        self.reload()

    def _commit_region(self, region: dict[str, Any], *, refresh: bool = True) -> bool:
        slug = self._selected_dataset_slug()
        if not slug:
            self._status.setText("Select an AudioFolder dataset before committing.")
            return False
        source = self._current_source_path
        if not source:
            self._status.setText("No source clip loaded.")
            return False
        label = str(region.get("label") or "").strip()
        if not label:
            label, ok = QInputDialog.getText(
                self,
                "Audio Class Label",
                "Class label for this training clip:",
            )
            if not ok:
                return False
            label = str(label or "").strip()
            if label:
                region["label"] = label
                self._region_label_edit.setText(label)
        if not label:
            self._status.setText("Commit cancelled: class label is required.")
            return False
        split = str(region.get("split") or "train")
        start_ms = int(region.get("start_ms") or 0)
        end_ms = int(region.get("end_ms") or 0)
        body: dict[str, Any] = {
            "dataset": slug,
            "source_path": source,
            "label": label,
            "split": split,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "clean": True,
            "noise_reduce": True,
            "trim_silence": False,
            "normalize": True,
        }
        rel = Path(source).name
        self._status.setText(f"Committing '{rel}' [{_fmt_ms(start_ms)}-{_fmt_ms(end_ms)}] to {slug}/{split}/{label}...")
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        try:
            self._http_json_direct("POST", "/audio/collect_clip", body, timeout=240.0)
        except Exception as exc:
            self._raise_error(f"Audio commit failed for '{rel}': {exc}")
            return False
        rid = str(region.get("id") or "")
        self._drafts_by_path[self._current_source_path] = [
            r for r in self._drafts_for_current() if str(r.get("id") or "") != rid
        ]
        if self._active_region_id == rid:
            self._active_region_id = ""
        self._timeline.set_regions(self._drafts_for_current())
        self._timeline.set_active_region(self._active_region_id)
        self._render_region_table()
        if refresh:
            self.reload()
            self.datasetChanged.emit(self._scenario_name)
        return True

    def _analyze_active_region(self) -> None:
        """Manual re-run for the active region. Auto-analyze already happens on
        selection; this button forces a refresh (e.g. after editing start/end).
        """
        region = self._active_draft()
        if not region or not self._current_source_path:
            self._status.setText("Select a draft region first.")
            return
        rel = Path(self._current_source_path).name
        self._status.setText(f"Re-analyzing region of {rel}...")
        self._ensure_region_processed(region, force=True)
        metrics = region.get("metrics") if isinstance(region.get("metrics"), dict) else {}
        if metrics:
            self._status.setText(f"[audio] {rel}: {self._format_audio_metrics(metrics)}")

    def _export_active_region(self) -> None:
        region = self._active_draft()
        source = self._current_source_path
        if not source:
            self._status.setText("Select an audio asset first.")
            return
        if region:
            start_ms = int(region.get("start_ms") or 0)
            end_ms = int(region.get("end_ms") or 0)
        else:
            s = self._timeline._sel_start_ms
            e = self._timeline._sel_end_ms
            if s is None or e is None or e <= s:
                self._status.setText("Pick a region or shift+drag a selection first.")
                return
            start_ms, end_ms = int(s), int(e)
        suggested = str(Path(source).with_suffix(".clip.wav"))
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Audio Clip",
            suggested,
            "WAV files (*.wav);;All files (*)",
        )
        if not dest:
            return
        body: dict[str, Any] = {
            "source_path": source,
            "dest_path": dest,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
        self._status.setText(f"Extracting clip from {Path(source).name}...")
        try:
            payload = self._http_json_direct("POST", "/audio/copy_clip", body, timeout=120.0)
        except Exception as exc:
            self._status.setText(f"Export failed: {exc}")
            return
        out = str(payload.get("clip_path") or dest) if isinstance(payload, dict) else dest
        self._status.setText(f"Clip saved: {Path(out).name}")

    @staticmethod
    def _format_audio_metrics(metrics: dict[str, Any]) -> str:
        def _num(key: str, default: float = 0.0) -> float:
            try:
                return float(metrics.get(key) or default)
            except Exception:
                return default

        duration = _num("duration_s")
        sample_rate = int(_num("sample_rate"))
        rms = _num("rms")
        peak = _num("peak")
        snr = _num("snr_db")
        silence = _num("silence_ratio")
        return (
            f"duration {duration:.2f}s | sample_rate {sample_rate} Hz | "
            f"rms {rms:.4f} | peak {peak:.4f} | snr {snr:.1f} dB | silence {silence:.0%}"
        )

    # ------------------------------------------------------------------
    # Ledger
    # ------------------------------------------------------------------

    def _render_ledger(self) -> None:
        query = str(self._ledger_search.text() or "").strip().lower()
        rows: list[dict[str, Any]] = []
        for entry in self._committed_items:
            rel = str(entry.get("relative_path") or entry.get("name") or "").strip()
            if not rel:
                continue
            split = str(entry.get("split") or "")
            cls = str(entry.get("classification_label") or "")
            if query:
                if (
                    query not in rel.lower()
                    and query not in split.lower()
                    and query not in cls.lower()
                ):
                    continue
            rows.append(entry)
        self._ledger_table.setRowCount(len(rows))
        for r, entry in enumerate(rows):
            rel = str(entry.get("relative_path") or entry.get("name") or "")
            cells = [
                Path(rel).name,
                str(entry.get("split") or ""),
                str(entry.get("classification_label") or ""),
                _human_size(entry.get("size")),
                rel,
            ]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(rel)
                self._ledger_table.setItem(r, c, item)

    def _sync_committed_regions(self) -> None:
        """When the loaded source path matches a committed clip's parent_path,
        paint a ghost band along the top of the timeline. Most committed clips
        come from a different *source*, so this is only useful when the user is
        looking at the same parent file again — but cheap enough to always run.
        """
        if not self._current_source_path:
            self._timeline.set_committed_regions([])
            return
        bands: list[dict[str, Any]] = []
        for entry in self._committed_items:
            parent = str(entry.get("parent_path") or entry.get("source_path") or "")
            if parent != self._current_source_path:
                continue
            try:
                start = int(entry.get("start_ms") or 0)
                end = int(entry.get("end_ms") or 0)
            except Exception:
                continue
            if end <= start:
                continue
            bands.append({"start_ms": start, "end_ms": end})
        self._timeline.set_committed_regions(bands)

    # ------------------------------------------------------------------
    # Header actions
    # ------------------------------------------------------------------

    def _on_dataset_changed(self, _index: int) -> None:
        self.reload()

    def _on_reload_clicked(self) -> None:
        self._load_audio_assets()
        self.reload_library_list()

    def _on_import_folder_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import AudioFolder Dataset")
        if not folder:
            return
        body = {"source_path": folder, "category": "audio"}
        self._status.setText(f"Importing AudioFolder dataset {folder}...")
        try:
            self._http_json_direct("POST", "/database/import_folder", body, timeout=300.0)
        except Exception as exc:
            self._raise_error(f"Folder import failed: {exc}")
            return
        self.reload_library_list()

    # ------------------------------------------------------------------
    # State + helpers
    # ------------------------------------------------------------------

    def _sync_action_state(self) -> None:
        has_source = bool(self._current_source_path)
        slug = self._selected_dataset_slug()
        has_drafts = bool(self._drafts_for_current())
        active = self._active_draft()
        has_active = bool(active)
        sel = self._timeline._sel_start_ms is not None and self._timeline._sel_end_ms is not None
        cleaned_ready = bool(active.get("denoised_path")) if active else False
        self._add_region_btn.setEnabled(has_source and sel)
        self._commit_region_btn.setEnabled(has_active and bool(slug))
        self._commit_all_btn.setEnabled(has_drafts and bool(slug))
        self._copy_region_btn.setEnabled(has_source and (has_active or sel))
        self._analyze_region_btn.setEnabled(has_active)
        self._analyze_region_btn.setToolTip(
            "Re-run /audio/analyze for the active region (results show in the row cells)."
        )
        self._remove_region_btn.setEnabled(has_active)
        self._clear_drafts_btn.setEnabled(has_drafts)
        self._play_region_btn.setEnabled(has_active)
        self._play_denoised_btn.setEnabled(has_active)
        self._play_denoised_btn.setToolTip(
            "Play the cached denoised version of the active region."
            if cleaned_ready
            else "Will run denoise for the active region, then play it."
        )
        self._stop_region_btn.setEnabled(
            self._region_play_end_ms is not None or self._denoise_active
        )

    def _http_json_direct(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        *,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        url = self._base_url + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _raise_error(self, message: str) -> None:
        self._status.setText(message)
        self.errorRaised.emit(message)
