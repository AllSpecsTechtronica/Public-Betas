from __future__ import annotations

import base64
import csv
import math
import json
import mimetypes
import shutil
import urllib.parse
import urllib.error
import urllib.request
import uuid
from array import array
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6 import sip
from PyQt6.QtCore import QEvent, QObject, QProcess, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QFontMetrics,
    QIcon,
    QMouseEvent,
    QPainter,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .audio_timeline import AudioTimeline, _fmt_ms

from .annotation_editor import AnnotationEditorDialog
from .selectable_panel import SelectablePanel
from .algo_catalog import reveal_in_finder
from .csv_table_editor import CsvTableEditorDialog
from .cvops_theme import cvops_color, cvops_qcolor, cvops_rgba
from .dataset_editor import DatasetEditorDialog, FolderInventoryDialog
from .schema_fix_panel import SchemaFixPanel
from ...config import ROOT_DIR
from ...ui.media_utils import pixmap_from_b64_jpeg

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _accent_item_selection_rules(view_selector: str) -> str:
    selected_bg = cvops_rgba("selection_active", 0.88)
    selected_text = cvops_color("selection_text")
    return (
        f"{view_selector}::item:selected {{ background: {selected_bg}; color: {selected_text}; }}"
        f" {view_selector}::item:selected:active {{ background: {selected_bg}; color: {selected_text}; }}"
        f" {view_selector}::item:selected:!active {{ background: {selected_bg}; color: {selected_text}; }}"
    )


class AudioWaveformPlayer(QWidget):
    """Self-contained waveform previewer with play/pause for a single audio file.

    Embeds an AudioTimeline for interactive display and an internal QMediaPlayer
    for real playback and seeking.  The selection_changed / selection_cleared
    signals mirror those of AudioTimeline so callers can react to region picks
    without reaching into the internals.
    """

    selection_changed = pyqtSignal(int, int)  # start_ms, end_ms
    selection_cleared = pyqtSignal()
    playback_state_changed = pyqtSignal(QMediaPlayer.PlaybackState)
    position_changed = pyqtSignal(int)
    duration_changed = pyqtSignal(int)
    waveform_visual_changed = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        show_transport: bool = True,
        timeline_visual_scale: float = 1.0,
        notes_side_rail: bool = False,
    ) -> None:
        super().__init__(parent)
        self._show_transport = bool(show_transport)
        self._notes_side_rail = bool(notes_side_rail)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._play_btn = QPushButton("Play")
        self._play_btn.setFixedWidth(64)
        self._play_btn.clicked.connect(self._toggle_play)
        self._time_label = QLabel("--:--")
        self._time_label.setFixedWidth(44)
        self._dur_label = QLabel("/ --:--")

        # Waveform
        self._timeline = AudioTimeline(self)
        if timeline_visual_scale != 1.0:
            self._timeline.set_visual_scale(float(timeline_visual_scale))
        self._timeline.seek_requested.connect(self._on_seek)
        self._timeline.selection_changed.connect(self.selection_changed)
        self._timeline.selection_cleared.connect(self.selection_cleared)

        if self._notes_side_rail and self._show_transport:
            rail_w = 72
            self._play_btn.setFixedWidth(rail_w)
            self._time_label.setFixedWidth(rail_w)
            self._dur_label.setFixedWidth(rail_w)
            self._time_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._dur_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            root_h = QHBoxLayout(self)
            root_h.setContentsMargins(0, 0, 0, 0)
            root_h.setSpacing(6)
            transport_col = QVBoxLayout()
            transport_col.setSpacing(4)
            transport_col.setContentsMargins(0, 0, 0, 0)
            transport_col.addWidget(self._play_btn, 0, Qt.AlignmentFlag.AlignLeft)
            transport_col.addWidget(self._time_label, 0, Qt.AlignmentFlag.AlignLeft)
            transport_col.addWidget(self._dur_label, 0, Qt.AlignmentFlag.AlignLeft)
            transport_col.addStretch(1)
            root_h.addLayout(transport_col, 0)
            root_h.addWidget(self._timeline, 1)
        else:
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(3)
            if self._show_transport:
                ctrl = QHBoxLayout()
                ctrl.setSpacing(6)
                ctrl.addWidget(self._play_btn)
                ctrl.addWidget(self._time_label)
                ctrl.addWidget(self._dur_label)
                ctrl.addStretch()
                root.addLayout(ctrl)
            root.addWidget(self._timeline)

        # Internal player
        self._player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self._player.setAudioOutput(self._audio_out)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        # ffmpeg waveform decode
        self._wf_process: Optional[QProcess] = None
        self._wf_buffer = bytearray()
        self._wf_token: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str) -> None:
        self._stop_wf_process()
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(path))
        self._timeline.reset()
        self._play_btn.setText("Play")
        self._time_label.setText("--:--")
        self._dur_label.setText("/ --:--")
        self.duration_changed.emit(0)
        self.position_changed.emit(0)
        self.waveform_visual_changed.emit()
        self._start_wf_decode(Path(path))

    def unload(self) -> None:
        self._stop_wf_process()
        self._player.stop()
        self._player.setSource(QUrl())
        self._timeline.reset()
        self._play_btn.setText("Play")
        self._time_label.setText("--:--")
        self._dur_label.setText("/ --:--")
        self.duration_changed.emit(0)
        self.position_changed.emit(0)
        self.waveform_visual_changed.emit()

    def set_selection(self, start_ms: int, end_ms: int) -> None:
        self._timeline.set_selection(start_ms, end_ms)

    def clear_selection(self) -> None:
        self._timeline.clear_selection()

    @property
    def sel_start_ms(self) -> Optional[int]:
        return self._timeline._sel_start_ms

    @property
    def sel_end_ms(self) -> Optional[int]:
        return self._timeline._sel_end_ms

    # ------------------------------------------------------------------
    # Player slots
    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_seek(self, ms: int) -> None:
        self._player.setPosition(ms)
        self._timeline.set_cursor(ms)

    def _on_position_changed(self, ms: int) -> None:
        self._timeline.set_cursor(ms)
        self._time_label.setText(_fmt_ms(ms))
        self.position_changed.emit(int(ms))

    def _on_duration_changed(self, ms: int) -> None:
        self._timeline.set_duration(ms)
        self._dur_label.setText(f"/ {_fmt_ms(ms)}")
        self.duration_changed.emit(int(ms))
        self.waveform_visual_changed.emit()

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_btn.setText("Pause")
        else:
            self._play_btn.setText("Play")
        self.playback_state_changed.emit(state)

    # ------------------------------------------------------------------
    # Waveform decode (ffmpeg → RMS buckets, same pipeline as VideoTestPanel)
    # ------------------------------------------------------------------

    def _stop_wf_process(self) -> None:
        self._wf_token += 1
        self._wf_buffer.clear()
        proc = self._wf_process
        self._wf_process = None
        if proc is None:
            return
        try:
            proc.disconnect()
        except TypeError:
            pass
        if proc.state() != QProcess.ProcessState.NotRunning:
            proc.kill()
            proc.waitForFinished(5000)
        proc.deleteLater()

    def _wf_alive(self) -> bool:
        if sip.isdeleted(self):
            return False
        return not sip.isdeleted(self._timeline)

    def _start_wf_decode(self, path: Path) -> None:
        self._wf_token += 1
        token = self._wf_token
        self._wf_buffer.clear()
        self._timeline.set_analyzing()
        self.waveform_visual_changed.emit()

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._timeline.set_unavailable("ffmpeg not found — waveform unavailable.")
            self.waveform_visual_changed.emit()
            return

        proc = QProcess(self)
        self._wf_process = proc
        proc.setProgram(ffmpeg)
        proc.setArguments([
            "-v", "error",
            "-i", str(path),
            "-vn", "-ac", "1", "-ar", "80", "-f", "f32le", "pipe:1",
        ])
        proc.readyReadStandardOutput.connect(
            lambda _token=token: self._wf_stdout(_token)
        )
        proc.errorOccurred.connect(
            lambda _err, _token=token: self._wf_error(_token)
        )
        proc.finished.connect(
            lambda _ec, _es, _token=token: self._wf_finished(_token)
        )
        proc.start()

    def _wf_stdout(self, token: int) -> None:
        if token != self._wf_token or self._wf_process is None:
            return
        if sip.isdeleted(self):
            return
        self._wf_buffer.extend(bytes(self._wf_process.readAllStandardOutput()))

    def _wf_error(self, token: int) -> None:
        if token != self._wf_token:
            return
        if not self._wf_alive():
            return
        self._timeline.set_unavailable("Waveform decode failed.")
        self.waveform_visual_changed.emit()

    def _wf_finished(self, token: int) -> None:
        if token != self._wf_token:
            return
        if self._wf_process is not None:
            self._wf_buffer.extend(bytes(self._wf_process.readAllStandardOutput()))
            try:
                self._wf_process.disconnect()
            except TypeError:
                pass
            self._wf_process.deleteLater()
        self._wf_process = None

        if sip.isdeleted(self) or sip.isdeleted(self._timeline):
            self._wf_buffer.clear()
            return

        raw = bytes(self._wf_buffer)
        self._wf_buffer.clear()
        usable = len(raw) - (len(raw) % 4)
        if usable <= 0:
            self._timeline.set_muted()
            self.waveform_visual_changed.emit()
            return

        samples: array = array("f")
        try:
            samples.frombytes(raw[:usable])
        except Exception:
            self._timeline.set_unavailable("Waveform decode failed.")
            self.waveform_visual_changed.emit()
            return
        if not samples:
            self._timeline.set_muted()
            self.waveform_visual_changed.emit()
            return

        bucket_size = max(1, len(samples) // 2400)
        levels: list[float] = []
        peak = 0.0
        for idx in range(0, len(samples), bucket_size):
            bucket = samples[idx : idx + bucket_size]
            if not bucket:
                continue
            rms = math.sqrt(sum(float(v) * float(v) for v in bucket) / len(bucket))
            levels.append(rms)
            if rms > peak:
                peak = rms
        if peak <= 0.0001:
            self._timeline.set_muted()
            self.waveform_visual_changed.emit()
            return
        self._timeline.set_levels([min(1.0, v / peak) for v in levels])
        self.waveform_visual_changed.emit()

    def toggle_playback(self) -> None:
        self._toggle_play()

    def pause_playback(self) -> None:
        self._player.pause()

    @property
    def timeline_widget(self) -> AudioTimeline:
        return self._timeline

    @property
    def audio_output(self) -> QAudioOutput:
        return self._audio_out

    def waveform_levels(self) -> list[float]:
        return list(getattr(self._timeline, "_levels", []) or [])

    def waveform_state(self) -> str:
        return str(getattr(self._timeline, "_state", "empty") or "empty")

    def duration_ms(self) -> int:
        try:
            return max(0, int(self._player.duration()))
        except Exception:
            return 0

    def position_ms(self) -> int:
        try:
            return max(0, int(self._player.position()))
        except Exception:
            return 0

    def is_playing(self) -> bool:
        return self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState


# Minimum visible rows in the dataset thumbnail grid (icon mode).
_DATASET_PREVIEW_ROWS_MIN = 3


def _highlight_html(text: str, term: str) -> str:
    """Return HTML where every case-insensitive occurrence of *term* in *text*
    is wrapped in a highlight span.  Returns plain-escaped text when term is empty."""
    import html as _html_mod
    escaped_text = _html_mod.escape(text)
    if not term:
        return escaped_text
    lower_text = text.lower()
    lower_term = term.lower()
    parts: list[str] = []
    i = 0
    while i < len(text):
        idx = lower_text.find(lower_term, i)
        if idx == -1:
            parts.append(_html_mod.escape(text[i:]))
            break
        parts.append(_html_mod.escape(text[i:idx]))
        matched = text[idx: idx + len(term)]
        parts.append(
            f'<span style="background-color: {cvops_rgba("accent_select", 0.38)};'
            f' color: {cvops_color("text_bright")}; font-weight: bold;">'
            f"{_html_mod.escape(matched)}</span>"
        )
        i = idx + len(term)
    return "".join(parts)


def _make_dot_icon(color: QColor) -> QIcon:
    pix = QPixmap(10, 10)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawEllipse(0, 0, 9, 9)
    painter.end()
    icon = QIcon()
    # Keep the dot color visible even when the button is disabled.
    icon.addPixmap(pix, QIcon.Mode.Normal)
    icon.addPixmap(pix, QIcon.Mode.Disabled)
    return icon


def _label_status_icon(has_label: bool) -> QIcon:
    if has_label:
        return _make_dot_icon(cvops_qcolor("accent_select"))
    return _make_dot_icon(cvops_qcolor("accent_alert"))


def _multipart_upload(
    url: str,
    *,
    fields: Optional[dict[str, str]] = None,
    files: Optional[dict[str, Path]] = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    boundary = f"----cvops-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in (fields or {}).items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    for field_name, file_path in (files or {}).items():
        ctype, _ = mimetypes.guess_type(str(file_path))
        if not ctype:
            ctype = "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'.encode()
        )
        body.extend(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def _expand_upload_paths(paths: list[str]) -> list[str]:
    """Normalize dropped/selected paths into a de-duplicated flat image list."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(str(raw or "")).expanduser()
        candidates: list[Path] = []
        if path.is_dir():
            try:
                candidates = [p for p in sorted(path.rglob("*")) if p.is_file()]
            except Exception:
                candidates = []
        elif path.is_file():
            candidates = [path]
        for candidate in candidates:
            if candidate.suffix.lower() not in IMAGE_EXTS:
                continue
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _matching_upload_label_path(image_path: Path) -> Optional[Path]:
    """Find the label that belongs to an uploaded image.

    Supports both simple sidecar labels (`foo.jpg` + `foo.txt`) and standard
    YOLO trees (`images/train/foo.jpg` + `labels/train/foo.txt`).
    """
    image = Path(image_path).expanduser()
    sidecar = image.with_suffix(".txt")
    if sidecar.is_file():
        return sidecar

    parts = image.parts
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() != "images":
            continue
        candidate = Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
        if candidate.is_file():
            return candidate
    return None


def _infer_upload_split(image_path: Path, default_split: str) -> str:
    """Infer train/val from folder names, falling back to the selected UI split."""
    default = str(default_split or "train").strip().lower()
    if default not in {"train", "val"}:
        default = "train"
    aliases = {
        "train": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "test": "val",
    }
    parts = [str(part).lower() for part in Path(image_path).parts]
    for idx, part in enumerate(parts[:-1]):
        if part == "images" and idx + 1 < len(parts):
            mapped = aliases.get(parts[idx + 1])
            if mapped:
                return mapped
    for part in reversed(parts[:-1]):
        mapped = aliases.get(part)
        if mapped:
            return mapped
    return default


def _human_size(size: Any) -> str:
    try:
        n = float(size or 0)
    except Exception:
        n = 0.0
    if n <= 0:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return ""


class _ImageDropList(QListWidget):
    filesDropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(True)
        self.setIconSize(QSize(124, 124))
        self.setGridSize(QSize(156, 232))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSpacing(8)
        self.setUniformItemSizes(False)
        self.setWordWrap(True)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.setStyleSheet(
            "QListWidget::item { padding: 4px; }"
            f" {_accent_item_selection_rules('QListWidget')}"
            " QListWidget::item:selected { border-radius: 0px; }"
        )
        self.setMinimumHeight(120)
        self._apply_preview_min_height()

    def _apply_preview_min_height(self) -> None:
        rows = 2
        cell_h = self.gridSize().height()
        sp = self.spacing()
        # Extra for item padding (stylesheet), list frame, and a horizontal scrollbar if shown.
        extra = 24
        target = rows * cell_h + max(0, rows - 1) * sp + extra
        self.setMinimumHeight(min(220, max(96, target)))

    def _sync_tile_geometry(self) -> None:
        # Keep tiles compact on small windows while allowing richer previews on larger ones.
        viewport_w = max(220, self.viewport().width())
        cols = max(1, viewport_w // 150)
        tile_w = int(max(132, min(176, (viewport_w - ((cols - 1) * self.spacing())) / cols)))
        icon_w = int(max(96, min(144, tile_w - 26)))
        tile_h = int(max(192, min(272, icon_w + 98)))
        self.setIconSize(QSize(icon_w, icon_w))
        self.setGridSize(QSize(tile_w, tile_h))
        self._apply_preview_min_height()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_tile_geometry()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is not None and md.hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData() is not None and event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is None or not md.hasUrls():
            return
        paths: list[str] = []
        for url in md.urls():
            if not url.isLocalFile():
                continue
            paths.append(url.toLocalFile())
        images = _expand_upload_paths(paths)
        if images:
            self.filesDropped.emit(images)
            event.acceptProposedAction()
            return
        event.ignore()


class _ImageTableWidget(QTableWidget):
    """Spreadsheet-style dataset browser with thumbnail and metadata columns."""

    filesDropped = pyqtSignal(list)

    COL_PREVIEW = 0
    COL_FILE = 1
    COL_SPLIT = 2
    COL_LABEL = 3
    COL_CLASS = 4
    COL_SIZE = 5
    COL_PATH = 6
    SEARCH_HIT_ROLE = int(Qt.ItemDataRole.UserRole.value) + 10

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setColumnCount(7)
        self.setHorizontalHeaderLabels(
            ["Preview", "File", "Split", "Label", "Class / Identity", "Size", "Relative path"]
        )
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setShowGrid(True)
        self.setIconSize(QSize(64, 64))
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(72)
        self.setMinimumHeight(220)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        header = self.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(self.COL_PREVIEW, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_SPLIT, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_LABEL, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_CLASS, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_SIZE, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_PATH, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(self.COL_PREVIEW, 84)
        self.setStyleSheet(
            "QTableWidget::item { padding: 4px 8px; }"
            f" {_accent_item_selection_rules('QTableWidget')}"
            " QHeaderView::section { padding: 5px 8px; }"
        )
        self._audio_mode = False

    def set_audio_mode(self, audio_mode: bool) -> None:
        self._audio_mode = bool(audio_mode)
        if self._audio_mode:
            self.setHorizontalHeaderLabels(
                ["Audio", "File", "Split", "Label", "Class", "Size", "Relative path"]
            )
        else:
            self.setHorizontalHeaderLabels(
                ["Preview", "File", "Split", "Label", "Class / Identity", "Size", "Relative path"]
            )

    def _sync_table_geometry(self) -> None:
        self.setColumnWidth(self.COL_PREVIEW, 84)
        for row in range(self.rowCount()):
            self.setRowHeight(row, 72)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is not None and md.hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData() is not None and event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is None or not md.hasUrls():
            event.ignore()
            return
        paths = [url.toLocalFile() for url in md.urls() if url.isLocalFile()]
        images = _expand_upload_paths(paths)
        if images:
            self.filesDropped.emit(images)
            event.acceptProposedAction()
            return
        event.ignore()

    def add_image_row(
        self,
        *,
        rel_path: str,
        display_name: str,
        split: str,
        has_label: bool,
        classification_label: str,
        size: Any,
        active_search: str = "",
    ) -> int:
        row = self.rowCount()
        self.insertRow(row)
        self.setRowHeight(row, 72)

        preview = QTableWidgetItem()
        preview.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setData(Qt.ItemDataRole.UserRole, rel_path)
        preview.setToolTip(rel_path)
        self.setItem(row, self.COL_PREVIEW, preview)

        file_item = self._item(display_name or Path(rel_path).name, rel_path)
        file_item.setToolTip(rel_path)
        self.setItem(row, self.COL_FILE, file_item)

        self.setItem(row, self.COL_SPLIT, self._item(split or "root", rel_path))

        label_item = self._item("labeled" if has_label else "missing", rel_path)
        label_item.setIcon(_label_status_icon(has_label))
        label_item.setToolTip(
            "Open annotation editor (edit boxes / YOLO labels)."
            if has_label
            else "No label file yet — open annotation editor to add labels."
        )
        label_item.setForeground(cvops_qcolor("text_iron" if has_label else "accent_alert"))
        self.setItem(row, self.COL_LABEL, label_item)

        class_item = self._item(classification_label, rel_path)
        class_item.setToolTip(classification_label or "")
        self.setItem(row, self.COL_CLASS, class_item)

        self.setItem(row, self.COL_SIZE, self._item(_human_size(size), rel_path))
        path_item = self._item(rel_path, rel_path)
        path_item.setToolTip(rel_path)
        self.setItem(row, self.COL_PATH, path_item)

        if active_search:
            haystack = " ".join([display_name, rel_path, classification_label, split]).lower()
            if str(active_search or "").strip().lower() in haystack:
                for col in range(self.columnCount()):
                    cell = self.item(row, col)
                    if cell is not None:
                        cell.setData(self.SEARCH_HIT_ROLE, True)
                        cell.setBackground(cvops_qcolor("accent_select", 38))
        return row

    def refresh_theme_styles(self) -> None:
        for row in range(self.rowCount()):
            label_item = self.item(row, self.COL_LABEL)
            has_label = label_item is not None and label_item.text().strip().lower() == "labeled"
            if label_item is not None:
                label_item.setIcon(_label_status_icon(has_label))
                label_item.setForeground(cvops_qcolor("text_iron" if has_label else "accent_alert"))
            for col in range(self.columnCount()):
                cell = self.item(row, col)
                if cell is not None and bool(cell.data(self.SEARCH_HIT_ROLE)):
                    cell.setBackground(cvops_qcolor("accent_select", 38))

    def add_audio_row(
        self,
        *,
        rel_path: str,
        display_name: str,
        split: str,
        classification_label: str,
        size: Any,
        active_search: str = "",
    ) -> int:
        row = self.add_image_row(
            rel_path=rel_path,
            display_name=display_name,
            split=split,
            has_label=bool(classification_label),
            classification_label=classification_label,
            size=size,
            active_search=active_search,
        )
        preview = self.item(row, self.COL_PREVIEW)
        if preview is not None:
            preview.setText("WAV")
            preview.setToolTip("Audio clip")
        label_item = self.item(row, self.COL_LABEL)
        if label_item is not None:
            label_item.setText("class" if classification_label else "unlabeled")
            label_item.setToolTip(classification_label or "No class label folder.")
        return row

    def set_thumbnail_pixmap(self, row: int, pix: QPixmap) -> None:
        item = self.item(row, self.COL_PREVIEW)
        if item is None or pix.isNull():
            return
        item.setIcon(QIcon(pix.scaled(
            self.iconSize(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )))

    def selected_relative_paths(self) -> list[str]:
        rows = sorted({index.row() for index in self.selectionModel().selectedRows()})
        out: list[str] = []
        for row in rows:
            item = self.item(row, self.COL_PREVIEW)
            rel = str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""
            if rel:
                out.append(rel)
        return out

    def current_relative_path(self) -> str:
        row = self.currentRow()
        if row < 0:
            return ""
        item = self.item(row, self.COL_PREVIEW)
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""

    @staticmethod
    def _item(text: str, rel_path: str) -> QTableWidgetItem:
        item = QTableWidgetItem(str(text or ""))
        item.setData(Qt.ItemDataRole.UserRole, rel_path)
        return item


class _UploadDropZone(QFrame):
    filesDropped = pyqtSignal(list)
    openFilesRequested = pyqtSignal()
    openFoldersRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("datasetUploadDropZone")
        self.setMinimumHeight(220)
        self.setStyleSheet(
            "QFrame#datasetUploadDropZone {"
            " border: 1px dashed rgba(133,153,0,0.45);"
            " border-radius: 0px;"
            " background: rgba(133,153,0,0.06);"
            "}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(10)

        title = QLabel("Drag images or folders here")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; font-weight: 700;")
        layout.addWidget(title)

        copy = QLabel(
            "Drop image files or full dataset folders here.\n"
            "Folders are scanned recursively, including YOLO images/ + labels/ trees."
        )
        copy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        copy.setWordWrap(True)
        copy.setStyleSheet("font-size: 11px; color: rgba(133,153,0,0.72);")
        layout.addWidget(copy)

        picker_row = QHBoxLayout()
        picker_row.setSpacing(8)
        picker_row.addStretch(1)
        self._files_btn = QPushButton("Add Files")
        self._files_btn.clicked.connect(self.openFilesRequested.emit)
        picker_row.addWidget(self._files_btn)
        self._folders_btn = QPushButton("Add Folder")
        self._folders_btn.clicked.connect(self.openFoldersRequested.emit)
        picker_row.addWidget(self._folders_btn)
        picker_row.addStretch(1)
        layout.addLayout(picker_row)
        layout.addStretch(1)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is not None and md.hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is not None and md.hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is None or not md.hasUrls():
            event.ignore()
            return
        paths = [url.toLocalFile() for url in md.urls() if url.isLocalFile()]
        if _expand_upload_paths(paths):
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        event.ignore()


class _DatasetThumbCell(QWidget):
    """One dataset tile: thumbnail, filename, optional identity/class subtitle, and [Labels] button."""

    def __init__(
        self,
        panel: "DatasetPanel",
        list_widget: QListWidget,
        item: QListWidgetItem,
        display_name: str,
        has_label: bool,
        classification_label: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._panel = panel
        self._list = list_widget
        self._item = item
        self._display_name = str(display_name or "")
        self._classification_label = str(classification_label or "")
        self._has_label = bool(has_label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        self._thumb = QLabel()
        self._thumb.setObjectName("datasetThumb")
        icon_size = self._list.iconSize()
        thumb_side = max(96, min(144, icon_size.width()))
        self._thumb.setFixedSize(thumb_side, thumb_side)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setProperty("hasLabel", bool(has_label))
        self._thumb.setText("…")
        layout.addWidget(self._thumb, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._title = QLabel(display_name if has_label else f"{display_name}\n[label missing]")
        self._title.setObjectName("datasetTitle")
        self._title.setProperty("hasLabel", bool(has_label))
        self._title.setWordWrap(True)
        self._title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._title.setMaximumWidth(max(120, self._list.gridSize().width() - 8))
        layout.addWidget(self._title)

        # Subtitle: shows identity / class label when present.
        self._subtitle = QLabel(classification_label)
        self._subtitle.setObjectName("datasetSubtitle")
        self._subtitle.setStyleSheet("font-size: 9px; color: rgba(133,153,0,0.7);")
        self._subtitle.setWordWrap(True)
        self._subtitle.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self._subtitle.setMaximumWidth(max(120, self._list.gridSize().width() - 8))
        self._subtitle.setVisible(bool(classification_label))
        layout.addWidget(self._subtitle)

        self._labels_btn = QPushButton("Edit labels")
        self._labels_btn.setIcon(_label_status_icon(has_label))
        self._labels_btn.setIconSize(QSize(10, 10))
        self._labels_btn.setToolTip(
            "Open annotation editor for this image."
            if has_label
            else "No label file yet — open annotation editor to add labels."
        )
        self._labels_btn.clicked.connect(self._on_labels_clicked)
        layout.addWidget(self._labels_btn)

        self._thumb.installEventFilter(self)
        self._title.installEventFilter(self)

    def apply_search_highlight(self, term: str) -> None:
        """Wrap every occurrence of *term* in both title and subtitle with a highlight span."""
        max_w = max(120, self._list.gridSize().width() - 8)
        if term:
            title_plain = self._display_name if self._has_label else f"{self._display_name}\n[label missing]"
            self._title.setText(_highlight_html(title_plain, term))
            self._title.setTextFormat(Qt.TextFormat.RichText)
            if self._classification_label:
                self._subtitle.setText(_highlight_html(self._classification_label, term))
                self._subtitle.setTextFormat(Qt.TextFormat.RichText)
                self._subtitle.setVisible(True)
        else:
            # Restore plain text when search is cleared.
            self._title.setTextFormat(Qt.TextFormat.AutoText)
            self._title.setText(
                self._display_name if self._has_label else f"{self._display_name}\n[label missing]"
            )
            self._subtitle.setTextFormat(Qt.TextFormat.AutoText)
            self._subtitle.setText(self._classification_label)
            self._subtitle.setVisible(bool(self._classification_label))
        self._title.setMaximumWidth(max_w)
        self._subtitle.setMaximumWidth(max_w)

    def set_thumbnail_pixmap(self, pix: QPixmap) -> None:
        if pix.isNull():
            return
        scaled = pix.scaled(
            self._thumb.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._thumb.setPixmap(scaled)
        self._thumb.setText("")

    def sync_geometry(self) -> None:
        icon_size = self._list.iconSize()
        thumb_side = max(96, min(144, icon_size.width()))
        self._thumb.setFixedSize(thumb_side, thumb_side)
        max_w = max(120, self._list.gridSize().width() - 8)
        self._title.setMaximumWidth(max_w)
        self._subtitle.setMaximumWidth(max_w)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
            if obj in (self._thumb, self._title) and event.button() == Qt.MouseButton.LeftButton:
                self._list.setCurrentItem(self._item)
                self._item.setSelected(True)
        return super().eventFilter(obj, event)

    def _on_labels_clicked(self) -> None:
        rel_path = str(self._item.data(Qt.ItemDataRole.UserRole) or "")
        if not rel_path:
            return
        self._panel._open_annotation_editor(start_relative_path=rel_path)


class _LabelTextDialog(QDialog):
    def __init__(
        self,
        *,
        image_path: str,
        text: str,
        line_count: int,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Labels — {image_path}")
        self.resize(520, 360)
        outer = QVBoxLayout(self)
        meta = QLabel(f"{line_count} non-empty line(s)")
        meta.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        outer.addWidget(meta)
        view = QTextEdit()
        view.setReadOnly(True)
        font = QFont("JetBrains Mono", 11)
        if not font.exactMatch():
            font = QFont("IBM Plex Mono", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        view.setFont(font)
        view.setPlainText(text)
        outer.addWidget(view)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)


class _UploadDialog(QDialog):
    def __init__(self, *, split: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._paths: list[str] = []
        self.setWindowTitle("Upload Dataset Images")
        self.resize(720, 560)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        intro = QLabel(
            f"Queue images for the `{split}` split. "
            "Use matching `.txt` files for labeled uploads, or enable empty-label mode below."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self._drop_zone = _UploadDropZone(self)
        self._drop_zone.filesDropped.connect(self._add_paths)
        self._drop_zone.openFilesRequested.connect(self._open_files_from_finder)
        self._drop_zone.openFoldersRequested.connect(self._open_folders_from_finder)
        outer.addWidget(self._drop_zone)

        self._empty_labels = QCheckBox("Create empty labels for uploaded images")
        self._empty_labels.setChecked(False)
        outer.addWidget(self._empty_labels)

        self._summary = QLabel("No images selected yet.")
        self._summary.setStyleSheet("font-size: 11px; color: rgba(133,153,0,0.72);")
        outer.addWidget(self._summary)

        self._list = QListWidget()
        self._list.setMinimumHeight(150)
        self._list.setStyleSheet(_accent_item_selection_rules("QListWidget"))
        outer.addWidget(self._list, stretch=1)

        actions = QHBoxLayout()
        add_files_btn = QPushButton("Add Files")
        add_files_btn.clicked.connect(self._open_files_from_finder)
        actions.addWidget(add_files_btn)
        add_folder_btn = QPushButton("Add Folder")
        add_folder_btn.clicked.connect(self._open_folders_from_finder)
        actions.addWidget(add_folder_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_paths)
        actions.addWidget(clear_btn)
        actions.addStretch(1)
        outer.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._upload_btn = buttons.addButton("Upload", QDialogButtonBox.ButtonRole.AcceptRole)
        self._upload_btn.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _open_files_from_finder(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images from Finder",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if paths:
            self._add_paths(paths)

    def _open_folders_from_finder(self) -> None:
        dlg = QFileDialog(self, "Select dataset folder(s)")
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        # The non-native dialog lets us enable multi-select for folder ingestion.
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        for view in dlg.findChildren(QAbstractItemView):
            try:
                view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            except Exception:
                pass
        if dlg.exec() == QDialog.DialogCode.Accepted:
            paths = dlg.selectedFiles()
            if paths:
                self._add_paths(paths)

    def _add_paths(self, paths: list[str]) -> None:
        expanded = _expand_upload_paths(paths)
        if not expanded:
            self._render_selected_paths()
            if self._paths:
                self._summary.setText(
                    f"{len(self._paths)} image(s) still queued. No new supported images found."
                )
            else:
                self._summary.setText("No supported images found in the dropped/selected items.")
            return
        seen = set(self._paths)
        added = 0
        for path in expanded:
            if path in seen:
                continue
            self._paths.append(path)
            seen.add(path)
            added += 1
        self._render_selected_paths(added=added, source_count=len(paths))

    def _render_selected_paths(self, *, added: int = 0, source_count: int = 0) -> None:
        self._list.clear()
        for path in self._paths:
            item = QListWidgetItem(Path(path).name)
            item.setToolTip(path)
            self._list.addItem(item)
        if self._paths:
            suffix = ""
            if source_count:
                suffix = f" from {source_count} selected item(s)"
            if added == 0:
                suffix += " (duplicates skipped)"
            self._summary.setText(f"{len(self._paths)} image(s) ready to upload{suffix}.")
        else:
            self._summary.setText("No supported images found in the dropped/selected items.")
        self._upload_btn.setEnabled(bool(self._paths))

    def _clear_paths(self) -> None:
        self._paths = []
        self._list.clear()
        self._summary.setText("No images selected yet.")
        self._upload_btn.setEnabled(False)

    def selected_paths(self) -> list[str]:
        return list(self._paths)

    def create_empty_labels(self) -> bool:
        return bool(self._empty_labels.isChecked())


class _NewDatasetEntryDialog(QDialog):
    def __init__(
        self,
        *,
        source_slug: str,
        selected_count: int,
        filtered_count: int,
        total_count: int,
        default_split: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Database Entry")
        self.resize(520, 340)
        self._counts = {
            "selected": max(0, int(selected_count)),
            "filtered": max(0, int(filtered_count)),
            "all": max(0, int(total_count)),
            "empty": 0,
        }

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        intro = QLabel(
            "Create a new YOLO dataset folder under database/. "
            "When a source dataset is selected, copied entries keep their matching YOLO label files."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name"))
        self._name = QLineEdit()
        base = str(source_slug or "dataset").strip() or "dataset"
        self._name.setText(f"{base}_subset" if source_slug else "new_dataset")
        self._name.setPlaceholderText("new_tiger_subset")
        name_row.addWidget(self._name, stretch=1)
        outer.addLayout(name_row)

        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Source"))
        self._scope = QComboBox()
        if selected_count > 0:
            self._scope.addItem(f"Selected rows ({selected_count})", "selected")
        if filtered_count > 0:
            self._scope.addItem(f"Filtered rows ({filtered_count})", "filtered")
        if total_count > 0:
            self._scope.addItem(f"All rows ({total_count})", "all")
        self._scope.addItem("Empty dataset folder", "empty")
        self._scope.currentIndexChanged.connect(self._sync_limit)
        scope_row.addWidget(self._scope, stretch=1)
        outer.addLayout(scope_row)

        count_row = QHBoxLayout()
        count_row.addWidget(QLabel("Max images"))
        self._max_images = QSpinBox()
        self._max_images.setMinimum(0)
        self._max_images.setSpecialValueText("all")
        count_row.addWidget(self._max_images)
        count_row.addStretch(1)
        outer.addLayout(count_row)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Target split"))
        self._target_split = QComboBox()
        self._target_split.addItem("train", "train")
        self._target_split.addItem("val", "val")
        self._target_split.addItem("test", "test")
        idx = self._target_split.findData(str(default_split or "train"))
        self._target_split.setCurrentIndex(max(0, idx))
        split_row.addWidget(self._target_split)
        split_row.addStretch(1)
        outer.addLayout(split_row)

        self._copy_labels = QCheckBox("Copy YOLO label coordinate files")
        self._copy_labels.setChecked(True)
        outer.addWidget(self._copy_labels)
        self._only_labeled = QCheckBox("Only include images that already have label files")
        self._only_labeled.setChecked(False)
        outer.addWidget(self._only_labeled)
        self._preserve_splits = QCheckBox("Preserve source train/val/test split")
        self._preserve_splits.setChecked(False)
        outer.addWidget(self._preserve_splits)

        self._summary = QLabel("")
        self._summary.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.72);")
        self._summary.setWordWrap(True)
        outer.addWidget(self._summary)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._create_btn = buttons.addButton("Create", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._name.textChanged.connect(self._sync_summary)
        self._max_images.valueChanged.connect(self._sync_summary)
        self._sync_limit()

    def _sync_limit(self) -> None:
        scope = self.scope()
        count = int(self._counts.get(scope, 0))
        self._max_images.setEnabled(scope != "empty" and count > 0)
        self._max_images.setMaximum(max(0, count))
        if scope == "empty":
            self._max_images.setValue(0)
        elif self._max_images.value() <= 0:
            self._max_images.setValue(min(count, 1) if count else 0)
        self._sync_summary()

    def _sync_summary(self) -> None:
        scope = self.scope()
        count = int(self._counts.get(scope, 0))
        enabled = bool(self.dataset_name())
        self._create_btn.setEnabled(enabled)
        if scope == "empty":
            self._summary.setText("Creates an empty YOLO dataset scaffold with images/ and labels/ folders.")
        else:
            cap = self.max_images()
            shown = count if cap <= 0 else min(count, cap)
            self._summary.setText(
                f"Will copy up to {shown} image(s) from the {scope} source set. "
                "Matching label files are copied verbatim when enabled."
            )

    def dataset_name(self) -> str:
        return str(self._name.text() or "").strip()

    def scope(self) -> str:
        return str(self._scope.currentData() or "empty")

    def max_images(self) -> int:
        return int(self._max_images.value())

    def target_split(self) -> str:
        return str(self._target_split.currentData() or "train")

    def copy_labels(self) -> bool:
        return bool(self._copy_labels.isChecked())

    def only_labeled(self) -> bool:
        return bool(self._only_labeled.isChecked())

    def preserve_splits(self) -> bool:
        return bool(self._preserve_splits.isChecked())


class DatasetPanel(SelectablePanel, QWidget):
    """Library picker (repo-root database/) + same thumbnail / label UX as before."""

    panel_entity_type = "dataset"

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
        self._scenario_name = ""
        self._dataset_folder_hint = ""
        self._scenario_backbone_type = ""
        self._scenario_backbone_config: dict[str, Any] = {}
        self._tabular_mode = False
        self._dataset_format = "yolo_detection"
        self._library_selected_value = ""
        self._all_library_entries: list[tuple[str, str]] = []
        self._all_images: list[dict[str, Any]] = []
        self._filtered_images: list[dict[str, Any]] = []
        self._page_index = 0
        self._thumb_cache: "OrderedDict[str, QPixmap]" = OrderedDict()
        self._thumb_cache_cap = 700
        self._thumb_cache_slug = ""
        self._audio_asset_entries: list[dict[str, Any]] = []
        self._dataset_classes: list[str] = []
        self._csv_editor_windows: list[CsvTableEditorDialog] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(3)
        layout.addWidget(splitter, stretch=1)

        catalog_pane = QWidget()
        catalog_pane.setMinimumWidth(28)
        catalog_pane.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        catalog_layout = QVBoxLayout(catalog_pane)
        catalog_layout.setContentsMargins(0, 0, 0, 0)
        catalog_layout.setSpacing(6)

        catalog_head = QHBoxLayout()
        self._catalog_title = QLabel("Database Catalog")
        self._catalog_title.setProperty("isTitle", True)
        catalog_head.addWidget(self._catalog_title, stretch=0)
        catalog_head.addStretch(1)
        self._import_folder_btn = QPushButton("Import Folder...")
        self._import_folder_btn.setToolTip("Copy a complete local dataset folder into the database library.")
        self._import_folder_btn.clicked.connect(self._import_dataset_folder_clicked)
        catalog_head.addWidget(self._import_folder_btn)
        self._new_entry_btn = QPushButton("New Entry...")
        self._new_entry_btn.setToolTip(
            "Create an empty database dataset or clone selected/filtered images into a new YOLO dataset."
        )
        self._new_entry_btn.clicked.connect(self._new_database_entry_clicked)
        catalog_head.addWidget(self._new_entry_btn)
        catalog_layout.addLayout(catalog_head)

        content_pane = QWidget()
        content_pane.setMinimumWidth(28)
        content_pane.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        content_layout = QVBoxLayout(content_pane)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        splitter.addWidget(catalog_pane)
        splitter.addWidget(content_pane)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)

        # Hidden host for widgets retained for API compatibility but no longer
        # displayed (the catalog list on the left is now the library selector).
        self._hidden_lib_host = QWidget()
        self._hidden_lib_host.setVisible(False)
        _hidden_lay = QVBoxLayout(self._hidden_lib_host)
        _hidden_lay.setContentsMargins(0, 0, 0, 0)
        self._library_label = QLabel("Selected:")
        self._library_combo = QComboBox()
        self._library_combo.setMinimumWidth(200)
        self._style_tall_combo(self._library_combo)
        self._library_combo.setVisible(False)
        self._selected_library_value = QLabel("Select a library from the catalog")
        self._selected_library_value.setWordWrap(True)
        _hidden_lay.addWidget(self._library_label)
        _hidden_lay.addWidget(self._library_combo)
        _hidden_lay.addWidget(self._selected_library_value)
        content_layout.addWidget(self._hidden_lib_host)
        self._library_combo.currentIndexChanged.connect(self._on_library_index_changed)

        # Row 1 — Library / scenario header, dataset format, and counts.
        header = QHBoxLayout()
        header.setSpacing(10)
        self._header = QLabel("Library: —")
        self._header.setStyleSheet("font-weight: 600;")
        header.addWidget(self._header)
        header_sep = QFrame()
        header_sep.setFrameShape(QFrame.Shape.VLine)
        header_sep.setFrameShadow(QFrame.Shadow.Plain)
        header_sep.setStyleSheet("color: rgba(120,120,120,0.25);")
        header.addWidget(header_sep)
        self._format_label = QLabel("Format: —")
        self._format_label.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        header.addWidget(self._format_label)
        header.addStretch(1)
        self._count = QLabel("")
        self._count.setStyleSheet("color: rgba(133,153,0,0.6);")
        header.addWidget(self._count)
        content_layout.addLayout(header)

        self._reload_library_btn = QPushButton("Reload list")
        self._reload_library_btn.clicked.connect(self.reload_library_list)

        library_search_row = QHBoxLayout()
        self._library_search_label = QLabel("Database Search:")
        library_search_row.addWidget(self._library_search_label)
        self._library_search_box = QLineEdit()
        self._library_search_box.setPlaceholderText("Filter database libraries...")
        self._library_search_box.setClearButtonEnabled(True)
        self._library_search_box.textChanged.connect(self._on_library_search_changed)
        library_search_row.addWidget(self._library_search_box, stretch=1)
        self._library_search_match_label = QLabel("")
        self._library_search_match_label.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        library_search_row.addWidget(self._library_search_match_label)
        catalog_layout.addLayout(library_search_row)

        self._library_list = QListWidget()
        self._library_list.setAlternatingRowColors(True)
        self._library_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._library_list.setStyleSheet(_accent_item_selection_rules("QListWidget"))
        self._library_list.itemSelectionChanged.connect(self._on_library_catalog_selection)
        catalog_layout.addWidget(self._library_list, stretch=1)

        self._audio_assets_title = QLabel("Audio Source Assets")
        self._audio_assets_title.setStyleSheet("font-size: 11px; font-weight: 600;")
        catalog_layout.addWidget(self._audio_assets_title)
        self._audio_assets_list = QListWidget()
        self._audio_assets_list.setAlternatingRowColors(True)
        self._audio_assets_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._audio_assets_list.setStyleSheet(_accent_item_selection_rules("QListWidget"))
        self._audio_assets_list.itemSelectionChanged.connect(self._on_audio_asset_selection_changed)
        self._audio_assets_list.itemActivated.connect(lambda _item: self._analyze_selected_audio_asset())
        self._audio_assets_list.setMinimumHeight(130)
        catalog_layout.addWidget(self._audio_assets_list, stretch=1)

        # Waveform player — loads automatically when an asset is selected
        self._audio_waveform = AudioWaveformPlayer()
        self._audio_waveform.selection_changed.connect(self._on_waveform_selection_changed)
        self._audio_waveform.selection_cleared.connect(self._on_waveform_selection_cleared)
        catalog_layout.addWidget(self._audio_waveform)

        audio_action_row = QHBoxLayout()
        audio_action_row.setSpacing(6)
        self._audio_analyze_btn = QPushButton("Analyze")
        self._audio_analyze_btn.clicked.connect(self._analyze_selected_audio_asset)
        self._audio_collect_btn = QPushButton("Add to Training")
        self._audio_collect_btn.clicked.connect(self._collect_selected_audio_asset)
        self._audio_copy_btn = QPushButton("Copy Clip")
        self._audio_copy_btn.setToolTip(
            "Extract the selected waveform region to a WAV file of your choice."
        )
        self._audio_copy_btn.clicked.connect(self._copy_selected_audio_clip)
        audio_action_row.addWidget(self._audio_analyze_btn)
        audio_action_row.addWidget(self._audio_collect_btn)
        audio_action_row.addWidget(self._audio_copy_btn)
        catalog_layout.addLayout(audio_action_row)

        audio_label_row = QHBoxLayout()
        audio_label_row.setSpacing(6)
        self._audio_class_label = QLabel("Class")
        audio_label_row.addWidget(self._audio_class_label)
        self._audio_label_edit = QLineEdit()
        self._audio_label_edit.setPlaceholderText("e.g. alarm, speech, engine")
        audio_label_row.addWidget(self._audio_label_edit, stretch=1)
        self._audio_collect_split = QComboBox()
        self._audio_collect_split.addItem("train", "train")
        self._audio_collect_split.addItem("val", "val")
        self._style_tall_combo(self._audio_collect_split)
        audio_label_row.addWidget(self._audio_collect_split)
        catalog_layout.addLayout(audio_label_row)

        audio_range_row = QHBoxLayout()
        audio_range_row.setSpacing(6)
        self._audio_start_label = QLabel("Start")
        audio_range_row.addWidget(self._audio_start_label)
        self._audio_start_ms = QSpinBox()
        self._audio_start_ms.setRange(0, 86_400_000)
        self._audio_start_ms.setSuffix(" ms")
        self._audio_start_ms.setSingleStep(1000)
        self._audio_start_ms.valueChanged.connect(self._on_audio_spinbox_changed)
        audio_range_row.addWidget(self._audio_start_ms)
        self._audio_end_label = QLabel("End")
        audio_range_row.addWidget(self._audio_end_label)
        self._audio_end_ms = QSpinBox()
        self._audio_end_ms.setRange(0, 86_400_000)
        self._audio_end_ms.setSuffix(" ms")
        self._audio_end_ms.setSpecialValueText("full")
        self._audio_end_ms.setSingleStep(1000)
        self._audio_end_ms.valueChanged.connect(self._on_audio_spinbox_changed)
        audio_range_row.addWidget(self._audio_end_ms)
        catalog_layout.addLayout(audio_range_row)

        self._topology_title = QLabel("Folder Topology")
        self._topology_title.setStyleSheet("font-size: 11px; font-weight: 600;")
        catalog_layout.addWidget(self._topology_title)
        self._topology_tree = QTreeWidget()
        self._topology_tree.setHeaderLabels(["Folder / File", "Count"])
        self._topology_tree.setRootIsDecorated(True)
        self._topology_tree.setAlternatingRowColors(True)
        self._topology_tree.setUniformRowHeights(True)
        self._topology_tree.setMinimumHeight(180)
        self._topology_tree.setStyleSheet(_accent_item_selection_rules("QTreeWidget"))
        topo_header = self._topology_tree.header()
        topo_header.setStretchLastSection(False)
        topo_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        topo_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        catalog_layout.addWidget(self._topology_tree, stretch=1)

        # Row 2 — single line of data manipulation: editors, split, browse, paging.
        self._open_editor_btn = QPushButton("Open Annotation Editor")
        self._open_editor_btn.clicked.connect(self._open_annotation_editor_clicked)
        self._open_dataset_editor_btn = QPushButton("Open Dataset Editor")
        self._open_dataset_editor_btn.setToolTip("Bulk edit dataset entries (classes, splits, labels).")
        self._open_dataset_editor_btn.clicked.connect(self._open_dataset_editor_clicked)
        self._open_inventory_btn = QPushButton("Inventory")
        self._open_inventory_btn.setToolTip("Summarize and bulk-manage files by type for the selected dataset folder.")
        self._open_inventory_btn.clicked.connect(self._open_inventory_clicked)
        self._open_csv_editor_btn = QPushButton("Open CSV Editor")
        self._open_csv_editor_btn.setToolTip("Edit the selected CSV dataset (table editor).")
        self._open_csv_editor_btn.clicked.connect(self._open_csv_editor_clicked)
        self._open_csv_editor_btn.setVisible(False)
        self._convert_import_btn = QPushButton("Convert ImageFolder -> YOLO (import labels)")
        self._convert_import_btn.setToolTip(
            "Reuse existing YOLO .txt annotations from sidecars or a mirrored labels/ folder."
        )
        self._convert_import_btn.clicked.connect(lambda: self._convert_imagefolder(mode="import_labels"))
        self._convert_full_btn = QPushButton("Convert ImageFolder -> YOLO (full-frame)")
        self._convert_full_btn.setToolTip("Generate one full-image YOLO box from each ImageFolder class.")
        self._convert_full_btn.clicked.connect(lambda: self._convert_imagefolder(mode="full_frame"))
        self._convert_empty_btn = QPushButton("Convert ImageFolder -> YOLO (empty labels)")
        self._convert_empty_btn.setToolTip("Copy images only and create annotations later in the editor.")
        self._convert_empty_btn.clicked.connect(lambda: self._convert_imagefolder(mode="empty"))

        self._split_combo = QComboBox()
        self._split_combo.addItem("train", "train")
        self._split_combo.addItem("val", "val")
        self._style_tall_combo(self._split_combo)

        self._folder_filter = QComboBox()
        self._style_tall_combo(self._folder_filter)
        self._folder_filter.currentIndexChanged.connect(lambda _i: self._on_nav_changed(reset_page=True))
        self._subfolder_filter = QComboBox()
        self._style_tall_combo(self._subfolder_filter)
        self._subfolder_filter.currentIndexChanged.connect(lambda _i: self._on_nav_changed(reset_page=True))
        self._page_size = QComboBox()
        for n in (30, 60, 120, 240):
            self._page_size.addItem(str(n), int(n))
        self._page_size.setCurrentIndex(1)  # 60
        self._style_tall_combo(self._page_size)
        self._page_size.currentIndexChanged.connect(lambda _i: self._on_nav_changed(reset_page=True))
        self._page_prev = QPushButton("Prev")
        self._page_prev.clicked.connect(lambda: self._change_page(-1))
        self._page_next = QPushButton("Next")
        self._page_next.clicked.connect(lambda: self._change_page(1))
        self._page_spin = QSpinBox()
        self._page_spin.setRange(1, 1)
        self._page_spin.valueChanged.connect(self._on_page_spin_changed)
        self._page_of = QLabel("of 1")
        self._page_of.setStyleSheet("color: rgba(133,153,0,0.65);")
        self._range = QLabel("")
        self._range.setStyleSheet("color: rgba(133,153,0,0.65);")

        def _vsep() -> QFrame:
            f = QFrame()
            f.setFrameShape(QFrame.Shape.VLine)
            f.setFrameShadow(QFrame.Shadow.Plain)
            f.setStyleSheet("color: rgba(120,120,120,0.25);")
            return f

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        toolbar.addWidget(self._reload_library_btn)
        toolbar.addWidget(_vsep())
        toolbar.addWidget(self._open_editor_btn)
        toolbar.addWidget(self._open_dataset_editor_btn)
        toolbar.addWidget(self._open_inventory_btn)
        toolbar.addWidget(self._open_csv_editor_btn)
        toolbar.addWidget(self._convert_import_btn)
        toolbar.addWidget(self._convert_full_btn)
        toolbar.addWidget(self._convert_empty_btn)
        toolbar.addWidget(_vsep())
        toolbar.addWidget(QLabel("Add to:"))
        toolbar.addWidget(self._split_combo)
        toolbar.addWidget(_vsep())
        toolbar.addWidget(QLabel("Folder"))
        toolbar.addWidget(self._folder_filter)
        toolbar.addWidget(QLabel("Subfolder"))
        toolbar.addWidget(self._subfolder_filter)
        toolbar.addWidget(QLabel("Page size"))
        toolbar.addWidget(self._page_size)
        toolbar.addWidget(self._page_prev)
        toolbar.addWidget(QLabel("Page"))
        toolbar.addWidget(self._page_spin)
        toolbar.addWidget(self._page_of)
        toolbar.addWidget(self._range, stretch=1)
        toolbar.addWidget(self._page_next)
        content_layout.addLayout(toolbar)

        search_row = QHBoxLayout()
        self._search_label = QLabel("Search:")
        search_row.addWidget(self._search_label)
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Filter by filename, label, or identity...")
        self._search_box.setClearButtonEnabled(True)
        self._search_box.textChanged.connect(lambda _: self._on_nav_changed(reset_page=True))
        search_row.addWidget(self._search_box, stretch=1)
        self._search_match_label = QLabel("")
        self._search_match_label.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        search_row.addWidget(self._search_match_label)
        content_layout.addLayout(search_row)

        self._list = _ImageTableWidget()
        self._list.filesDropped.connect(self._upload_labeled_paths)
        self._list.itemActivated.connect(self._on_item_activated)
        content_layout.addWidget(self._list, stretch=1)

        self._tabular_preview = QTextEdit()
        self._tabular_preview.setReadOnly(True)
        self._tabular_preview.setPlaceholderText("Select a tabular CSV to preview.")
        self._tabular_preview.setVisible(False)
        content_layout.addWidget(self._tabular_preview, stretch=1)

        self._schema_panel = SchemaFixPanel()
        self._schema_panel.setVisible(False)
        self._schema_panel.revealRequested.connect(self._on_tabular_reveal_csv)
        self._schema_panel.applyRequested.connect(self._on_tabular_apply_schema)
        content_layout.addWidget(self._schema_panel)

        btn_row = QHBoxLayout()
        self._upload_modal_btn = QPushButton("Upload...")
        self._upload_modal_btn.setToolTip("Open a large drag-and-drop uploader with a Finder picker.")
        self._upload_modal_btn.clicked.connect(self._open_upload_dialog)
        btn_row.addWidget(self._upload_modal_btn)
        self._upload_btn = QPushButton("Add Labeled Data...")
        self._upload_btn.clicked.connect(self._pick_labeled_files)
        self._upload_empty_btn = QPushButton("Add Empty-Label Images...")
        self._upload_empty_btn.clicked.connect(self._pick_empty_label_files)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.reload)
        self._delete_btn = QPushButton("Delete Selected")
        self._delete_btn.clicked.connect(self._delete_selected)
        self._delete_library_btn = QPushButton("Delete Library")
        self._delete_library_btn.setToolTip("Remove the entire dataset library (folder and all images).")
        self._delete_library_btn.clicked.connect(self._delete_library)
        btn_row.addWidget(self._upload_btn)
        btn_row.addWidget(self._upload_empty_btn)
        btn_row.addWidget(self._refresh_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._delete_btn)
        btn_row.addWidget(self._delete_library_btn)
        content_layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.6);")
        content_layout.addWidget(self._status)
        self._apply_dataset_format(self._dataset_format)
        self._apply_audio_asset_controls(False)
        self._refresh_nav_controls_enabled()

    def _apply_tabular_mode(self, enabled: bool) -> None:
        """Toggle between CV image-library UX and ML tabular CSV UX."""
        self._tabular_mode = bool(enabled)
        if self._tabular_mode:
            try:
                self._library_label.setText("Selected Tabular:")
                self._library_search_label.setText("Tabular Search:")
                self._library_search_box.setPlaceholderText("Filter tabular datasets...")
            except Exception:
                pass
            self._apply_dataset_format("csv_tabular")
        else:
            try:
                self._library_label.setText("Selected Database:")
                self._library_search_label.setText("Database Search:")
                self._library_search_box.setPlaceholderText("Filter database libraries...")
            except Exception:
                pass
            self._apply_dataset_format(self._dataset_format or "yolo_detection")

        # Toggle widgets that are image-only.
        for w in (
            self._open_editor_btn,
            self._open_dataset_editor_btn,
            self._open_inventory_btn,
            self._convert_import_btn,
            self._convert_full_btn,
            self._convert_empty_btn,
            self._split_combo,
            self._folder_filter,
            self._subfolder_filter,
            self._page_size,
            self._page_prev,
            self._page_next,
            self._page_spin,
            self._page_of,
            self._range,
            self._upload_modal_btn,
            self._upload_btn,
            self._upload_empty_btn,
            self._delete_btn,
            self._delete_library_btn,
            self._search_label,
            self._search_box,
            self._search_match_label,
        ):
            try:
                w.setVisible(not self._tabular_mode)
            except Exception:
                pass
        try:
            self._list.setVisible(not self._tabular_mode)
        except Exception:
            pass
        try:
            self._open_csv_editor_btn.setVisible(self._tabular_mode)
        except Exception:
            pass
        try:
            self._tabular_preview.setVisible(self._tabular_mode)
        except Exception:
            pass
        try:
            self._schema_panel.setVisible(self._tabular_mode)
        except Exception:
            pass
        try:
            self._topology_title.setVisible(not self._tabular_mode)
            self._topology_tree.setVisible(not self._tabular_mode)
        except Exception:
            pass
        self._apply_audio_asset_controls(
            self._scenario_backbone_type == "audio_recognition" and not self._tabular_mode
        )

    def _apply_audio_asset_controls(self, enabled: bool) -> None:
        visible = bool(enabled)
        for w in (
            getattr(self, "_audio_assets_title", None),
            getattr(self, "_audio_assets_list", None),
            getattr(self, "_audio_waveform", None),
            getattr(self, "_audio_analyze_btn", None),
            getattr(self, "_audio_collect_btn", None),
            getattr(self, "_audio_copy_btn", None),
            getattr(self, "_audio_class_label", None),
            getattr(self, "_audio_label_edit", None),
            getattr(self, "_audio_collect_split", None),
            getattr(self, "_audio_start_label", None),
            getattr(self, "_audio_start_ms", None),
            getattr(self, "_audio_end_label", None),
            getattr(self, "_audio_end_ms", None),
        ):
            if w is None:
                continue
            try:
                w.setVisible(visible)
            except Exception:
                pass
        self._sync_audio_asset_action_state()

    def _sync_audio_asset_action_state(self) -> None:
        is_audio = self._scenario_backbone_type == "audio_recognition"
        selected = self._current_audio_asset_entry()
        has_source = bool(selected.get("path")) if selected else False
        slug, _enc = self._library_slug_encoded()
        has_sel = (
            self._audio_waveform.sel_start_ms is not None
            and self._audio_waveform.sel_end_ms is not None
        )
        try:
            self._audio_analyze_btn.setEnabled(is_audio and has_source)
            self._audio_collect_btn.setEnabled(is_audio and has_source and bool(slug))
            self._audio_copy_btn.setEnabled(is_audio and has_source)
        except Exception:
            pass

    def refresh_responsive_layout(self) -> None:
        try:
            self._list._sync_table_geometry()
        except Exception:
            return

    def _style_tall_combo(self, combo: QComboBox) -> None:
        """Make combo and its popup list ~50% taller for easier targeting."""
        base_h = max(combo.sizeHint().height(), 1)
        combo.setMinimumHeight(int(min(34, max(base_h + 4, base_h * 1.3))))
        view = combo.view()
        if view is None:
            return
        fm = QFontMetrics(combo.font())
        row = max(fm.height() + 8, 1)
        item_min = int(min(36, max(row + 3, row * 1.3)))
        view.setStyleSheet(
            f"QAbstractItemView::item {{ min-height: {item_min}px; padding: 4px 10px; }}"
        )

    def _refresh_nav_controls_enabled(self) -> None:
        enabled = True
        # Even for ImageFolder datasets, browsing is still useful; only hide it for "no dataset".
        for w in (
            getattr(self, "_folder_filter", None),
            getattr(self, "_subfolder_filter", None),
            getattr(self, "_page_size", None),
            getattr(self, "_page_prev", None),
            getattr(self, "_page_next", None),
            getattr(self, "_page_spin", None),
        ):
            if w is None:
                continue
            try:
                w.setEnabled(enabled)
            except Exception:
                pass

    @staticmethod
    def _entry_subfolder(entry: dict[str, Any]) -> str:
        # ImageFolder classification sources already provide the class folder.
        cl = str(entry.get("classification_label") or "").strip()
        if cl:
            return cl
        rel_path = str(entry.get("relative_path") or "")
        if not rel_path:
            return ""
        parts = Path(rel_path).parts
        # YOLO-style: images/<split>/<class>/.../img.jpg
        if "images" in parts:
            idx = parts.index("images")
            split = str(entry.get("split") or "")
            if split and split != "root":
                # Need at least one path part after the class folder to treat it as a folder.
                if len(parts) > idx + 3:
                    return str(parts[idx + 2])
                return ""
            # Root layout: images/<class>/.../img.jpg
            if len(parts) > idx + 2:
                return str(parts[idx + 1])
        return ""

    @staticmethod
    def _split_sort_key(name: str) -> tuple[int, str]:
        v = str(name or "")
        order = {"train": 0, "val": 1, "valid": 2, "test": 3, "root": 9}
        return (order.get(v.lower(), 5), v.lower())

    def _populate_filters(self) -> None:
        """Refresh Folder/Subfolder filter options based on current dataset listing."""
        prev_folder = str(self._folder_filter.currentData() or "")
        prev_sub = str(self._subfolder_filter.currentData() or "")

        splits = sorted(
            {str(im.get("split") or "") for im in self._all_images if str(im.get("split") or "")},
            key=self._split_sort_key,
        )
        self._folder_filter.blockSignals(True)
        self._folder_filter.clear()
        self._folder_filter.addItem("All", "")
        for s in splits:
            self._folder_filter.addItem(s, s)
        idx = self._folder_filter.findData(prev_folder)
        self._folder_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self._folder_filter.blockSignals(False)

        self._update_subfolder_filter(prev_sub)

    def _update_subfolder_filter(self, prefer_value: str = "") -> None:
        folder = str(self._folder_filter.currentData() or "")
        candidates = self._all_images
        if folder:
            folder_l = folder.lower()
            candidates = [im for im in candidates if str(im.get("split") or "").lower() == folder_l]

        subs = sorted(
            {self._entry_subfolder(im) for im in candidates if self._entry_subfolder(im)},
            key=lambda s: s.lower(),
        )
        self._subfolder_filter.blockSignals(True)
        self._subfolder_filter.clear()
        self._subfolder_filter.addItem("All", "")
        for s in subs:
            self._subfolder_filter.addItem(s, s)
        idx = self._subfolder_filter.findData(prefer_value)
        self._subfolder_filter.setCurrentIndex(idx if idx >= 0 else 0)
        self._subfolder_filter.blockSignals(False)

    def _on_nav_changed(self, *, reset_page: bool) -> None:
        # Folder change affects available subfolders.
        self._update_subfolder_filter(str(self._subfolder_filter.currentData() or ""))
        if reset_page:
            self._page_index = 0
        self._apply_filters_and_render()

    def _change_page(self, delta: int) -> None:
        if not self._filtered_images:
            return
        page_count = self._page_count()
        if page_count <= 1:
            return
        self._page_index = max(0, min(page_count - 1, self._page_index + int(delta)))
        self._render_page()

    def _on_page_spin_changed(self, value: int) -> None:
        page = int(value or 1)
        self._page_index = max(0, page - 1)
        self._render_page()

    def _page_size_value(self) -> int:
        try:
            v = int(self._page_size.currentData() or 60)
        except Exception:
            v = 60
        return max(10, min(1000, v))

    def _page_count(self) -> int:
        total = len(self._filtered_images)
        size = self._page_size_value()
        return max(1, int(math.ceil(total / float(size))) if total else 1)

    def _apply_filters_and_render(self) -> None:
        images = list(self._all_images)
        folder = str(self._folder_filter.currentData() or "")
        sub = str(self._subfolder_filter.currentData() or "")
        if folder:
            folder_l = folder.lower()
            images = [im for im in images if str(im.get("split") or "").lower() == folder_l]
        if sub:
            sub_l = sub.lower()
            images = [im for im in images if self._entry_subfolder(im).lower() == sub_l]

        # Search filter: match against filename, display name, label/identity, or split.
        try:
            search = str(self._search_box.text() or "").strip().lower()
        except Exception:
            search = ""
        if search:
            filtered: list[dict[str, Any]] = []
            for im in images:
                haystack = " ".join([
                    str(im.get("name") or ""),
                    str(im.get("display_name") or ""),
                    str(im.get("classification_label") or ""),
                    str(im.get("split") or ""),
                    str(im.get("stem") or ""),
                ]).lower()
                if search in haystack:
                    filtered.append(im)
            images = filtered

        # Update search match label.
        try:
            total_all = len(self._all_images)
            if search and total_all:
                self._search_match_label.setText(f"{len(images)} / {total_all} match")
            else:
                self._search_match_label.setText("")
        except Exception:
            pass

        self._filtered_images = images
        # Clamp page index and render.
        self._page_index = max(0, min(self._page_count() - 1, self._page_index))
        self._render_page()

    def _render_page(self) -> None:
        self._list.setRowCount(0)
        self._list._sync_table_geometry()
        slug, enc_slug = self._library_slug_encoded()
        if not slug:
            return
        if slug != self._thumb_cache_slug:
            # Avoid cross-dataset thumbnail collisions.
            self._thumb_cache_slug = slug
            self._thumb_cache.clear()

        total = len(self._filtered_images)
        size = self._page_size_value()
        page_count = self._page_count()
        page_idx = max(0, min(page_count - 1, self._page_index))
        start = page_idx * size
        end = min(total, start + size)

        # Update nav controls.
        self._page_prev.setEnabled(page_idx > 0)
        self._page_next.setEnabled(page_idx < page_count - 1)
        self._page_spin.blockSignals(True)
        self._page_spin.setRange(1, max(1, page_count))
        self._page_spin.setValue(page_idx + 1)
        self._page_spin.blockSignals(False)
        self._page_of.setText(f"of {page_count}")
        if total:
            self._range.setText(f"Showing {start + 1}-{end} of {total}")
        else:
            self._range.setText("No matches")

        # Grab current raw search term (original casing) for highlight rendering.
        try:
            active_search = str(self._search_box.text() or "").strip()
        except Exception:
            active_search = ""

        page_images = self._filtered_images[start:end]
        is_audio = self._dataset_format == "audiofolder_classification"
        for img in page_images:
            rel_path = str(img.get("relative_path") or img.get("name") or "")
            display_name = str(img.get("display_name") or rel_path)
            has_label = bool(img.get("has_label"))
            classification_label = str(img.get("classification_label") or "")
            if is_audio:
                row = self._list.add_audio_row(
                    rel_path=rel_path,
                    display_name=display_name,
                    split=str(img.get("split") or "root"),
                    classification_label=classification_label,
                    size=img.get("size"),
                    active_search=active_search,
                )
            else:
                row = self._list.add_image_row(
                    rel_path=rel_path,
                    display_name=display_name,
                    split=str(img.get("split") or "root"),
                    has_label=has_label,
                    classification_label=classification_label,
                    size=img.get("size"),
                    active_search=active_search,
                )

            if is_audio:
                continue

            # Thumbnail fetch with a small LRU cache.
            cache_key = f"{slug}::{rel_path}" if rel_path else ""
            if cache_key and cache_key in self._thumb_cache:
                pix = self._thumb_cache.pop(cache_key)
                self._thumb_cache[cache_key] = pix  # move to end
                self._list.set_thumbnail_pixmap(row, pix)
                continue
            try:
                encoded = urllib.parse.quote(rel_path, safe="")
                thumb = self._http_get(f"/database/{enc_slug}/thumb/{encoded}")
                b64 = str(thumb.get("thumb_b64") or "")
                if b64:
                    pix = pixmap_from_b64_jpeg(b64)
                    if not pix.isNull():
                        self._list.set_thumbnail_pixmap(row, pix)
                        if cache_key:
                            self._thumb_cache[cache_key] = pix
                            while len(self._thumb_cache) > self._thumb_cache_cap:
                                try:
                                    self._thumb_cache.popitem(last=False)
                                except Exception:
                                    break
            except Exception:
                pass

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

    def _on_library_index_changed(self, _index: int) -> None:
        self._library_selected_value = str(self._library_combo.currentData() or "").strip()
        self._selected_library_value.setText(self._library_selected_value or "Select a library from the catalog")
        self._sync_library_catalog_selection(self._library_selected_value)
        if self._library_selected_value:
            self.emit_entity_selected("dataset", self._library_selected_value)
        # Clear search when switching datasets so stale terms don't carry over.
        try:
            self._search_box.blockSignals(True)
            self._search_box.clear()
            self._search_match_label.setText("")
            self._search_box.blockSignals(False)
        except Exception:
            pass
        self._sync_header()
        self._sync_audio_asset_action_state()
        if self._tabular_mode:
            rel = str(self._library_combo.currentData() or "").strip()
            scen = str(self._scenario_name or "").strip()
            current = str(self._scenario_backbone_config.get("dataset_csv") or "").strip()
            if scen and rel and rel != current:
                try:
                    self._http_json_direct(
                        "POST",
                        f"/scenarios/{scen}/backbone_config",
                        {"patch": {"dataset_csv": rel}},
                        timeout=30.0,
                    )
                    self._scenario_backbone_config["dataset_csv"] = rel
                    self.datasetChanged.emit(scen)
                except Exception as exc:
                    msg = f"Failed to apply dataset_csv to scenario: {exc}"
                    self._status.setText(msg)
                    self.errorRaised.emit(msg)
        elif self._scenario_backbone_type in {"yolo_detection", "face_recognition", "audio_recognition"}:
            rel = str(self._library_combo.currentData() or "").strip()
            scen = str(self._scenario_name or "").strip()
            current = str(self._dataset_folder_hint or "").strip()
            if scen and rel and rel != current:
                try:
                    self._http_post(f"/scenarios/{scen}/dataset", {"dataset": rel})
                    self._dataset_folder_hint = rel
                    self.datasetChanged.emit(scen)
                except Exception as exc:
                    msg = f"Failed to apply dataset to scenario: {exc}"
                    self._status.setText(msg)
                    self.errorRaised.emit(msg)
        self.reload()

    def _on_library_search_changed(self, _text: str) -> None:
        preferred = str(self._library_combo.currentData() or self._library_selected_value or "").strip()
        self._apply_library_filter(preferred=preferred)

    def _on_library_catalog_selection(self) -> None:
        item = self._library_list.currentItem()
        value = str(item.data(Qt.ItemDataRole.UserRole) or "").strip() if item is not None else ""
        if not value:
            return
        idx = self._library_combo.findData(value)
        if idx >= 0 and idx != self._library_combo.currentIndex():
            self._library_combo.setCurrentIndex(idx)

    def select_library(self, value: str) -> bool:
        """Public hook: select the catalog entry whose UserRole equals ``value``.

        Used by Collect mode to keep this panel in sync with the active scrape
        job. Returns True when a matching entry was found.
        """
        target = str(value or "").strip()
        if not target:
            return False
        idx = self._library_combo.findData(target)
        if idx < 0:
            # The entry may not be in the catalog yet (e.g. a fresh scrape job
            # before promotion). Reload once and retry — silent miss is fine.
            try:
                self.reload_library_list()
            except Exception:
                return False
            idx = self._library_combo.findData(target)
        if idx < 0:
            return False
        if idx != self._library_combo.currentIndex():
            self._library_combo.setCurrentIndex(idx)
        else:
            self._sync_library_catalog_selection(target)
        return True

    def show_tabular_dataset(self, value: str) -> None:
        """Public hook (Collect mode): flip into tabular mode and select a tabular dataset.

        Accepts either the dataset slug or its relative csv path; the tabular library
        combo keys entries by relative path, so both are matched.
        """
        name = str(value or "").strip()
        try:
            self._apply_tabular_mode(True)
        except Exception:
            pass
        if not name:
            return
        try:
            self.reload_library_list()
        except Exception:
            pass
        if self.select_library(name):
            return
        combo = self._library_combo
        for i in range(combo.count()):
            data = str(combo.itemData(i) or "")
            if not data:
                continue
            if data == name or Path(data).name == f"{name}.csv" or Path(data).stem == name or data.endswith(f"/{name}.csv"):
                combo.setCurrentIndex(i)
                return

    def _suggest_label_cols(self, columns: list[str]) -> list[str]:
        cols = [str(c).strip() for c in (columns or []) if str(c).strip()]
        lower_map = {c.lower(): c for c in cols}
        suggestions: list[str] = []
        for key in (
            "label",
            "labels",
            "target",
            "y",
            "class",
            "outcome",
            "price",
            "saleprice",
            "sellingprice",
        ):
            if key in lower_map:
                suggestions.append(lower_map[key])
        # Common suffix/prefix patterns.
        for c in cols:
            cl = c.lower()
            if cl.endswith(("_label", "_target")) or cl.startswith(("label_", "target_")):
                suggestions.append(c)
        # Default: last column.
        if cols:
            suggestions.append(cols[-1])
        # De-dupe preserving order.
        out: list[str] = []
        seen: set[str] = set()
        for c in suggestions:
            if c in seen:
                continue
            if c in cols:
                out.append(c)
                seen.add(c)
        return out

    def _on_tabular_reveal_csv(self, dataset_csv: str) -> None:
        if dataset_csv:
            reveal_in_finder(dataset_csv)

    def _on_tabular_apply_schema(self, patch: object, rerun: bool) -> None:
        scen = str(self._scenario_name or "").strip()
        if not scen:
            self._schema_panel.set_status("Select a scenario first.")
            return
        if not isinstance(patch, dict) or not patch.get("label_col"):
            self._schema_panel.set_status("Choose a label column first.")
            return
        self._schema_panel.set_status("")
        try:
            self._http_json_direct(
                "POST",
                f"/scenarios/{scen}/backbone_config",
                {"patch": patch},
                timeout=30.0,
            )
            for k, v in patch.items():
                self._scenario_backbone_config[k] = v
            self.datasetChanged.emit(scen)
        except Exception as exc:
            msg = f"Failed to apply schema: {exc}"
            self._schema_panel.set_status(msg)
            self.errorRaised.emit(msg)
            return
        if rerun:
            try:
                self._http_json_direct("POST", f"/scenarios/{scen}/train", None, timeout=30.0)
            except Exception as exc:
                msg = f"Schema applied, but training re-run failed: {exc}"
                self._schema_panel.set_status(msg)
                self.errorRaised.emit(msg)
                return
        self._schema_panel.set_status("Applied.")

    def _sync_header(self) -> None:
        slug = str(self._library_combo.currentData() or "")
        self._selected_library_value.setText(slug or "Select a library from the catalog")
        if self._tabular_mode:
            parts = [f"Tabular: {Path(slug).name if slug else '—'}"]
        elif self._scenario_backbone_type == "audio_recognition":
            parts = [f"Audio Assets: {slug or '—'}"]
        else:
            parts = [f"Library: {slug or '—'}"]
        if self._scenario_name:
            parts.append(f"scenario: {self._scenario_name}")
        self._header.setText("   |   ".join(parts))

    def _library_slug_encoded(self) -> tuple[str, str]:
        slug = str(self._library_combo.currentData() or "")
        return slug, urllib.parse.quote(slug, safe="")

    def _apply_dataset_format(self, fmt: str) -> None:
        value = str(fmt or "").strip() or "unknown"
        self._dataset_format = value
        human = {
            "yolo_detection": "YOLO detection (images/ + labels/)",
            "imagefolder_classification": "ImageFolder classification (split/class/image)",
            "audiofolder_classification": "AudioFolder classification (split/class/audio)",
            "csv_tabular": "CSV / tabular data",
            "face_csv": "Face recognition (CSV + identities)",
            "unknown": "Unknown",
        }.get(value, value)
        self._format_label.setText(f"Format: {human}")

        is_imagefolder = value == "imagefolder_classification"
        is_audio = value == "audiofolder_classification"
        is_csv = value == "csv_tabular"
        is_face_csv = value == "face_csv"
        yolo_ops_ok = not is_imagefolder and not is_audio and not is_csv and not is_face_csv

        if is_imagefolder:
            self._format_label.setStyleSheet(
                "font-size: 10px; font-weight: 700; color: rgba(220,50,47,0.9);"
            )
        elif is_csv:
            self._format_label.setStyleSheet(
                "font-size: 10px; font-weight: 600; color: rgba(42,161,152,0.9);"
            )
        elif is_face_csv:
            self._format_label.setStyleSheet(
                "font-size: 10px; font-weight: 600; color: rgba(108,113,196,0.9);"
            )
        else:
            self._format_label.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")

        try:
            self._list.set_audio_mode(is_audio)
        except Exception:
            pass

        # Annotation editor and conversion tools are image/YOLO-only.
        self._open_editor_btn.setVisible(not is_audio and not is_csv and not is_face_csv)
        self._open_dataset_editor_btn.setVisible(not is_audio and not is_csv and not is_face_csv)
        self._open_csv_editor_btn.setVisible(bool(is_csv))
        self._open_editor_btn.setText("Convert to YOLO" if is_imagefolder else "Open Annotation Editor")
        self._convert_import_btn.setVisible(is_imagefolder)
        self._convert_full_btn.setVisible(is_imagefolder)
        self._convert_empty_btn.setVisible(is_imagefolder)
        self._new_entry_btn.setEnabled(True)

        # Upload/image controls are YOLO-only. Audio clips enter through the audio collector.
        for w in (self._split_combo, self._upload_modal_btn, self._upload_btn, self._upload_empty_btn):
            try:
                w.setVisible(not is_audio and not is_csv and not is_face_csv)
                w.setEnabled(yolo_ops_ok)
            except Exception:
                pass
        try:
            self._list.setAcceptDrops(yolo_ops_ok)
        except Exception:
            pass

    def _open_annotation_editor_clicked(self) -> None:
        self._open_annotation_editor(start_relative_path="")

    def _open_dataset_editor_clicked(self) -> None:
        slug, _enc = self._library_slug_encoded()
        if not slug:
            return
        dlg = DatasetEditorDialog(base_url=self._base_url, dataset_slug=slug, parent=self)
        dlg.exec()
        self.reload()

    def _open_inventory_clicked(self) -> None:
        slug, _enc = self._library_slug_encoded()
        if not slug:
            return
        dlg = FolderInventoryDialog(base_url=self._base_url, dataset_slug=slug, parent=self)
        dlg.exec()
        self.reload()

    def _open_csv_editor_clicked(self) -> None:
        rel = str(self._library_combo.currentData() or "").strip()
        if not rel:
            return
        p = Path(rel)
        if not p.is_absolute():
            p = (Path(ROOT_DIR) / p).resolve()
        dlg = CsvTableEditorDialog(csv_path=p, parent=None)
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._csv_editor_windows.append(dlg)
        dlg.destroyed.connect(lambda _obj=None, window=dlg: self._on_csv_editor_closed(window))
        dlg.show()
        try:
            dlg.raise_()
            dlg.activateWindow()
        except Exception:
            pass

    def _on_csv_editor_closed(self, window: CsvTableEditorDialog) -> None:
        try:
            self._csv_editor_windows.remove(window)
        except ValueError:
            pass
        self.reload()

    def _open_annotation_editor(self, *, start_relative_path: str = "") -> None:
        slug, enc_slug = self._library_slug_encoded()
        if not slug:
            return
        fmt = self._dataset_format
        try:
            payload = self._http_get(f"/database/{enc_slug}")
            fmt = str(payload.get("format") or fmt)
        except Exception:
            fmt = self._dataset_format
        out_slug = slug
        if fmt == "imagefolder_classification":
            detection_label_count = 0
            if isinstance(payload, dict):
                try:
                    detection_label_count = int(payload.get("detection_label_count") or 0)
                except Exception:
                    detection_label_count = 0
            convert_mode = "import_labels" if detection_label_count > 0 else "empty"
            mode_note = "import labels" if convert_mode == "import_labels" else "empty labels"
            self._status.setText(f"Converting '{slug}' to YOLO ({mode_note})...")
            try:
                payload = self._http_json_direct(
                    "POST",
                    f"/database/{enc_slug}/convert/imagefolder_to_yolo",
                    {"mode": convert_mode, "include_test": True},
                    timeout=120.0,
                )
            except Exception as exc:
                msg = f"Conversion failed for '{slug}': {exc}"
                self._status.setText(msg)
                self.errorRaised.emit(msg)
                return
            out_slug = str(payload.get("output_slug") or "") or slug
            # Refresh and select the output dataset so the user sees the converted tree.
            self.reload_library_list()
            idx = self._library_combo.findData(out_slug)
            if idx >= 0:
                self._library_combo.setCurrentIndex(idx)

        classes_override: list[str] = []
        if self._scenario_name and out_slug == self._dataset_folder_hint:
            try:
                enc_scen = urllib.parse.quote(self._scenario_name, safe="")
                status = self._http_get(f"/scenarios/{enc_scen}/status")
                raw = status.get("classes") if isinstance(status, dict) else []
                if isinstance(raw, list) and all(isinstance(c, str) for c in raw):
                    classes_override = list(raw)
            except Exception:
                classes_override = []

        if not classes_override:
            try:
                final_enc = urllib.parse.quote(out_slug, safe="")
                ds_payload = self._http_get(f"/database/{final_enc}")
                if isinstance(ds_payload, dict):
                    raw_ds = ds_payload.get("classes") or []
                    if isinstance(raw_ds, list) and all(isinstance(c, str) for c in raw_ds):
                        classes_override = [str(c) for c in raw_ds if str(c)]
            except Exception:
                pass

        dlg = AnnotationEditorDialog(
            base_url=self._base_url,
            dataset_slug=out_slug,
            start_relative_path=start_relative_path,
            classes_override=classes_override,
            parent=self,
        )
        dlg.exec()
        self.reload()

    def _convert_imagefolder(self, *, mode: str) -> None:
        slug, enc_slug = self._library_slug_encoded()
        if not slug:
            return
        self._status.setText(f"Converting '{slug}' to YOLO ({mode})...")
        try:
            payload = self._http_json_direct(
                "POST",
                f"/database/{enc_slug}/convert/imagefolder_to_yolo",
                {"mode": str(mode or "full_frame")},
                timeout=120.0,
            )
        except Exception as exc:
            msg = f"Conversion failed for '{slug}': {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        out_slug = str(payload.get("output_slug") or "")
        converted = int(payload.get("converted") or 0)
        errors = payload.get("errors") or []
        msg = f"Converted {converted} image(s) to '{out_slug}'."
        if mode == "import_labels":
            imported = int(payload.get("imported_labels") or 0)
            missing = int(payload.get("missing_labels") or 0)
            normalized = int(payload.get("normalized_label_lines") or 0)
            invalid = int(payload.get("invalid_label_lines") or 0)
            msg += f" Imported {imported} existing label file(s)."
            if normalized:
                msg += f" Normalized {normalized} coord-only line(s)."
            if invalid:
                msg += f" Skipped {invalid} invalid label line(s)."
            if missing:
                msg += f" {missing} image(s) still need labels."
        elif mode == "empty":
            msg += " Images are ready for annotation."
        class_to_id = payload.get("class_to_id") if isinstance(payload, dict) else None
        if isinstance(class_to_id, dict) and mode == "full_frame":
            pairs = []
            for k in sorted(class_to_id.keys(), key=lambda s: str(s).lower()):
                try:
                    pairs.append(f"{k}={int(class_to_id.get(k))}")
                except Exception:
                    continue
            if pairs:
                msg += " IDs: " + ", ".join(pairs[:8]) + ("..." if len(pairs) > 8 else "")
        if errors:
            msg += f" Errors: {errors[0]}"

        # Refresh list and jump to the output dataset if possible.
        self.reload_library_list()
        if out_slug:
            idx = self._library_combo.findData(out_slug)
            if idx >= 0:
                self._library_combo.setCurrentIndex(idx)
        self._status.setText(msg)

    def _pick_existing_folders(self, title: str) -> list[str]:
        dlg = QFileDialog(self, title)
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        for view in dlg.findChildren(QAbstractItemView):
            try:
                view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            except Exception:
                pass
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return []
        return [str(p) for p in dlg.selectedFiles() if str(p or "").strip()]

    def _import_dataset_folder_clicked(self) -> None:
        folders = self._pick_existing_folders("Import dataset folder(s)")
        if not folders:
            return
        imported: list[str] = []
        errors: list[str] = []
        for idx, folder in enumerate(folders, start=1):
            self._status.setText(f"Importing dataset folder {idx}/{len(folders)}: {folder}")
            app = QApplication.instance()
            if app is not None:
                app.processEvents()
            try:
                payload = self._http_post("/database/import_folder", {"source_path": folder})
            except Exception as exc:
                errors.append(f"{Path(folder).name}: {exc}")
                continue
            slug = str((payload or {}).get("slug") or "").strip()
            if slug:
                imported.append(slug)

        if imported:
            self._library_selected_value = imported[-1]
            self.reload_library_list()
        if errors:
            msg = (
                f"Imported {len(imported)} folder(s). Errors: "
                + " | ".join(errors[:3])
            )
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        if imported:
            self._status.setText(
                f"Imported {len(imported)} dataset folder(s): {', '.join(imported[:4])}"
                + ("..." if len(imported) > 4 else "")
            )

    @staticmethod
    def _entry_rel_paths(entries: list[dict[str, Any]]) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            rel = str(entry.get("relative_path") or entry.get("name") or "").strip().lstrip("/").replace("\\", "/")
            if not rel or rel in seen:
                continue
            paths.append(rel)
            seen.add(rel)
        return paths

    def _new_database_entry_clicked(self) -> None:
        slug, enc_slug = self._library_slug_encoded()
        selected_paths = self._list.selected_relative_paths() if slug else []
        filtered_paths = self._entry_rel_paths(self._filtered_images) if slug else []
        all_paths = self._entry_rel_paths(self._all_images) if slug else []
        dlg = _NewDatasetEntryDialog(
            source_slug=slug,
            selected_count=len(selected_paths),
            filtered_count=len(filtered_paths),
            total_count=len(all_paths),
            default_split=self.current_split(),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        name = dlg.dataset_name()
        if not name:
            return

        scope = dlg.scope()
        if not slug or scope == "empty":
            self._status.setText(f"Creating empty database dataset '{name}'...")
            try:
                payload = self._http_json_direct(
                    "POST",
                    "/database/create_yolo_template",
                    {"name": name, "classes": self._dataset_classes or ["object"], "unique": True},
                    timeout=60.0,
                )
            except Exception as exc:
                msg = f"New database entry failed: {exc}"
                self._status.setText(msg)
                self.errorRaised.emit(msg)
                return
            out_slug = str(payload.get("slug") or "").strip()
            if out_slug:
                self._library_selected_value = out_slug
                self.reload_library_list()
                self.datasetChanged.emit(out_slug)
                self._status.setText(f"Created empty dataset '{out_slug}'. Add images or open the annotation editor next.")
            return

        if scope == "selected":
            rels = selected_paths
        elif scope == "filtered":
            rels = filtered_paths
        else:
            rels = all_paths
        limit = dlg.max_images()
        if limit > 0:
            rels = rels[:limit]
        if not rels:
            msg = "New database entry failed: no image rows selected for the chosen source."
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return

        self._status.setText(f"Creating '{name}' from {len(rels)} image(s) in '{slug}'...")
        try:
            payload = self._http_json_direct(
                "POST",
                f"/database/{enc_slug}/clone_subset",
                {
                    "name": name,
                    "relative_paths": rels,
                    "max_images": 0,
                    "target_split": dlg.target_split(),
                    "preserve_splits": dlg.preserve_splits(),
                    "include_labels": dlg.copy_labels(),
                    "only_labeled": dlg.only_labeled(),
                    "unique": True,
                },
                timeout=180.0,
            )
        except Exception as exc:
            msg = f"New database entry failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        out_slug = str(payload.get("output_slug") or "").strip()
        copied = int(payload.get("copied") or 0)
        errors = payload.get("errors") if isinstance(payload, dict) else []
        if out_slug:
            self._library_selected_value = out_slug
            self.reload_library_list()
            self.datasetChanged.emit(out_slug)
        msg = f"Created '{out_slug or name}' with {copied} copied image(s)."
        if errors:
            msg += f" {len(errors)} copy issue(s); first: {errors[0]}"
        else:
            msg += " Labels were preserved where available."
        self._status.setText(msg)

    def set_scenario(
        self,
        scenario: str,
        dataset_folder: str = "",
        backbone_type: str = "",
        backbone_config: Optional[dict[str, Any]] = None,
    ) -> None:
        new_scenario = str(scenario or "").strip()
        if new_scenario != self._scenario_name:
            # Scenario switched — clear the user's explicit pick so the new
            # scenario's configured dataset hint takes effect on next reload.
            self._library_selected_value = ""
        self._scenario_name = new_scenario
        self._dataset_folder_hint = str(dataset_folder or "").strip()
        self._scenario_backbone_type = str(backbone_type or "").strip().lower()
        self._scenario_backbone_config = dict(backbone_config or {})
        self._apply_tabular_mode(self._scenario_backbone_type == "torch_tabular")
        is_audio = self._scenario_backbone_type == "audio_recognition"
        try:
            self._catalog_title.setText("Audio Assets" if is_audio else "Database Catalog")
            self._library_search_label.setText("Audio Search:" if is_audio else "Database Search:")
            self._library_search_box.setPlaceholderText(
                "Filter audio asset datasets..." if is_audio else "Filter database libraries..."
            )
            self._import_folder_btn.setToolTip(
                "Copy a complete local AudioFolder dataset into assets/ml_audio."
                if is_audio
                else "Copy a complete local dataset folder into the database library."
            )
            self._search_box.setPlaceholderText(
                "Filter by filename, class, or split..."
                if is_audio
                else "Filter by filename, label, or identity..."
            )
        except Exception:
            pass
        self._apply_audio_asset_controls(is_audio)
        self._load_audio_assets()
        self.reload_library_list()

    def reload_library_list(self) -> None:
        # _library_selected_value: user explicitly chose this in the current session.
        # _dataset_folder_hint: scenario's server-configured dataset (used as fallback).
        # Combo currentData is only used as last resort so stale UI state never
        # overrides either of the above.
        pending = str(self._library_selected_value or "")
        try:
            payload = self._http_get("/database")
        except Exception as exc:
            msg = f"Database list failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        if self._scenario_backbone_type == "audio_recognition":
            self._load_audio_assets()

        if self._tabular_mode:
            entries = payload.get("tabular_datasets") or []
            if not isinstance(entries, list):
                entries = []
            selected = str(self._scenario_backbone_config.get("dataset_csv") or "").strip()
            self._all_library_entries = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                label = str(entry.get("filename") or entry.get("name") or entry.get("path") or "").strip()
                path = str(entry.get("path") or "").strip()
                if label and path:
                    self._all_library_entries.append((label, path))
            self._apply_library_filter(preferred=selected or pending)
            return

        names = list(payload.get("datasets") or [])
        categories = payload.get("categories") if isinstance(payload, dict) else {}
        if not isinstance(categories, dict):
            categories = {}
        self._all_library_entries = []
        for n in names:
            value = str(n or "").strip()
            if self._scenario_backbone_type == "audio_recognition":
                if str(categories.get(value) or "") != "audio":
                    continue
            elif str(categories.get(value) or "") == "audio":
                continue
            if value:
                self._all_library_entries.append((value, value))
        self._apply_library_filter(preferred=pending or self._dataset_folder_hint)

    def _apply_library_filter(self, *, preferred: str = "") -> None:
        search = str(self._library_search_box.text() or "").strip().lower()
        filtered = list(self._all_library_entries)
        if search:
            filtered = [
                (label, value)
                for label, value in filtered
                if search in label.lower() or search in value.lower()
            ]

        self._library_combo.blockSignals(True)
        self._library_combo.clear()
        for label, value in filtered:
            self._library_combo.addItem(label, value)
        self._library_list.blockSignals(True)
        self._library_list.clear()
        for label, value in filtered:
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, value)
            item.setToolTip(value)
            self._library_list.addItem(item)

        idx = -1
        target = str(preferred or "").strip()
        if target:
            idx = self._library_combo.findData(target)
        if idx >= 0:
            self._library_combo.setCurrentIndex(idx)
        elif not search and self._library_combo.count():
            self._library_combo.setCurrentIndex(0)
        else:
            self._library_combo.setCurrentIndex(-1)
        self._library_combo.blockSignals(False)
        self._sync_library_catalog_selection(str(self._library_combo.currentData() or target or "").strip())
        self._library_list.blockSignals(False)

        if search:
            self._library_search_match_label.setText(
                f"{len(filtered)} / {len(self._all_library_entries)} match"
            )
        else:
            self._library_search_match_label.setText("")
        self._sync_header()
        self._sync_audio_asset_action_state()
        self.reload()

    def _sync_library_catalog_selection(self, value: str) -> None:
        target = str(value or "").strip()
        self._library_list.blockSignals(True)
        if not target:
            self._library_list.clearSelection()
            self._library_list.blockSignals(False)
            return
        for i in range(self._library_list.count()):
            item = self._library_list.item(i)
            if item is None:
                continue
            if str(item.data(Qt.ItemDataRole.UserRole) or "").strip() == target:
                self._library_list.setCurrentItem(item)
                self._library_list.scrollToItem(item)
                break
        self._library_list.blockSignals(False)

    def _load_audio_assets(self) -> None:
        if self._scenario_backbone_type != "audio_recognition":
            self._audio_asset_entries = []
            try:
                self._audio_assets_list.clear()
            except Exception:
                pass
            self._sync_audio_asset_action_state()
            return
        try:
            payload = self._http_get("/audio/assets")
        except Exception as exc:
            self._audio_asset_entries = []
            self._audio_assets_list.clear()
            item = QListWidgetItem(f"Audio asset list failed: {exc}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._audio_assets_list.addItem(item)
            self._sync_audio_asset_action_state()
            return
        items = payload.get("items") if isinstance(payload, dict) else []
        self._audio_asset_entries = [dict(x) for x in (items or []) if isinstance(x, dict)]
        previous = ""
        selected = self._audio_assets_list.currentItem()
        if selected is not None:
            previous = str((selected.data(Qt.ItemDataRole.UserRole) or {}).get("path") or "")
        self._audio_assets_list.blockSignals(True)
        self._audio_assets_list.clear()
        for entry in self._audio_asset_entries:
            rel = str(entry.get("relative_path") or entry.get("name") or "").strip()
            if not rel:
                continue
            label = rel
            size = _human_size(entry.get("size"))
            class_label = str(entry.get("classification_label") or "").strip()
            if class_label:
                label += f"  [{class_label}]"
            if size:
                label += f"  {size}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, entry)
            item.setToolTip(str(entry.get("path") or rel))
            self._audio_assets_list.addItem(item)
        if self._audio_assets_list.count() == 0:
            item = QListWidgetItem("No audio or video assets found under assets/ml_audio.")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._audio_assets_list.addItem(item)
        elif previous:
            for i in range(self._audio_assets_list.count()):
                item = self._audio_assets_list.item(i)
                data = item.data(Qt.ItemDataRole.UserRole) if item is not None else {}
                if isinstance(data, dict) and str(data.get("path") or "") == previous:
                    self._audio_assets_list.setCurrentRow(i)
                    break
        if self._audio_assets_list.currentRow() < 0:
            for i in range(self._audio_assets_list.count()):
                item = self._audio_assets_list.item(i)
                data = item.data(Qt.ItemDataRole.UserRole) if item is not None else {}
                if isinstance(data, dict):
                    self._audio_assets_list.setCurrentRow(i)
                    break
        self._audio_assets_list.blockSignals(False)
        self._sync_audio_asset_action_state()
        # Manually trigger waveform load for the auto-selected item because
        # setCurrentRow() above ran with signals blocked.
        self._on_audio_asset_selection_changed()

    def _current_audio_asset_entry(self) -> dict[str, Any]:
        try:
            item = self._audio_assets_list.currentItem()
        except Exception:
            item = None
        if item is None:
            return {}
        data = item.data(Qt.ItemDataRole.UserRole)
        return dict(data) if isinstance(data, dict) else {}

    def _on_audio_asset_selection_changed(self) -> None:
        entry = self._current_audio_asset_entry()
        label = str(entry.get("classification_label") or "").strip()
        if label and not str(self._audio_label_edit.text() or "").strip():
            self._audio_label_edit.setText(label)
        # Load waveform for the newly selected asset
        path = str(entry.get("path") or "").strip()
        if path:
            self._audio_waveform.load_file(path)
        else:
            self._audio_waveform.unload()
        self._sync_audio_asset_action_state()

    def _on_waveform_selection_changed(self, start_ms: int, end_ms: int) -> None:
        """Waveform region drag → update start/end spinboxes."""
        self._audio_start_ms.blockSignals(True)
        self._audio_end_ms.blockSignals(True)
        self._audio_start_ms.setValue(start_ms)
        self._audio_end_ms.setValue(end_ms)
        self._audio_start_ms.blockSignals(False)
        self._audio_end_ms.blockSignals(False)
        self._sync_audio_asset_action_state()

    def _on_waveform_selection_cleared(self) -> None:
        """Waveform selection cleared → reset spinboxes to 0 / full."""
        self._audio_start_ms.blockSignals(True)
        self._audio_end_ms.blockSignals(True)
        self._audio_start_ms.setValue(0)
        self._audio_end_ms.setValue(0)
        self._audio_start_ms.blockSignals(False)
        self._audio_end_ms.blockSignals(False)

    def _on_audio_spinbox_changed(self) -> None:
        """Start/end spinbox edited → push selection back to the waveform."""
        start_ms = max(0, int(self._audio_start_ms.value()))
        end_ms = int(self._audio_end_ms.value())
        if end_ms > start_ms:
            self._audio_waveform.set_selection(start_ms, end_ms)
        else:
            self._audio_waveform.clear_selection()

    def _copy_selected_audio_clip(self) -> None:
        """Extract the selected time region to a user-chosen WAV file."""
        entry = self._current_audio_asset_entry()
        source_path = str(entry.get("path") or "").strip()
        if not source_path:
            self._status.setText("Select an audio asset first.")
            return
        start_ms, end_ms = self._audio_range_payload()
        dest, _ = QFileDialog.getSaveFileName(
            self,
            "Save Audio Clip",
            str(Path(source_path).with_suffix(".clip.wav")),
            "WAV files (*.wav);;All files (*)",
        )
        if not dest:
            return
        body: dict[str, Any] = {
            "source_path": source_path,
            "dest_path": dest,
            "start_ms": start_ms,
        }
        if end_ms is not None:
            body["end_ms"] = end_ms
        self._status.setText(f"Extracting clip from {Path(source_path).name}...")
        try:
            payload = self._http_json_direct("POST", "/audio/copy_clip", body, timeout=120.0)
            out = str(payload.get("clip_path") or dest) if isinstance(payload, dict) else dest
            self._status.setText(f"Clip saved: {Path(out).name}")
        except Exception as exc:
            self._status.setText(f"Copy clip failed: {exc}")

    def _audio_range_payload(self) -> tuple[int, Optional[int]]:
        try:
            start_ms = int(self._audio_start_ms.value())
        except Exception:
            start_ms = 0
        try:
            end_raw = int(self._audio_end_ms.value())
        except Exception:
            end_raw = 0
        end_ms: Optional[int] = end_raw if end_raw > start_ms else None
        return max(0, start_ms), end_ms

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

    def _analyze_selected_audio_asset(self) -> None:
        entry = self._current_audio_asset_entry()
        source_path = str(entry.get("path") or "").strip()
        rel = str(entry.get("relative_path") or entry.get("name") or source_path)
        if not source_path:
            self._status.setText("Select an audio asset first.")
            return
        start_ms, end_ms = self._audio_range_payload()
        is_wav = Path(source_path).suffix.lower() == ".wav"
        extra = "" if is_wav else " (extracting audio from video — may take a few seconds)"
        body: dict[str, Any] = {"path": source_path, "start_ms": start_ms}
        if end_ms is not None:
            body["end_ms"] = end_ms
        self._status.setText(f"Analyzing: {rel}{extra}")
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        try:
            payload = self._http_json_direct("POST", "/audio/analyze", body, timeout=180.0)
        except Exception as exc:
            msg = f"Audio analysis failed for '{rel}': {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        metrics = payload.get("metrics") if isinstance(payload, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        summary = self._format_audio_metrics(metrics)
        self._status.setText(f"[audio] {rel}: {summary}")
        # For non-WAV files show the metrics in a dialog since the status bar
        # is easy to miss after the blocking analysis call.
        if not is_wav and metrics:
            QMessageBox.information(
                self,
                f"Audio Analysis — {Path(rel).name}",
                summary,
            )

    def _collect_selected_audio_asset(self) -> None:
        entry = self._current_audio_asset_entry()
        source_path = str(entry.get("path") or "").strip()
        rel = str(entry.get("relative_path") or entry.get("name") or source_path)
        slug, _enc_slug = self._library_slug_encoded()
        if not source_path:
            self._status.setText("Select an audio asset first.")
            return
        if not slug:
            self._status.setText("Select an AudioFolder dataset before adding a training clip.")
            return
        label = str(self._audio_label_edit.text() or "").strip()
        if not label:
            label, ok = QInputDialog.getText(
                self,
                "Audio Class Label",
                "Class label for this training clip:",
            )
            if not ok:
                return
            label = str(label or "").strip()
            if label:
                self._audio_label_edit.setText(label)
        if not label:
            self._status.setText("Audio training add cancelled: class label is required.")
            return
        start_ms, end_ms = self._audio_range_payload()
        split = str(self._audio_collect_split.currentData() or "train")
        body: dict[str, Any] = {
            "dataset": slug,
            "source_path": source_path,
            "label": label,
            "split": split,
            "start_ms": start_ms,
            "clean": True,
            "noise_reduce": True,
            "trim_silence": False,
            "normalize": True,
        }
        if end_ms is not None:
            body["end_ms"] = end_ms
        self._status.setText(f"Adding '{rel}' to {slug}/{split}/{label}...")
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        try:
            payload = self._http_json_direct("POST", "/audio/collect_clip", body, timeout=240.0)
        except Exception as exc:
            msg = f"Audio training add failed for '{rel}': {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        clip_path = str(payload.get("clip_path") or "") if isinstance(payload, dict) else ""
        self.reload()
        self.datasetChanged.emit(self._scenario_name)
        self._status.setText(
            f"Added audio training clip to {slug}/{split}/{label}: {Path(clip_path).name if clip_path else rel}"
        )

    def _clear_topology(self, message: str = "") -> None:
        self._topology_tree.clear()
        if message:
            item = QTreeWidgetItem([message, ""])
            self._topology_tree.addTopLevelItem(item)

    @staticmethod
    def _path_parts_for_topology(entry: dict[str, Any]) -> list[str]:
        rel = str(entry.get("relative_path") or entry.get("name") or "").strip()
        if not rel:
            return []
        return [part for part in Path(rel).parts if part and part not in (".", "/")]

    def _populate_topology(self, images: list[dict[str, Any]]) -> None:
        self._topology_tree.clear()
        if not images:
            self._clear_topology("No files in selected dataset")
            return

        counts: dict[str, int] = {}
        sample_leaf: dict[str, str] = {}
        for entry in images:
            parts = self._path_parts_for_topology(entry)
            if not parts:
                continue
            for idx in range(len(parts)):
                key = "/".join(parts[: idx + 1])
                counts[key] = counts.get(key, 0) + 1
            parent = "/".join(parts[:-1])
            if parent and parent not in sample_leaf:
                sample_leaf[parent] = parts[-1]

        if not counts:
            self._clear_topology("No files in selected dataset")
            return

        def sort_key(path: str) -> tuple[int, str]:
            return (path.count("/"), path.lower())

        nodes: dict[str, QTreeWidgetItem] = {}
        for key in sorted(counts.keys(), key=sort_key):
            name = key.rsplit("/", 1)[-1]
            count = counts.get(key, 0)
            suffix = ""
            if key in sample_leaf and "/" not in sample_leaf.get(key, ""):
                suffix = f" e.g. {sample_leaf[key]}"
            item = QTreeWidgetItem([name + suffix, str(count)])
            nodes[key] = item
            parent_key = key.rsplit("/", 1)[0] if "/" in key else ""
            if parent_key and parent_key in nodes:
                nodes[parent_key].addChild(item)
            else:
                self._topology_tree.addTopLevelItem(item)

        self._topology_tree.expandToDepth(1)

    def reload(self) -> None:
        if self._tabular_mode:
            rel = str(self._library_combo.currentData() or "").strip()
            if not rel:
                self._count.setText("")
                self._status.setText("No tabular CSV selected. Add .csv files under mlops/datasets/.")
                self._tabular_preview.setPlainText("")
                self._clear_topology("")
                return
            p = Path(rel)
            if not p.is_absolute():
                p = Path(ROOT_DIR) / p
            try:
                st = p.stat()
                size = int(st.st_size)
            except Exception:
                size = 0

            head_rows: list[list[str]] = []
            errors: list[str] = []
            try:
                with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        head_rows.append([str(c) for c in row])
                        if i >= 50:
                            break
            except Exception as exc:
                errors.append(str(exc))

            self._count.setText(f"{p.name}  |  {size / (1024.0 * 1024.0):.1f} MB")
            self._sync_header()
            if errors:
                self._status.setText(f"CSV preview failed: {errors[0]}")
                self._tabular_preview.setPlainText("")
                try:
                    self._schema_panel.set_context(
                        scenario=self._scenario_name,
                        dataset_csv=rel,
                        attempted_label_col=str(self._scenario_backbone_config.get("label_col") or ""),
                        columns=[],
                        suggested_label_cols=[],
                        current_label_col=str(self._scenario_backbone_config.get("label_col") or ""),
                        current_feature_cols=self._scenario_backbone_config.get("feature_cols")
                        if isinstance(self._scenario_backbone_config.get("feature_cols"), list)
                        else None,
                    )
                    self._schema_panel.set_buttons_enabled(False)
                except Exception:
                    pass
                return

            columns = []
            if head_rows:
                try:
                    columns = [str(c).strip() for c in head_rows[0] if str(c).strip()]
                except Exception:
                    columns = []
            current_label = str(self._scenario_backbone_config.get("label_col") or "").strip()
            current_feats = (
                self._scenario_backbone_config.get("feature_cols")
                if isinstance(self._scenario_backbone_config.get("feature_cols"), list)
                else None
            )
            try:
                self._schema_panel.set_context(
                    scenario=self._scenario_name,
                    dataset_csv=rel,
                    attempted_label_col=current_label or "label",
                    columns=columns,
                    suggested_label_cols=self._suggest_label_cols(columns),
                    current_label_col=current_label,
                    current_feature_cols=current_feats,
                )
                self._schema_panel.set_buttons_enabled(bool(self._scenario_name and columns))
                if current_label and columns and current_label not in columns:
                    self._schema_panel.set_status(
                        f"label_col '{current_label}' not found in CSV columns. Choose one and click Apply."
                    )
            except Exception:
                pass

            # Format a simple fixed-width preview.
            col_widths: list[int] = []
            for row in head_rows[:10]:
                for j, cell in enumerate(row):
                    if j >= len(col_widths):
                        col_widths.append(0)
                    col_widths[j] = max(col_widths[j], min(32, len(cell)))
            col_widths = [min(32, max(6, w)) for w in col_widths[:24]]

            def fmt_row(row: list[str]) -> str:
                out = []
                for j, w in enumerate(col_widths):
                    val = row[j] if j < len(row) else ""
                    val = val.replace("\t", " ").replace("\r", " ").replace("\n", " ")
                    if len(val) > w:
                        val = val[: max(0, w - 1)] + "…"
                    out.append(val.ljust(w))
                return " | ".join(out)

            lines = [fmt_row(r) for r in head_rows]
            if head_rows and len(head_rows[0]) > len(col_widths):
                lines.append(f"... ({len(head_rows[0])} columns total; showing first {len(col_widths)})")
            self._tabular_preview.setPlainText("\n".join(lines).strip())
            self._status.setText(f"[CSV] Previewing: {rel}")
            return

        slug, enc_slug = self._library_slug_encoded()
        if not slug:
            self._count.setText("")
            if self._scenario_backbone_type == "audio_recognition":
                self._status.setText(
                    "No AudioFolder dataset selected. Add or import a dataset under assets/ml_audio/ for training."
                )
            else:
                self._status.setText(
                    "No database folder selected. Add dataset directories under database/."
                )
            self._sync_header()
            self._all_images = []
            self._filtered_images = []
            self._dataset_classes = []
            self._page_index = 0
            self._clear_topology("Select a dataset library to view topology")
            try:
                self._populate_filters()
            except Exception:
                pass
            self._render_page()
            return
        try:
            payload = self._http_get(f"/database/{enc_slug}")
        except Exception as exc:
            msg = f"Dataset list failed for library '{slug}': {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        fmt = str(payload.get("format") or "yolo_detection")
        self._apply_dataset_format(fmt)
        raw_classes = payload.get("classes") if isinstance(payload, dict) else []
        self._dataset_classes = [str(c).strip() for c in (raw_classes or []) if str(c).strip()] if isinstance(raw_classes, list) else []
        if fmt == "audiofolder_classification":
            all_items = payload.get("audio_files") or []
        else:
            all_items = payload.get("images") or []
        self._all_images = [im for im in all_items if isinstance(im, dict)]
        self._populate_topology(self._all_images)
        split_counts = payload.get("split_counts") if isinstance(payload, dict) else {}
        if not isinstance(split_counts, dict):
            split_counts = {}
        labeled_count = sum(1 for im in self._all_images if im.get("has_label"))
        total_count = len(self._all_images)
        item_word = "audio clip(s)" if fmt == "audiofolder_classification" else "image(s)"
        parts: list[str] = [f"{total_count} {item_word}"]
        # Preserve a sensible order: train/val/test first, then anything else.
        for key in ("train", "val", "test"):
            if (split_counts or {}).get(key) is not None:
                try:
                    n = int((split_counts or {}).get(key) or 0)
                except Exception:
                    n = 0
                if n:
                    parts.append(f"{key} {n}")
        extras = []
        for k, v in (split_counts or {}).items():
            if k in ("train", "val", "test") or k == "root":
                continue
            try:
                n = int(v or 0)
            except Exception:
                n = 0
            if n:
                extras.append((k, n))
        for k, n in sorted(extras, key=lambda kv: str(kv[0]).lower()):
            parts.append(f"{k} {n}")
        try:
            root_n = int((split_counts or {}).get("root") or 0)
        except Exception:
            root_n = 0
        if root_n:
            parts.append(f"root {root_n}")
        parts.append(f"labels {labeled_count}")
        self._count.setText("  |  ".join(parts))
        # Update browse controls and render the current page.
        self._populate_filters()
        self._apply_filters_and_render()

        suffix = ""
        classes = payload.get("classes") if isinstance(payload, dict) else None
        class_note = ""
        if fmt == "imagefolder_classification" and isinstance(classes, list) and classes:
            shown = [str(c) for c in classes[:8] if str(c)]
            if shown:
                class_note = " Classes: " + ", ".join(shown) + ("..." if len(classes) > 8 else "") + "."
        if fmt == "yolo_detection":
            self._status.setText(suffix.strip())
        elif fmt == "csv_tabular":
            csv_files = payload.get("csv_files") or []
            file_names = ", ".join(str(f.get("name") or "") for f in csv_files[:6])
            extra = f" + {len(csv_files) - 6} more" if len(csv_files) > 6 else ""
            self._status.setText(f"[CSV] Tabular dataset — {len(csv_files)} file(s): {file_names}{extra}")
        elif fmt == "audiofolder_classification":
            classes_list = payload.get("classes") if isinstance(payload, dict) else []
            n_classes = len(classes_list) if isinstance(classes_list, list) else 0
            class_word = "class" if n_classes == 1 else "classes"
            shown = [str(c) for c in (classes_list or [])[:8] if str(c)] if isinstance(classes_list, list) else []
            if shown and not str(self._audio_label_edit.text() or "").strip():
                self._audio_label_edit.setText(shown[0])
            class_note = ""
            if shown:
                class_note = f" Classes: {', '.join(shown)}"
                if isinstance(classes_list, list) and len(classes_list) > 8:
                    class_note += "..."
                class_note += "."
            self._status.setText(
                f"[audio] {total_count} audio clip(s) — {n_classes} {class_word}."
                " Select a clip here for review, analysis, or training readiness."
                + class_note
            )
        elif fmt == "face_csv":
            classes_list = payload.get("classes") if isinstance(payload, dict) else []
            n_classes = len(classes_list) if isinstance(classes_list, list) else 0
            n_images = len(self._all_images)
            labeled = sum(1 for im in self._all_images if im.get("has_label"))
            id_word = "identity" if n_classes == 1 else "identities"
            self._status.setText(
                f"[face_csv] {n_images} face image(s) — {n_classes} {id_word} — {labeled} labeled."
                " Train to build the gallery.db recognition model."
            )
        elif fmt == "imagefolder_classification":
            detection_label_count = 0
            missing_detection_label_count = 0
            if isinstance(payload, dict):
                try:
                    detection_label_count = int(payload.get("detection_label_count") or 0)
                    missing_detection_label_count = int(payload.get("missing_detection_label_count") or 0)
                except Exception:
                    detection_label_count = 0
                    missing_detection_label_count = 0
            if detection_label_count:
                label_note = (
                    f" Found {detection_label_count} existing YOLO .txt label(s)"
                    f" ({missing_detection_label_count} missing). Use import-label conversion to reuse them."
                )
            else:
                label_note = " No YOLO .txt sidecar labels found; use empty conversion for annotation."
            self._status.setText(
                "ImageFolder classification dataset detected."
                + label_note
                + class_note
                + suffix
            )
        else:
            self._status.setText(f"Dataset format '{fmt}'. Viewer is best-effort." + suffix)

    def current_split(self) -> str:
        return str(self._split_combo.currentData() or "train")

    def _pick_labeled_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Add labeled images to {self.current_split()}",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if paths:
            self._upload_labeled_paths(paths)

    def _pick_empty_label_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Add empty-label images to {self.current_split()}",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if paths:
            self._upload_empty_label_paths(paths)

    def _open_upload_dialog(self) -> None:
        slug, _enc_slug = self._library_slug_encoded()
        if not slug:
            msg = "Dataset upload failed: select a database folder first."
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        dlg = _UploadDialog(split=self.current_split(), parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._upload_paths(dlg.selected_paths(), create_empty_label=dlg.create_empty_labels())

    def _upload_labeled_paths(self, paths: list[str]) -> None:
        self._upload_paths(paths, create_empty_label=False)

    def _upload_empty_label_paths(self, paths: list[str]) -> None:
        self._upload_paths(paths, create_empty_label=True)

    def _upload_paths(self, paths: list[str], *, create_empty_label: bool) -> None:
        paths = _expand_upload_paths(paths)
        slug, enc_slug = self._library_slug_encoded()
        if not slug:
            msg = "Dataset upload failed: select a database folder first."
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        if not paths:
            msg = "Dataset upload failed: no supported image files were provided."
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        split = self.current_split()
        url = self._base_url + f"/database/{enc_slug}/add"
        ok_count = 0
        errors: list[str] = []
        split_counts: dict[str, int] = {}
        empty_label_count = 0
        labeled_count = 0
        missing_label_count = 0
        for index, p in enumerate(paths, start=1):
            path = Path(p)
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            upload_split = _infer_upload_split(path, split)
            files: dict[str, Path] = {"image": path}
            if not create_empty_label:
                label_path = _matching_upload_label_path(path)
                if label_path is None:
                    missing_label_count += 1
                    errors.append(f"{path.name}: missing matching YOLO/sidecar label")
                    continue
                files["label"] = label_path
                labeled_count += 1
            else:
                empty_label_count += 1
            try:
                _multipart_upload(
                    url,
                    fields={
                        "split": upload_split,
                        "create_empty_label": "1" if create_empty_label else "0",
                    },
                    files=files,
                )
                ok_count += 1
                split_counts[upload_split] = split_counts.get(upload_split, 0) + 1
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                errors.append(f"{path.name}: HTTP {exc.code} {detail[:80]}")
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")
            if index == 1 or index % 10 == 0 or index == len(paths):
                split_note = ", ".join(f"{k} {v}" for k, v in sorted(split_counts.items()))
                self._status.setText(
                    f"Uploading folder data... {index}/{len(paths)} scanned, {ok_count} added"
                    + (f" ({split_note})" if split_note else "")
                )
                app = QApplication.instance()
                if app is not None:
                    app.processEvents()
        if errors:
            msg = f"Uploaded {ok_count}. Errors: " + " | ".join(errors[:3])
            if missing_label_count:
                msg += (
                    f" | {missing_label_count} missing label(s). "
                    "Use a YOLO labels/ mirror, sidecar .txt files, or enable empty-label mode."
                )
            self._status.setText(msg)
            self.errorRaised.emit(msg)
        else:
            mode = "empty-label" if create_empty_label else "labeled"
            split_note = ", ".join(f"{k} {v}" for k, v in sorted(split_counts.items()))
            label_note = (
                f", {empty_label_count} empty label(s)"
                if create_empty_label
                else f", {labeled_count} label file(s)"
            )
            self._status.setText(
                f"Added {ok_count} {mode} image(s)"
                + (f" ({split_note})" if split_note else f" to {split}")
                + label_note
                + "."
            )
        self.reload()
        if ok_count:
            self.datasetChanged.emit(self._scenario_name)

    def _delete_selected(self) -> None:
        rel_paths = self._list.selected_relative_paths()
        slug, enc_slug = self._library_slug_encoded()
        if not rel_paths or not slug:
            return
        removed = 0
        for rel_path in rel_paths:
            if not rel_path:
                continue
            try:
                encoded = urllib.parse.quote(rel_path, safe="")
                self._http_delete(f"/database/{enc_slug}/{encoded}")
                removed += 1
            except Exception as exc:
                msg = f"Dataset delete failed for '{rel_path}': {exc}"
                self._status.setText(msg)
                self.errorRaised.emit(msg)
                return
        if removed:
            self._status.setText(f"Deleted {removed} image(s).")
            self.reload()
            self.datasetChanged.emit(self._scenario_name)

    def _delete_library(self) -> None:
        slug, enc_slug = self._library_slug_encoded()
        if not slug:
            return
        confirm = QMessageBox.question(
            self,
            "Delete Library",
            f"Permanently delete the entire dataset library '{slug}'?\n\n"
            "This removes the folder and every image it contains. This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        force = 0
        while True:
            try:
                path = f"/database/{enc_slug}" + (f"?force={force}" if force else "")
                payload = self._http_delete(path)
                break
            except Exception as exc:
                msg = str(exc)
                if "in use" in msg.lower() and not force:
                    scen_confirm = QMessageBox.warning(
                        self,
                        "Dataset In Use",
                        f"'{slug}' is referenced by one or more scenarios.\n\n"
                        "Delete anyway? Affected scenarios will lose their dataset link.",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                        QMessageBox.StandardButton.Cancel,
                    )
                    if scen_confirm != QMessageBox.StandardButton.Yes:
                        self._status.setText("Delete cancelled.")
                        return
                    force = 1
                    continue
                fail = f"Library delete failed: {msg}"
                self._status.setText(fail)
                self.errorRaised.emit(fail)
                return
        affected = payload.get("scenarios_affected") if isinstance(payload, dict) else []
        suffix = f" (scenarios affected: {', '.join(affected)})" if affected else ""
        self._status.setText(f"Deleted library '{slug}'{suffix}.")
        self._dataset_folder_hint = ""
        self.reload_library_list()
        self.datasetChanged.emit(self._scenario_name)

    def _on_item_activated(self, item: QTableWidgetItem) -> None:
        rel_path = str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else self._list.current_relative_path()
        if not rel_path:
            return
        self._open_annotation_editor(start_relative_path=rel_path)

    def _open_label_viewer_for_item(self, item: QListWidgetItem) -> None:
        rel_path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        self._open_label_viewer_for_path(rel_path)

    def _open_label_viewer_for_path(self, rel_path: str) -> None:
        slug, enc_slug = self._library_slug_encoded()
        if not slug or not rel_path:
            return
        try:
            encoded = urllib.parse.quote(rel_path, safe="")
            payload = self._http_get(f"/database/{enc_slug}/label/{encoded}")
        except Exception as exc:
            msg = f"Could not load labels for '{rel_path}': {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        text = str(payload.get("text") or "")
        line_count = int(payload.get("line_count") or 0)
        dlg = _LabelTextDialog(image_path=rel_path, text=text, line_count=line_count, parent=self)
        dlg.exec()
