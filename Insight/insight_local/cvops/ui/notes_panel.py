"""CV Ops Notes tab — drop files, save text notes, session log, and voice recordings."""

from __future__ import annotations

import shutil
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QEvent, QMimeData, QObject, QPointF, QProcess, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent, QMouseEvent, QResizeEvent
from PyQt6.QtMultimedia import (
    QCamera,
    QCameraDevice,
    QImageCapture,
    QMediaCaptureSession,
    QMediaDevices,
)
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..paths import CVOPS_STATE_DIR
from ..notes_transcription import (
    format_transcript_markdown,
    is_transcribable_audio_note,
    transcript_filename_for_source,
    transcribe_audio_note,
    vosk_setup_hint,
)
from ...ui.theme import current_color_scheme, is_aurora_family_scheme
from .algo_catalog import reveal_in_finder
from .audio_timeline import AudioTimeline, _fmt_ms
from .cvops_theme import repolish
from .dataset_panel import AudioWaveformPlayer
from .notes_ai_workspace import NotesAiWorkspace
from .notes_spaces import (
    create_space,
    DEFAULT_SPACE_ID,
    ensure_notes_spaces_layout,
    list_space_ids,
    read_space_goals,
    read_space_pinned,
    read_space_title,
)

_NOTES_SUBDIRS = ("files", "recordings", "sessions", "captures")
_DOC_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".json",
    ".yaml",
    ".yml",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".webm",
    ".mp4",
    ".mov",
}

# Formats we offer waveform + QMediaPlayer preview for (subset of _DOC_SUFFIXES audio).
_AUDIO_PREVIEW_SUFFIXES = frozenset(
    {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}
)


def _path_audio_previewable(path: Path) -> bool:
    return path.suffix.lower() in _AUDIO_PREVIEW_SUFFIXES


_MD_DROP_SUFFIXES = frozenset({".md", ".markdown"})
_TEXT_OPEN_SUFFIXES = frozenset({".txt", ".md", ".markdown"})


def _paths_from_mime_for_notes_ingest(mime: Optional[QMimeData]) -> list[str]:
    """Collect local file paths from a drag/drop mime payload (same rules as the notes vault ingest)."""
    if mime is None or not mime.hasUrls():
        return []
    paths: list[str] = []
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        p = Path(url.toLocalFile())
        if p.is_file():
            paths.append(str(p))
        elif p.is_dir():
            try:
                for child in sorted(p.rglob("*")):
                    if child.is_file() and child.suffix.lower() in _DOC_SUFFIXES:
                        paths.append(str(child))
            except Exception:
                continue
    return paths


def _waveform_selection_range_ms(wave: AudioWaveformPlayer) -> tuple[int, Optional[int]]:
    """Clip bounds for /audio/analyze — same semantics as dataset spinboxes when a region is picked."""
    try:
        s = wave.sel_start_ms
        e = wave.sel_end_ms
    except Exception:
        return 0, None
    if s is None or e is None:
        return 0, None
    si, ei = int(s), int(e)
    if ei <= si:
        return 0, None
    return max(0, si), ei


def _format_notes_audio_metrics(metrics: dict[str, Any]) -> str:
    """Same summary line as DatasetPanel._format_audio_metrics (audio recognition pipeline)."""

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


def _notes_kind_label(kind: str) -> str:
    return {
        "file": "Upload",
        "recording": "Voice memo",
        "session": "Session log",
        "capture": "Capture",
    }.get(str(kind or "").lower(), str(kind or "Asset").title())


def _notes_suffix_label(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix.upper() if suffix else "FILE"


def _format_notes_file_size(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return "0 B"


def _format_notes_timestamp(ts: float) -> str:
    try:
        dt = datetime.fromtimestamp(float(ts)).astimezone()
    except Exception:
        return "Unknown time"
    now = datetime.now(dt.tzinfo)
    if dt.date() == now.date():
        return dt.strftime("Today %I:%M %p").replace(" 0", " ")
    if dt.year == now.year:
        return dt.strftime("%b %d %I:%M %p").replace(" 0", " ")
    return dt.strftime("%b %d, %Y %I:%M %p").replace(" 0", " ")


def _notes_asset_meta(path: Path, kind: str) -> str:
    try:
        stat = path.stat()
        size_label = _format_notes_file_size(int(stat.st_size))
        stamp = _format_notes_timestamp(float(stat.st_mtime))
    except Exception:
        size_label = "Unknown size"
        stamp = "Unknown time"
    return f"{_notes_kind_label(kind)}  •  {_notes_suffix_label(path)}  •  {size_label}  •  {stamp}"


def _notes_asset_size_and_stamp(path: Path) -> tuple[str, str]:
    try:
        stat = path.stat()
        return _format_notes_file_size(int(stat.st_size)), _format_notes_timestamp(float(stat.st_mtime))
    except Exception:
        return "Unknown size", "Unknown time"


def _read_notes_preview_text(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            chunk = handle.read(2048)
    except Exception:
        return ""
    for raw in chunk.splitlines():
        line = " ".join(raw.strip().split())
        if line:
            return line
    return ""


def _notes_asset_preview(path: Path, kind: str) -> str:
    suffix = path.suffix.lower()
    if kind == "session":
        return "Project session artifact stored in the notes vault."
    if suffix in (_TEXT_OPEN_SUFFIXES | {".json", ".yaml", ".yml", ".csv"}):
        snippet = _read_notes_preview_text(path)
        if snippet:
            return snippet
        return "Click to open this note in the editor pane."
    if suffix == ".pdf":
        return "PDF document stored with the current project materials."
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return "Visual reference or saved capture for this project."
    if suffix in {".mp4", ".mov", ".webm"}:
        return "Video capture stored in the project notes vault."
    return "Double-click to reveal this file in Finder."


class _NotesRawEditor(QTextEdit):
    """Source editor — accepts drops of notes-ingestible files (panel handles routing)."""

    filesDropped = pyqtSignal(list)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if _paths_from_mime_for_notes_ingest(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if _paths_from_mime_for_notes_ingest(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        paths = _paths_from_mime_for_notes_ingest(event.mimeData())
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class _NotesElidedLabel(QLabel):
    """QLabel with stored source text and automatic resize-time elision."""

    def __init__(
        self,
        full_text: str = "",
        *,
        elide_mode: Qt.TextElideMode = Qt.TextElideMode.ElideRight,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__("", parent)
        self._full_text = str(full_text or "")
        self._elide_mode = elide_mode
        self.setWordWrap(False)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._sync_text()

    def set_full_text(self, text: str) -> None:
        self._full_text = str(text or "")
        self._sync_text()

    def full_text(self) -> str:
        return self._full_text

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_text()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._sync_text()

    def _sync_text(self) -> None:
        width = max(24, self.contentsRect().width() or self.width() or 24)
        self.setText(self.fontMetrics().elidedText(self._full_text, self._elide_mode, width))


class _AudioTranscriptionWorker(QThread):
    """Background ASR worker for notes audio, backed by the archive ASR helper."""

    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, path_str: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._path_str = path_str

    def run(self) -> None:  # type: ignore[override]
        try:
            payload = transcribe_audio_note(self._path_str)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(payload)


class _NotesInlineAudioRow(QFrame):
    """Notes list row card for inline-audio assets."""

    def __init__(self, path_str: str, kind: str, filename: str, panel: NotesPanel) -> None:
        super().__init__(panel)
        self._path_str = path_str
        self._panel = panel
        self.setObjectName("notesAssetCard")
        self.setProperty("role", "audio")
        path = Path(path_str)
        self._path = path
        self._kind = kind

        self._wave = AudioWaveformPlayer(self, show_transport=False)
        if is_aurora_family_scheme(current_color_scheme()):
            self._wave.timeline_widget.set_aurora_waveform_cyan(True)
        self._wave.load_file(path_str)
        self._wave.duration_changed.connect(self._refresh_meta)
        self._wave.playback_state_changed.connect(self._on_playback_state_changed)

        rail_btn_w = 70
        self._play = QPushButton("Play")
        self._play.setFixedWidth(rail_btn_w)
        self._play.setToolTip("Toggle playback")
        self._play.clicked.connect(self._wave.toggle_playback)

        self._analyze = QPushButton("Analyze")
        self._analyze.setFixedWidth(rail_btn_w)
        self._analyze.setToolTip(
            "POST /audio/analyze — analyze this clip with the same service used by Database audio recognition."
        )
        self._analyze.setEnabled(panel.has_audio_analyze_http())
        self._analyze.clicked.connect(self._on_analyze_clicked)

        self._transcribe = QPushButton("Transcribe")
        self._transcribe.setFixedWidth(86)
        self._transcribe.setToolTip("Create a Markdown transcript note from this audio using local Whisper when available.")
        self._transcribe.clicked.connect(self._on_transcribe_clicked)

        self._mute = QPushButton("Mute")
        self._mute.setCheckable(True)
        self._mute.setFixedWidth(rail_btn_w)
        self._mute.clicked.connect(self._on_mute_clicked)

        self._delete = QPushButton("Delete")
        self._delete.setFixedWidth(rail_btn_w)
        self._delete.setToolTip("Permanently delete this audio file from the notes vault.")
        self._delete.clicked.connect(self._on_delete_clicked)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)
        root.addWidget(self._wave.timeline_widget)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        badge = QLabel(_notes_kind_label(kind))
        badge.setObjectName("notesAssetKindBadge")
        self._title = _NotesElidedLabel(filename, elide_mode=Qt.TextElideMode.ElideMiddle)
        self._title.setObjectName("notesAssetTitle")
        self._title.setToolTip(path_str)
        title_row.addWidget(badge, 0)
        title_row.addWidget(self._title, 1)
        root.addLayout(title_row)

        self._meta = _NotesElidedLabel("")
        self._meta.setObjectName("notesAssetMeta")
        self._meta.setToolTip(path_str)
        self._refresh_meta()
        root.addWidget(self._meta)

        primary_actions = QHBoxLayout()
        primary_actions.setContentsMargins(0, 0, 0, 0)
        primary_actions.setSpacing(6)
        primary_actions.addWidget(self._play)
        primary_actions.addWidget(self._analyze)
        primary_actions.addWidget(self._transcribe)
        primary_actions.addStretch(1)
        root.addLayout(primary_actions)

        secondary_actions = QHBoxLayout()
        secondary_actions.setContentsMargins(0, 0, 0, 0)
        secondary_actions.setSpacing(6)
        secondary_actions.addWidget(self._mute)
        secondary_actions.addWidget(self._delete)
        secondary_actions.addStretch(1)
        root.addLayout(secondary_actions)

    def path_str(self) -> str:
        return self._path_str

    def timeline(self) -> AudioTimeline:
        return self._wave.timeline_widget

    def unload_wave(self) -> None:
        self._wave.unload()

    def pause_playback(self) -> None:
        self._wave.pause_playback()

    def _refresh_meta(self, *_args) -> None:
        size_label, stamp = _notes_asset_size_and_stamp(self._path)
        parts = [_notes_kind_label(self._kind)]
        duration_ms = self._wave.duration_ms()
        if duration_ms > 0:
            parts.append(_fmt_ms(duration_ms))
        parts.extend((_notes_suffix_label(self._path), size_label, stamp))
        self._meta.set_full_text("  •  ".join(parts))

    def _on_analyze_clicked(self) -> None:
        self._panel.run_audio_recognition_analyze(self._path_str, self._wave)

    def _on_transcribe_clicked(self) -> None:
        self._panel.transcribe_audio_note(self._path_str)

    def _on_playback_state_changed(self, *_args) -> None:
        self._play.setText("Pause" if self._wave.is_playing() else "Play")

    def _on_mute_clicked(self) -> None:
        self._wave.audio_output.setMuted(bool(self._mute.isChecked()))
        self._mute.setText("Unmute" if self._mute.isChecked() else "Mute")

    def _on_delete_clicked(self) -> None:
        self._panel._request_delete_notes_asset(self._path_str)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        ch = self.childAt(event.position().toPoint())
        if ch is not None:
            w: Optional[QWidget] = ch
            while w is not None:
                if isinstance(w, QPushButton):
                    super().mouseDoubleClickEvent(event)
                    return
                w = w.parentWidget()
        reveal_in_finder(self._path_str)
        event.accept()


class _NotesFileRow(QFrame):
    """Notes list row card for document and non-inline preview assets."""

    def __init__(self, path_str: str, kind: str, filename: str, panel: NotesPanel) -> None:
        super().__init__(panel)
        self._path_str = path_str
        self._panel = panel
        self.setObjectName("notesAssetCard")
        self.setProperty("role", "file")
        path = Path(path_str)
        suffix = path.suffix.lower()
        can_open_editor = suffix in _TEXT_OPEN_SUFFIXES

        self._open = QPushButton("Open" if can_open_editor else "Reveal")
        self._open.setFixedWidth(72)
        self._open.clicked.connect(self._on_primary_action)
        self._transcribe: Optional[QPushButton] = None
        if is_transcribable_audio_note(path):
            self._transcribe = QPushButton("Transcribe")
            self._transcribe.setFixedWidth(86)
            self._transcribe.setToolTip("Create a Markdown transcript note from this audio using local Whisper when available.")
            self._transcribe.clicked.connect(lambda: panel.transcribe_audio_note(path_str))
        self._delete = QPushButton("Delete")
        self._delete.setFixedWidth(72)
        self._delete.setToolTip("Permanently delete this file from the notes vault.")
        self._delete.clicked.connect(lambda: panel._request_delete_notes_asset(path_str))

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)
        info = QVBoxLayout()
        info.setContentsMargins(0, 0, 0, 0)
        info.setSpacing(3)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        badge = QLabel(_notes_kind_label(kind))
        badge.setObjectName("notesAssetKindBadge")
        self._title = _NotesElidedLabel(filename, elide_mode=Qt.TextElideMode.ElideMiddle)
        self._title.setObjectName("notesAssetTitle")
        self._title.setToolTip(path_str)
        title_row.addWidget(badge, 0)
        title_row.addWidget(self._title, 1)
        info.addLayout(title_row)

        self._meta = _NotesElidedLabel(_notes_asset_meta(path, kind))
        self._meta.setObjectName("notesAssetMeta")
        self._meta.setToolTip(path_str)
        info.addWidget(self._meta)
        header.addLayout(info, 1)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        actions.addWidget(self._open)
        if self._transcribe is not None:
            actions.addWidget(self._transcribe)
        actions.addWidget(self._delete)
        header.addLayout(actions, 0)
        root.addLayout(header)

        self._preview = _NotesElidedLabel(_notes_asset_preview(path, kind))
        self._preview.setObjectName("notesAssetPreview")
        self._preview.setToolTip(path_str)
        root.addWidget(self._preview)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        ch = self.childAt(event.position().toPoint())
        w: Optional[QWidget] = ch
        while w is not None:
            if isinstance(w, QPushButton):
                super().mousePressEvent(event)
                return
            w = w.parentWidget()
        if Path(self._path_str).suffix.lower() in _TEXT_OPEN_SUFFIXES:
            self._panel._open_path_in_notes_editor(self._path_str)
        super().mousePressEvent(event)

    def _on_primary_action(self) -> None:
        if Path(self._path_str).suffix.lower() in _TEXT_OPEN_SUFFIXES:
            self._panel._open_path_in_notes_editor(self._path_str)
            return
        reveal_in_finder(self._path_str)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        ch = self.childAt(event.position().toPoint())
        if ch is not None:
            w: Optional[QWidget] = ch
            while w is not None:
                if isinstance(w, QPushButton):
                    super().mouseDoubleClickEvent(event)
                    return
                w = w.parentWidget()
        reveal_in_finder(self._path_str)
        event.accept()


class _NewNotesProjectDialog(QDialog):
    """Two-field create flow: title (required) and goals (optional)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New notes project")
        outer = QVBoxLayout(self)
        outer.setSpacing(12)
        form = QFormLayout()
        self._field_title = QLineEdit()
        self._field_title.setPlaceholderText("Project title")
        form.addRow("Title", self._field_title)
        self._field_goals = QPlainTextEdit()
        self._field_goals.setPlaceholderText("Project goals")
        self._field_goals.setMinimumHeight(120)
        self._field_goals.setTabChangesFocus(True)
        form.addRow("Goals", self._field_goals)
        outer.addLayout(form)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._field_title.setFocus()

    def values(self) -> tuple[str, str]:
        return self._field_title.text().strip(), self._field_goals.toPlainText().strip()


class NotesPanel(QWidget):
    """Local workspace under CV Ops state: files, recordings, session logs."""

    errorRaised = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        http_json: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("notesPanel")
        self._http_json = http_json
        self._vault_container = (Path(CVOPS_STATE_DIR) / "notes").resolve()
        ensure_notes_spaces_layout(self._vault_container)
        self._spaces_root = (self._vault_container / "spaces").resolve()
        space_ids = list_space_ids(self._spaces_root)
        start_id = space_ids[0] if space_ids else DEFAULT_SPACE_ID
        self._notes_root = (self._spaces_root / start_id).resolve()
        self._applied_space_id = start_id
        self._session_path: Optional[Path] = None
        self._record_process: Optional[QProcess] = None
        self._record_token: int = 0
        self._recording_pcm = bytearray()
        self._recording = False
        self._transcription_worker: Optional[_AudioTranscriptionWorker] = None
        self._transcription_source_path: str = ""
        self._camera: Optional[QCamera] = None
        self._capture_session: Optional[QMediaCaptureSession] = None
        self._image_capture: Optional[QImageCapture] = None
        self._webcam_devices: list[QCameraDevice] = []
        self._webcam_seen_active: bool = False

        for sub in (*_NOTES_SUBDIRS, "rag_index"):
            (self._notes_root / sub).mkdir(parents=True, exist_ok=True)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("Notes")
        title.setProperty("isTitle", True)
        repolish(title)
        root.addWidget(title)

        self._main_tabs = QTabWidget()
        library = QWidget()
        lib_layout = QVBoxLayout(library)
        lib_layout.setContentsMargins(0, 0, 0, 0)
        lib_layout.setSpacing(10)

        toolbar_shell = QFrame()
        toolbar_shell.setObjectName("notesLibraryToolbar")
        toolbar_layout = QVBoxLayout(toolbar_shell)
        toolbar_layout.setContentsMargins(12, 10, 12, 10)
        toolbar_layout.setSpacing(8)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(8)
        self._library_project_label = QLabel("")
        self._library_project_label.setObjectName("notesLibraryProject")
        repolish(self._library_project_label)
        bar.addWidget(self._library_project_label)
        bar.addStretch(1)

        self._btn_open_folder = QPushButton("Open project folder")
        self._btn_open_folder.setToolTip("Reveal the active project folder in the file manager.")
        self._btn_open_folder.clicked.connect(self._on_open_folder)
        bar.addWidget(self._btn_open_folder)

        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._refresh_list)
        bar.addWidget(self._btn_refresh)
        toolbar_layout.addLayout(bar)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)

        create_group = QFrame()
        create_group.setObjectName("notesToolbarGroup")
        create_layout = QVBoxLayout(create_group)
        create_layout.setContentsMargins(10, 8, 10, 8)
        create_layout.setSpacing(6)
        create_label = QLabel("Create")
        create_label.setObjectName("notesToolbarGroupLabel")
        create_layout.addWidget(create_label)
        create_buttons = QHBoxLayout()
        create_buttons.setContentsMargins(0, 0, 0, 0)
        create_buttons.setSpacing(6)
        self._btn_add_files = QPushButton("Add files…")
        self._btn_add_files.clicked.connect(self._on_add_files)
        create_buttons.addWidget(self._btn_add_files)

        self._btn_session = QPushButton("Start session")
        self._btn_session.setCheckable(False)
        self._btn_session.clicked.connect(self._on_toggle_session)
        create_buttons.addWidget(self._btn_session)

        self._btn_save_note = QPushButton("Save text note")
        self._btn_save_note.setProperty("isPrimary", True)
        repolish(self._btn_save_note)
        self._btn_save_note.clicked.connect(self._on_save_text_note)
        create_buttons.addWidget(self._btn_save_note)
        create_layout.addLayout(create_buttons)
        action_row.addWidget(create_group, 1)

        capture_group = QFrame()
        capture_group.setObjectName("notesToolbarGroup")
        capture_layout = QVBoxLayout(capture_group)
        capture_layout.setContentsMargins(10, 8, 10, 8)
        capture_layout.setSpacing(6)
        capture_label = QLabel("Capture")
        capture_label.setObjectName("notesToolbarGroupLabel")
        capture_layout.addWidget(capture_label)
        capture_buttons = QHBoxLayout()
        capture_buttons.setContentsMargins(0, 0, 0, 0)
        capture_buttons.setSpacing(6)

        self._btn_record = QPushButton("Record voice")
        self._btn_record.clicked.connect(self._on_toggle_record)
        self._btn_record.setEnabled(self._microphone_available())
        capture_buttons.addWidget(self._btn_record)

        self._btn_webcam_toggle = QPushButton("Start webcam")
        self._btn_webcam_toggle.setObjectName("notesWebcamToggle")
        self._btn_webcam_toggle.setProperty("webcamActive", False)
        repolish(self._btn_webcam_toggle)
        self._btn_webcam_toggle.setToolTip("Preview the selected camera; save stills to notes/captures/.")
        self._btn_webcam_toggle.clicked.connect(self._on_toggle_webcam)
        capture_buttons.addWidget(self._btn_webcam_toggle)

        self._btn_webcam_shot = QPushButton("Save photo")
        self._btn_webcam_shot.setEnabled(False)
        self._btn_webcam_shot.setToolTip("Save a JPEG from the active webcam preview.")
        self._btn_webcam_shot.clicked.connect(self._on_webcam_save_photo)
        capture_buttons.addWidget(self._btn_webcam_shot)
        capture_layout.addLayout(capture_buttons)

        cam_row = QHBoxLayout()
        cam_row.setContentsMargins(0, 0, 0, 0)
        cam_row.setSpacing(6)
        cam_label = QLabel("Camera")
        cam_label.setObjectName("notesToolbarGroupLabel")
        cam_row.addWidget(cam_label)
        self._webcam_combo = QComboBox()
        self._webcam_combo.setMinimumWidth(180)
        self._webcam_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._populate_webcam_devices()
        cam_row.addWidget(self._webcam_combo, stretch=1)
        capture_layout.addLayout(cam_row)
        action_row.addWidget(capture_group, 1)
        toolbar_layout.addLayout(action_row)
        lib_layout.addWidget(toolbar_shell)

        # Video preview lives above the splitter so QVideoWidget is never resized
        # during splitter drags (which causes macOS compositor stalls).
        self._video_widget = QVideoWidget()
        self._video_widget.setMinimumHeight(180)
        self._video_widget.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self._video_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._video_widget.setVisible(False)
        lib_layout.addWidget(self._video_widget)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        self._list = QListWidget()
        self._list.setObjectName("notesAssetList")
        self._list.setAlternatingRowColors(False)
        self._list.setSpacing(10)
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.itemClicked.connect(self._on_list_item_clicked)
        self._list.itemDoubleClicked.connect(self._on_item_activated)
        ll.addWidget(self._list, stretch=1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        mode_row.addWidget(QLabel("Notes editor"))
        mode_row.addStretch(1)
        self._btn_view_md = QPushButton("Markdown")
        self._btn_view_md.setCheckable(True)
        self._btn_view_md.setToolTip("Render the current source as Markdown (read-only preview).")
        self._btn_view_raw = QPushButton("Raw")
        self._btn_view_raw.setCheckable(True)
        self._btn_view_raw.setToolTip("Edit Markdown / plain text source.")
        self._note_view_group = QButtonGroup(self)
        self._note_view_group.setExclusive(True)
        self._note_view_group.addButton(self._btn_view_md, 0)
        self._note_view_group.addButton(self._btn_view_raw, 1)
        self._btn_view_raw.setChecked(True)
        self._note_view_group.idClicked.connect(self._on_note_view_mode_changed)
        mode_row.addWidget(self._btn_view_md)
        mode_row.addWidget(self._btn_view_raw)
        rl.addLayout(mode_row)

        self._editor_stack = QStackedWidget()
        self._editor_stack.setMinimumHeight(200)
        self._editor_raw = _NotesRawEditor(self)
        self._editor_raw.setPlaceholderText(
            "Write a note, or drop a .md file here to copy it into notes and preview…"
        )
        self._editor_raw.filesDropped.connect(self._on_raw_editor_files_dropped)
        self._editor_raw.textChanged.connect(self._on_raw_note_text_changed)
        self._editor_preview = QTextBrowser()
        self._editor_preview.setReadOnly(True)
        self._editor_preview.setOpenExternalLinks(False)
        self._editor_stack.addWidget(self._editor_raw)
        self._editor_stack.addWidget(self._editor_preview)
        self._editor_stack.setCurrentIndex(0)
        rl.addWidget(self._editor_stack, stretch=1)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 420])
        lib_layout.addWidget(split, stretch=1)

        self._main_tabs.addTab(library, "Library")
        self._ai_workspace = NotesAiWorkspace(self)
        self._ai_workspace.errorRaised.connect(self.errorRaised.emit)
        self._ai_workspace.projectSelected.connect(self._on_ai_workspace_project_selected)
        self._ai_workspace.newProjectClicked.connect(self._on_new_space)
        self._ai_workspace.projectsMetadataChanged.connect(self._sync_workspace_projects)
        self._main_tabs.addTab(self._ai_workspace, "AI workspace")
        root.addWidget(self._main_tabs, stretch=1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setProperty("muted", True)
        repolish(self._status)
        root.addWidget(self._status)

        self._notes_drop_filter_objs: list[QObject] = []
        self._sync_workspace_projects()
        self._update_library_project_banner()
        self._ai_workspace.set_space_root(self._notes_root)
        self._refresh_list()  # installs tab-wide file drop filters after building the list
        self._update_session_button_label()

        try:
            QMediaDevices.videoInputsChanged.connect(self._on_webcam_devices_changed)
        except Exception:
            pass

    def _sync_workspace_projects(self) -> None:
        rows = [
            (
                sid,
                read_space_title(self._spaces_root / sid, sid),
                read_space_goals(self._spaces_root / sid),
                read_space_pinned(self._spaces_root / sid),
            )
            for sid in list_space_ids(self._spaces_root)
        ]
        self._ai_workspace.sync_project_list(rows, self._applied_space_id)
        self._applied_space_id = str(self._notes_root.name)

    def take_ai_workspace_for_assistant(self) -> NotesAiWorkspace:
        idx = self._main_tabs.indexOf(self._ai_workspace)
        self._ai_workspace_restore_index = 1 if idx < 0 else idx
        self._ai_workspace_restore_current = self._main_tabs.currentWidget() is self._ai_workspace
        if idx >= 0:
            self._main_tabs.removeTab(idx)
        self._ai_workspace.setParent(None)
        return self._ai_workspace

    def restore_ai_workspace_from_assistant(self) -> None:
        self._ai_workspace.set_compact_overlay_mode(False)
        if self._main_tabs.indexOf(self._ai_workspace) >= 0:
            return
        idx = int(getattr(self, "_ai_workspace_restore_index", 1))
        idx = max(0, min(idx, self._main_tabs.count()))
        self._main_tabs.insertTab(idx, self._ai_workspace, "AI workspace")
        if bool(getattr(self, "_ai_workspace_restore_current", False)):
            self._main_tabs.setCurrentWidget(self._ai_workspace)

    def _update_library_project_banner(self) -> None:
        label = read_space_title(
            self._spaces_root / self._applied_space_id,
            self._applied_space_id,
        )
        self._library_project_label.setText(f"Project: {label}")

    def _on_ai_workspace_project_selected(self, sid: str) -> None:
        if sid == self._applied_space_id:
            return
        if self._ai_workspace.is_ai_busy():
            QMessageBox.information(
                self,
                "Notes",
                "Wait for the AI chat or RAG task to finish before switching projects.",
            )
            self._ai_workspace.set_project_list_selection(self._applied_space_id)
            return
        if self._session_path is not None:
            QMessageBox.information(
                self,
                "Notes",
                "End the session log before switching projects.",
            )
            self._ai_workspace.set_project_list_selection(self._applied_space_id)
            return
        if self._recording:
            self._set_status("Stop voice recording before switching projects.")
            self._ai_workspace.set_project_list_selection(self._applied_space_id)
            return
        if self._audio_transcription_busy():
            self._set_status("Wait for audio transcription to finish before switching projects.")
            self._ai_workspace.set_project_list_selection(self._applied_space_id)
            return
        self._notes_root = (self._spaces_root / sid).resolve()
        for sub in (*_NOTES_SUBDIRS, "rag_index"):
            (self._notes_root / sub).mkdir(parents=True, exist_ok=True)
        self._applied_space_id = sid
        self._ai_workspace.set_space_root(self._notes_root)
        self._sync_workspace_projects()
        self._update_library_project_banner()
        self._refresh_list()

    def _on_new_space(self) -> None:
        if self._ai_workspace.is_ai_busy():
            QMessageBox.information(
                self,
                "Notes",
                "Wait for the AI chat or RAG task to finish before creating a project.",
            )
            return
        if self._session_path is not None:
            QMessageBox.information(
                self,
                "Notes",
                "End the session log before creating a new project.",
            )
            return
        if self._recording:
            QMessageBox.information(
                self,
                "Notes",
                "Stop voice recording before creating a new project.",
            )
            return
        if self._audio_transcription_busy():
            QMessageBox.information(
                self,
                "Notes",
                "Wait for audio transcription to finish before creating a new project.",
            )
            return
        dlg = _NewNotesProjectDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        title, goals = dlg.values()
        if not title:
            QMessageBox.warning(self, "Notes", "Enter a project title.")
            return
        try:
            sid = create_space(self._spaces_root, title, goals)
        except Exception as exc:
            self.errorRaised.emit(f"Could not create notes project: {exc}")
            return
        self._notes_root = (self._spaces_root / sid).resolve()
        for sub in (*_NOTES_SUBDIRS, "rag_index"):
            (self._notes_root / sub).mkdir(parents=True, exist_ok=True)
        self._applied_space_id = sid
        self._sync_workspace_projects()
        self._update_library_project_banner()
        self._ai_workspace.set_space_root(self._notes_root)
        self._ai_workspace.focus_chats_sidebar_mode()
        self._refresh_list()
        self._set_status(f"Switched to new project: {sid}")

    def _on_webcam_devices_changed(self) -> None:
        if self._camera is not None:
            return
        self._populate_webcam_devices()

    def hideEvent(self, event) -> None:  # type: ignore[override]
        if self._recording:
            self._stop_recording()
        self._stop_webcam()
        self._unload_all_note_audio_rows()
        super().hideEvent(event)

    def _unload_all_note_audio_rows(self) -> None:
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it is None:
                continue
            w = self._list.itemWidget(it)
            if isinstance(w, _NotesInlineAudioRow):
                w.unload_wave()

    def _pause_all_note_audio_players(self) -> None:
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it is None:
                continue
            w = self._list.itemWidget(it)
            if isinstance(w, _NotesInlineAudioRow):
                w.pause_playback()

    def _microphone_available(self) -> bool:
        try:
            dev = QMediaDevices.defaultAudioInput()
            return bool(dev and not dev.isNull())
        except Exception:
            return False

    def _populate_webcam_devices(self) -> None:
        self._webcam_combo.blockSignals(True)
        self._webcam_combo.clear()
        self._webcam_devices = list(QMediaDevices.videoInputs())
        if not self._webcam_devices:
            self._webcam_combo.addItem("No webcam found", userData=None)
            self._webcam_combo.setEnabled(False)
        else:
            self._webcam_combo.setEnabled(True)
            for dev in self._webcam_devices:
                self._webcam_combo.addItem(dev.description(), userData=None)
        self._webcam_combo.blockSignals(False)

    def _set_webcam_toggle_previewing(self, previewing: bool) -> None:
        self._btn_webcam_toggle.setText("Stop webcam" if previewing else "Start webcam")
        self._btn_webcam_toggle.setProperty("webcamActive", bool(previewing))
        repolish(self._btn_webcam_toggle)

    def _sync_webcam_shot_enabled(self) -> None:
        ready = False
        if self._camera is not None and self._image_capture is not None:
            try:
                ready = bool(
                    self._camera.isActive()
                    and self._image_capture.isReadyForCapture()
                )
            except Exception:
                ready = False
        self._btn_webcam_shot.setEnabled(ready)

    def _on_toggle_webcam(self) -> None:
        if self._camera is not None or self._capture_session is not None:
            self._stop_webcam()
            return
        if self._recording:
            self._set_status("Stop voice recording before starting the webcam.")
            return
        self._populate_webcam_devices()
        if not self._webcam_devices:
            self._set_status("No webcam detected.")
            return
        idx = max(0, min(self._webcam_combo.currentIndex(), len(self._webcam_devices) - 1))
        device = self._webcam_devices[idx]
        self._webcam_seen_active = False
        self._video_widget.setVisible(True)
        self._video_widget.show()
        self._video_widget.updateGeometry()
        if sys.platform == "darwin":
            self._video_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        try:
            _ = int(self._video_widget.winId())
        except Exception:
            pass
        try:
            cam = QCamera(device, self)
            session = QMediaCaptureSession(self)
            session.setCamera(cam)
            session.setVideoOutput(self._video_widget)
            img = QImageCapture(self)
            img.setQuality(QImageCapture.Quality.HighQuality)
            img.setFileFormat(QImageCapture.FileFormat.JPEG)
            img.errorOccurred.connect(self._on_webcam_image_error)
            img.imageSaved.connect(self._on_webcam_image_saved)
            img.readyForCaptureChanged.connect(lambda _v: self._sync_webcam_shot_enabled())
            session.setImageCapture(img)
            cam.errorOccurred.connect(self._on_webcam_camera_error)
            cam.activeChanged.connect(self._on_webcam_active_changed)
            self._camera = cam
            self._capture_session = session
            self._image_capture = img
            QTimer.singleShot(0, cam.start)
        except Exception as exc:
            self._set_status(f"Webcam error: {exc}")
            self._stop_webcam()
            return
        self._set_webcam_toggle_previewing(True)
        self._webcam_combo.setEnabled(False)
        self._append_session("webcam preview started")
        self._set_status("Webcam preview starting…")
        self._sync_webcam_shot_enabled()

    def _stop_webcam(self) -> None:
        self._webcam_seen_active = False
        cam = self._camera
        session = self._capture_session
        img = self._image_capture
        if cam is not None:
            try:
                cam.activeChanged.disconnect(self._on_webcam_active_changed)
            except TypeError:
                pass
            try:
                cam.errorOccurred.disconnect(self._on_webcam_camera_error)
            except TypeError:
                pass
        if img is not None:
            try:
                img.errorOccurred.disconnect(self._on_webcam_image_error)
            except TypeError:
                pass
            try:
                img.imageSaved.disconnect(self._on_webcam_image_saved)
            except TypeError:
                pass
        self._camera = None
        self._capture_session = None
        self._image_capture = None
        if session is not None:
            try:
                session.setImageCapture(None)
                session.setVideoOutput(None)
                session.setCamera(None)
            except Exception:
                pass
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass
            cam.deleteLater()
        if img is not None:
            img.deleteLater()
        if session is not None:
            session.deleteLater()
        self._video_widget.setVisible(False)
        if sys.platform == "darwin":
            self._video_widget.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, False)
        self._set_webcam_toggle_previewing(False)
        self._btn_webcam_shot.setEnabled(False)
        if self._webcam_devices:
            self._webcam_combo.setEnabled(True)

    def _on_webcam_active_changed(self, active: bool) -> None:
        if active:
            self._webcam_seen_active = True
            self._set_status("Webcam preview on — use Save photo to write a JPEG under notes/captures/.")
            return
        if not self._webcam_seen_active:
            return
        if self._camera is not None:
            self._set_status("Webcam stopped unexpectedly.")
            self._stop_webcam()

    def _on_webcam_camera_error(self, error: QCamera.Error, _msg: str) -> None:
        if error != QCamera.Error.NoError:
            self.errorRaised.emit(f"Webcam: {_msg}")
            self._stop_webcam()

    def _on_webcam_image_error(self, _id: int, error: QImageCapture.Error, msg: str) -> None:
        if error != QImageCapture.Error.NoError:
            self.errorRaised.emit(f"Webcam capture: {msg}")
            self._sync_webcam_shot_enabled()

    def _on_webcam_image_saved(self, _id: int, file_path: str) -> None:
        self._append_session(f"saved webcam capture {Path(file_path).name}")
        self._refresh_list()
        self._set_status(f"Saved {file_path}")
        self._sync_webcam_shot_enabled()

    def _on_webcam_save_photo(self) -> None:
        if self._image_capture is None or self._camera is None or not self._camera.isActive():
            self._set_status("Start the webcam preview before saving a photo.")
            return
        if not self._image_capture.isReadyForCapture():
            self._set_status("Camera is not ready to capture yet — try again in a moment.")
            return
        cap_dir = self._notes_root / "captures"
        cap_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = cap_dir / f"webcam_{ts}.jpg"
        n = 0
        while path.exists():
            n += 1
            path = cap_dir / f"webcam_{ts}_{n}.jpg"
        self._image_capture.captureToFile(str(path))

    def _on_add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add to notes",
            "",
            "Documents and media (*.txt *.md *.pdf *.json *.yaml *.yml *.csv *.png *.jpg *.jpeg *.webp "
            "*.wav *.mp3 *.m4a);;All files (*.*)",
        )
        if paths:
            self._ingest_paths(list(paths))

    def _copy_src_to_notes_files(self, src: Path) -> Optional[Path]:
        """Copy a single file into notes/files with a timestamp prefix; return destination or None."""
        src = src.expanduser()
        if not src.is_file():
            return None
        dest_dir = self._notes_root / "files"
        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = f"{ts}_{src.name}"
        target = dest_dir / name
        n = 0
        while target.exists():
            n += 1
            target = dest_dir / f"{ts}_{n}_{src.name}"
        try:
            shutil.copy2(src, target)
            self._append_session(f"ingested file {target.name}")
            return target
        except Exception:
            return None

    def _ingest_paths(self, paths: list[str]) -> None:
        copied = 0
        errors: list[str] = []
        ingested: list[str] = []
        for raw in paths:
            src = Path(raw).expanduser()
            if not src.is_file():
                continue
            target = self._copy_src_to_notes_files(src)
            if target is not None:
                copied += 1
                ingested.append(str(target))
            else:
                errors.append(f"{src.name}: copy failed")
        self._refresh_list()
        self._notes_rag_auto_index(ingested)
        if errors:
            self.errorRaised.emit("; ".join(errors[:5]))
        self._set_status(f"Copied {copied} file(s) into notes/files/.")

    def _notes_rag_auto_index(self, paths: list[str]) -> None:
        """Forward freshly ingested notes to the global notes RAG for auto-indexing."""
        if not paths:
            return
        ws = getattr(self, "_ai_workspace", None)
        if ws is None or not hasattr(ws, "notes_rag_add_paths"):
            return
        try:
            ws.notes_rag_add_paths(paths)
        except Exception:
            pass

    def _audio_transcription_busy(self) -> bool:
        worker = self._transcription_worker
        return bool(worker is not None and worker.isRunning())

    def transcribe_audio_note(self, path_str: str) -> None:
        if self._audio_transcription_busy():
            current = Path(self._transcription_source_path).name if self._transcription_source_path else "audio"
            self._set_status(f"Audio transcription is already running for {current}.")
            return
        p = Path(path_str).expanduser()
        if not p.is_file():
            self._set_status("That audio note is not available for transcription.")
            return
        if not is_transcribable_audio_note(p):
            self._set_status(f"Cannot transcribe {p.suffix or 'this file'} audio note type.")
            return
        self._pause_all_note_audio_players()
        self._transcription_source_path = str(p)
        worker = _AudioTranscriptionWorker(str(p), self)
        self._transcription_worker = worker
        worker.completed.connect(self._on_audio_transcription_completed)
        worker.failed.connect(self._on_audio_transcription_failed)
        worker.finished.connect(self._on_audio_transcription_thread_finished)
        worker.finished.connect(worker.deleteLater)
        worker.start()
        self._append_session(f"audio transcription started {p.name}")
        self._set_status(f"Transcribing {p.name} with local Vosk...")

    def _next_transcript_note_path(self, source_path: str) -> Path:
        dest_dir = self._notes_root / "files"
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / transcript_filename_for_source(source_path)
        base_stem = path.stem
        suffix = path.suffix
        n = 0
        while path.exists():
            n += 1
            path = dest_dir / f"{base_stem}_{n}{suffix}"
        return path

    def _on_audio_transcription_completed(self, payload: dict[str, Any]) -> None:
        source_name = str(payload.get("source_name") or Path(str(payload.get("source_path") or "")).name or "audio")
        capability = str(payload.get("capability") or "")
        text = str(payload.get("text") or "").strip()
        if capability == "capability_unavailable":
            msg = (
                "Audio transcription provider is not available for this file."
            )
            self._set_status(msg)
            self.errorRaised.emit(msg)
            self._append_session(f"audio transcription unavailable {source_name}")
            return
        if capability == "dependency_unavailable":
            msg = "Audio transcription needs Vosk installed. Install with: pip install vosk"
            self._set_status(msg)
            self.errorRaised.emit(msg)
            self._append_session(f"audio transcription dependency missing {source_name}")
            return
        if capability == "model_unavailable":
            msg = f"Audio transcription needs a local Vosk model. {vosk_setup_hint()}"
            self._set_status(msg)
            self.errorRaised.emit(msg)
            self._append_session(f"audio transcription model missing {source_name}")
            return
        if capability == "decode_unavailable":
            msg = "Audio transcription needs ffmpeg to decode this audio into Vosk-ready WAV."
            self._set_status(msg)
            self.errorRaised.emit(msg)
            self._append_session(f"audio transcription decoder missing {source_name}")
            return
        if capability == "decode_failed":
            msg = f"Audio transcription could not decode {source_name}; check the media file."
            self._set_status(msg)
            self.errorRaised.emit(msg)
            self._append_session(f"audio transcription decode failed {source_name}")
            return
        if capability == "failed":
            msg = f"Audio transcription failed for {source_name}. Check Vosk model compatibility and the audio file format."
            self._set_status(msg)
            self.errorRaised.emit(msg)
            self._append_session(f"audio transcription failed {source_name}")
            return
        if not text:
            msg = f"No speech transcript returned for {source_name}."
            self._set_status(msg)
            self._append_session(f"audio transcription empty {source_name}")
            return
        markdown = format_transcript_markdown(payload, notes_root=self._notes_root)
        note_path = self._next_transcript_note_path(str(payload.get("source_path") or source_name))
        try:
            note_path.write_text(markdown, encoding="utf-8")
        except Exception as exc:
            msg = f"Could not save transcript note: {exc}"
            self._set_status(msg)
            self.errorRaised.emit(msg)
            return
        self._editor_raw.setPlainText(markdown)
        self._editor_preview.setMarkdown(markdown)
        self._btn_view_md.setChecked(True)
        self._editor_stack.setCurrentIndex(1)
        self._append_session(f"saved audio transcript {note_path.name}")
        self._refresh_list()
        self._notes_rag_auto_index([str(note_path)])
        self._set_status(f"Saved transcript note {note_path.name}")

    def _on_audio_transcription_failed(self, message: str) -> None:
        msg = f"Audio transcription failed: {message}"
        self._set_status(msg)
        self.errorRaised.emit(msg)

    def _on_audio_transcription_thread_finished(self) -> None:
        self._transcription_worker = None
        self._transcription_source_path = ""

    def _on_note_view_mode_changed(self, button_id: int) -> None:
        if button_id == 0:
            self._editor_preview.setMarkdown(self._editor_raw.toPlainText())
            self._editor_stack.setCurrentIndex(1)
        else:
            self._editor_stack.setCurrentIndex(0)

    def _on_raw_editor_files_dropped(self, paths: list[str]) -> None:
        if len(paths) == 1 and Path(paths[0]).suffix.lower() in _MD_DROP_SUFFIXES:
            self._on_editor_markdown_dropped(paths[0])
        else:
            self._ingest_paths(paths)

    def _open_path_in_notes_editor(self, path_str: str) -> None:
        p = Path(path_str)
        if not p.is_file():
            return
        suf = p.suffix.lower()
        if suf not in _TEXT_OPEN_SUFFIXES:
            return
        try:
            text = p.read_text(encoding="utf-8")
        except Exception as exc:
            self.errorRaised.emit(f"Could not open file: {exc}")
            self._set_status(f"Could not open {p.name}.")
            return
        self._editor_raw.setPlainText(text)
        if suf in _MD_DROP_SUFFIXES:
            self._editor_preview.setMarkdown(text)
            self._btn_view_md.setChecked(True)
            self._editor_stack.setCurrentIndex(1)
        else:
            self._editor_preview.clear()
            self._btn_view_raw.setChecked(True)
            self._editor_stack.setCurrentIndex(0)
        self._set_status(f"Opened {p.name} in editor")

    def _on_list_item_clicked(self, item: QListWidgetItem) -> None:
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if path and Path(path).suffix.lower() in _TEXT_OPEN_SUFFIXES:
            self._open_path_in_notes_editor(path)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        et = event.type()
        if et == QEvent.Type.DragEnter and isinstance(event, QDragEnterEvent):
            if _paths_from_mime_for_notes_ingest(event.mimeData()):
                event.acceptProposedAction()
                return True
            return False
        if et == QEvent.Type.DragMove and isinstance(event, QDragMoveEvent):
            if _paths_from_mime_for_notes_ingest(event.mimeData()):
                event.acceptProposedAction()
                return True
            return False
        if et == QEvent.Type.Drop and isinstance(event, QDropEvent):
            paths = _paths_from_mime_for_notes_ingest(event.mimeData())
            if paths:
                self._ingest_paths(paths)
                event.acceptProposedAction()
                return True
            return False
        return super().eventFilter(watched, event)

    def _setup_notes_file_drop_filters(self) -> None:
        self._notes_drop_filter_objs = []
        skip_types: tuple[type, ...] = (
            _NotesRawEditor,
            QComboBox,
            QAbstractButton,
            QLineEdit,
            QPlainTextEdit,
            QTextBrowser,
            QTabWidget,
        )
        for w in self.findChildren(QWidget):
            if isinstance(w, skip_types):
                continue
            w.installEventFilter(self)
            self._notes_drop_filter_objs.append(w)

    def _teardown_notes_file_drop_filters(self) -> None:
        for o in self._notes_drop_filter_objs:
            try:
                o.removeEventFilter(self)
            except Exception:
                pass
        self._notes_drop_filter_objs = []

    def _on_raw_note_text_changed(self) -> None:
        if self._editor_stack.currentIndex() == 1:
            self._editor_preview.setMarkdown(self._editor_raw.toPlainText())

    def _on_editor_markdown_dropped(self, path_str: str) -> None:
        src = Path(path_str)
        target = self._copy_src_to_notes_files(src)
        read_from = target if target is not None else src
        try:
            text = read_from.read_text(encoding="utf-8")
        except Exception as exc:
            self.errorRaised.emit(f"Could not read Markdown file: {exc}")
            self._set_status(f"Could not read {read_from.name}.")
            return
        self._editor_raw.setPlainText(text)
        self._editor_preview.setMarkdown(text)
        self._btn_view_md.setChecked(True)
        self._editor_stack.setCurrentIndex(1)
        self._refresh_list()
        if target is not None:
            self._notes_rag_auto_index([str(target)])
        self._set_status(
            f"Loaded {read_from.name} into editor (Markdown preview)"
            + (f"; saved copy as {target.name}" if target is not None else "")
            + "."
        )

    def _on_save_text_note(self) -> None:
        text = self._editor_raw.toPlainText().strip()
        if not text:
            self._set_status("Nothing to save — write something in the editor first.")
            return
        dest_dir = self._notes_root / "files"
        dest_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = dest_dir / f"note_{ts}.md"
        n = 0
        while path.exists():
            n += 1
            path = dest_dir / f"note_{ts}_{n}.md"
        try:
            path.write_text(text + "\n", encoding="utf-8")
        except Exception as exc:
            self.errorRaised.emit(f"Save note failed: {exc}")
            return
        self._editor_raw.clear()
        self._editor_preview.clear()
        self._btn_view_raw.setChecked(True)
        self._editor_stack.setCurrentIndex(0)
        self._append_session(f"saved text note {path.name}")
        self._refresh_list()
        self._notes_rag_auto_index([str(path)])
        self._set_status(f"Saved {path.name}")

    def _on_open_folder(self) -> None:
        reveal_in_finder(str(self._notes_root))

    def _is_notes_asset_path(self, path: Path) -> bool:
        """True if path is a regular file directly under notes/files|recordings|sessions|captures."""
        try:
            rel = path.resolve().relative_to(self._notes_root.resolve())
        except ValueError:
            return False
        parts = rel.parts
        if len(parts) != 2:
            return False
        return parts[0] in _NOTES_SUBDIRS and path.is_file()

    def _unload_note_audio_row_if_path(self, path_str: str) -> None:
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it is None:
                continue
            w = self._list.itemWidget(it)
            if isinstance(w, _NotesInlineAudioRow) and w.path_str() == path_str:
                w.unload_wave()
                return

    def _request_delete_notes_asset(self, path_str: str) -> None:
        if self._recording:
            self._set_status("Stop voice recording before deleting a notes file.")
            return
        try:
            path = Path(path_str).expanduser().resolve()
        except Exception:
            self._set_status("Invalid path.")
            return
        if self._audio_transcription_busy():
            current_path: Optional[Path]
            try:
                current_path = Path(self._transcription_source_path).expanduser().resolve()
            except Exception:
                current_path = None
            if current_path is not None and path == current_path:
                self._set_status("Wait for audio transcription to finish before deleting this audio note.")
                return
        if not self._is_notes_asset_path(path):
            self._set_status("Can only delete files under the notes vault (files, recordings, sessions, captures).")
            return
        if self._session_path is not None:
            try:
                if path.resolve() == self._session_path.resolve():
                    self._set_status("End the current session before deleting its log file.")
                    return
            except Exception:
                pass
        reply = QMessageBox.question(
            self,
            "Delete notes file",
            f"Permanently delete this file? This cannot be undone.\n\n{path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._unload_note_audio_row_if_path(path_str)
        try:
            path.unlink()
        except Exception as exc:
            self.errorRaised.emit(f"Delete failed: {exc}")
            self._set_status(f"Could not delete {path.name}: {exc}")
            return
        self._append_session(f"deleted {path.name}")
        self._refresh_list()
        self._set_status(f"Deleted {path.name}")

    def _on_toggle_session(self) -> None:
        if self._session_path is None:
            sess_dir = self._notes_root / "sessions"
            sess_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._session_path = sess_dir / f"session_{ts}.md"
            header = (
                f"# CV Ops session {ts} (UTC)\n\n"
                "Append-only log of actions in the Notes tab during this session.\n\n"
                "---\n\n"
            )
            try:
                self._session_path.write_text(header, encoding="utf-8")
            except Exception as exc:
                self._session_path = None
                self.errorRaised.emit(f"Could not start session: {exc}")
                return
            self._append_session("session started")
            self._update_session_button_label()
            self._set_status(f"Session log: {self._session_path.name}")
            return

        self._append_session("session ended")
        try:
            with self._session_path.open("a", encoding="utf-8") as f:
                f.write("\n---\n\nSession closed.\n")
        except Exception:
            pass
        self._session_path = None
        self._update_session_button_label()
        self._set_status("Session ended.")

    def _update_session_button_label(self) -> None:
        if self._session_path is None:
            self._btn_session.setText("Start session")
            self._btn_session.setToolTip("Begin a timestamped session log under notes/sessions/.")
        else:
            self._btn_session.setText("End session")
            self._btn_session.setToolTip("Close the current session log.")

    def _append_session(self, line: str) -> None:
        if self._session_path is None:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        try:
            with self._session_path.open("a", encoding="utf-8") as f:
                f.write(f"- **{stamp}** — {line}\n")
        except Exception:
            pass

    def _on_toggle_record(self) -> None:
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        if self._recording:
            return
        self._stop_webcam()
        self._pause_all_note_audio_players()
        if not self._microphone_available():
            self._set_status("No microphone device available.")
            return
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._set_status("ffmpeg not found — install ffmpeg to enable voice recording.")
            return
        self._recording_pcm = bytearray()
        self._record_token += 1
        token = self._record_token
        proc = QProcess(self)
        proc.setProgram(ffmpeg)
        if sys.platform == "darwin":
            input_args = ["-f", "avfoundation", "-i", ":0"]
        elif sys.platform == "win32":
            input_args = ["-f", "dshow", "-i", "audio=default"]
        else:
            input_args = ["-f", "pulse", "-i", "default"]
        proc.setArguments(
            input_args + ["-ar", "16000", "-ac", "1", "-f", "s16le", "pipe:1"]
        )
        proc.readyReadStandardOutput.connect(lambda _t=token: self._record_stdout(_t))
        proc.finished.connect(lambda _ec, _es, _t=token: self._on_record_finished(_t))
        self._record_process = proc
        self._recording = True
        proc.start()
        self._btn_record.setText("Stop recording")
        self._append_session("voice recording started")
        self._set_status("Recording… click Stop recording to save a WAV under notes/recordings/.")

    def _record_stdout(self, token: int) -> None:
        if token != self._record_token or self._record_process is None:
            return
        chunk = bytes(self._record_process.readAllStandardOutput())
        if chunk:
            self._recording_pcm.extend(chunk)

    def _stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        self._btn_record.setText("Record voice")
        proc = self._record_process
        if proc is not None and proc.state() != QProcess.ProcessState.NotRunning:
            try:
                proc.write(b"q")
            except Exception:
                proc.terminate()
        self._set_status("Stopping recording — finalizing WAV…")

    def _on_record_finished(self, token: int) -> None:
        if token != self._record_token:
            return
        proc = self._record_process
        self._record_process = None
        if proc is not None:
            tail = bytes(proc.readAllStandardOutput())
            if tail:
                self._recording_pcm.extend(tail)
            proc.deleteLater()
        rec_dir = self._notes_root / "recordings"
        rec_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = rec_dir / f"voice_{ts}.wav"
        n = 0
        while out.exists():
            n += 1
            out = rec_dir / f"voice_{ts}_{n}.wav"
        frames = bytes(self._recording_pcm)
        self._recording_pcm.clear()
        try:
            with wave.open(str(out), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(frames)
        except Exception as exc:
            self.errorRaised.emit(f"Could not save recording: {exc}")
            self._set_status("Recording failed to save.")
            return
        self._append_session(f"saved voice recording {out.name}")
        self._refresh_list()
        if len(frames) == 0:
            self._set_status(
                f"Saved {out.name}, but no audio data captured — "
                f"check System Settings > Privacy & Security > Microphone for ffmpeg/terminal access."
            )
        else:
            self._set_status(f"Saved {out.name}")

    def _refresh_list(self) -> None:
        self._teardown_notes_file_drop_filters()
        self._unload_all_note_audio_rows()
        self._list.clear()

        items: list[tuple[float, str, str]] = []
        files_dir = self._notes_root / "files"
        rec_dir = self._notes_root / "recordings"
        sess_dir = self._notes_root / "sessions"
        cap_dir = self._notes_root / "captures"
        for base, label in (
            (files_dir, "file"),
            (rec_dir, "recording"),
            (sess_dir, "session"),
            (cap_dir, "capture"),
        ):
            if not base.is_dir():
                continue
            try:
                for p in sorted(base.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    if not p.is_file():
                        continue
                    try:
                        mtime = p.stat().st_mtime
                    except Exception:
                        mtime = 0.0
                    items.append((mtime, str(p), label))
            except Exception:
                continue
        items.sort(key=lambda t: t[0], reverse=True)
        for _mtime, path_str, kind in items[:500]:
            p = Path(path_str)
            if p.is_file() and _path_audio_previewable(p):
                it = QListWidgetItem()
                it.setData(Qt.ItemDataRole.UserRole, path_str)
                it.setToolTip(path_str)
                row = _NotesInlineAudioRow(path_str, kind, p.name, self)
                self._list.addItem(it)
                self._list.setItemWidget(it, row)
                it.setSizeHint(row.sizeHint())
            else:
                it = QListWidgetItem()
                it.setData(Qt.ItemDataRole.UserRole, path_str)
                it.setToolTip(path_str)
                row = _NotesFileRow(path_str, kind, p.name, self)
                self._list.addItem(it)
                self._list.setItemWidget(it, row)
                it.setSizeHint(row.sizeHint())

        self._setup_notes_file_drop_filters()

    def has_audio_analyze_http(self) -> bool:
        return self._http_json is not None

    def run_audio_recognition_analyze(self, path_str: str, wave: AudioWaveformPlayer) -> None:
        """POST /audio/analyze — same contract as DatasetPanel._analyze_selected_audio_asset."""
        if self._http_json is None:
            self._set_status("Audio analysis requires the CV Ops server.")
            return
        p = Path(path_str)
        if not p.is_file():
            self._set_status("That file is not available for analysis.")
            return
        start_ms, end_ms = _waveform_selection_range_ms(wave)
        is_wav = p.suffix.lower() == ".wav"
        body: dict[str, Any] = {"path": path_str, "start_ms": start_ms}
        if end_ms is not None:
            body["end_ms"] = end_ms
        extra = "" if is_wav else " (decoding media — may take a few seconds)"
        self._set_status(f"Analyzing: {p.name}{extra}")
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        try:
            payload = self._http_json("POST", "/audio/analyze", body, timeout=180.0)
        except Exception as exc:
            msg = f"Audio analysis failed for '{p.name}': {exc}"
            self._set_status(msg)
            self.errorRaised.emit(msg)
            return
        metrics = payload.get("metrics") if isinstance(payload, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}
        summary = _format_notes_audio_metrics(metrics)
        self._set_status(f"[audio] {p.name}: {summary}")
        if not is_wav and metrics:
            QMessageBox.information(
                self,
                f"Audio analysis — {p.name}",
                summary,
            )

    def _on_item_activated(self, item: QListWidgetItem) -> None:
        path = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if path:
            reveal_in_finder(path)

    def _set_status(self, text: str) -> None:
        self._status.setText(text)
