"""Per-project Ollama chat and RAG controls for the CV Ops Notes tab."""

from __future__ import annotations

import html
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from PyQt6.QtCore import QEvent, QMimeData, QObject, QSize, QThread, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QKeyEvent,
    QResizeEvent,
    QTextCursor,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QLayout,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QTextBrowser,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:  # python-markdown is optional; fall back to escape() if missing
    import markdown as _md_lib  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised only when dep is missing
    _md_lib = None  # type: ignore[assignment]

_MD_EXTENSIONS = ("fenced_code", "tables", "sane_lists", "nl2br")


def _render_markdown_html(text: str) -> str:
    """Render assistant/user message text as HTML.

    Falls back to escape()+<br> if python-markdown isn't installed so the
    chat keeps working without a hard dependency.
    """
    body = str(text or "")
    if _md_lib is None:
        return html.escape(body).replace("\n", "<br>")
    try:
        return _md_lib.markdown(body, extensions=list(_MD_EXTENSIONS), output_format="html5")
    except Exception:
        return html.escape(body).replace("\n", "<br>")


_DEFAULT_PROMPT_SUGGESTIONS: tuple[tuple[str, str], ...] = (
    ("Summarize my recent notes", "Summarize the most recent notes in this project — pull out decisions, open questions, and TODOs."),
    ("Explain a CV concept", "Explain the trade-offs between IoU and GIoU for object detection, with a short worked example."),
    ("Draft a dataset spec", "Draft a one-page dataset spec for a new defect-detection scenario: classes, capture conditions, edge cases, label rules."),
    ("Debug a failed run", "I had a training run diverge after epoch 3. Help me list the most likely causes and the fastest checks to confirm each."),
)

from mlops.ChatbotAndRag.solo_rag_chat.chat_manager import ChatManager
from mlops.ChatbotAndRag.solo_rag_chat.cloud_chat_workers import (
    AnthropicChatWorker,
    GeminiChatWorker,
    OpenAICompatChatWorker,
    chat_messages_to_anthropic,
    chat_messages_to_gemini,
    chat_messages_to_openai,
)
from mlops.ChatbotAndRag.solo_rag_chat.workers import OllamaWorker, RAGWorker

from ..tacitus_mcp import TacitusMcpSurface, parse_controlled_run_request
from .collapsible_section import CollapsibleSection
from .cvops_theme import cvops_color, cvops_rgba, repolish
from .speech_support import (
    SpeechDictationController,
    TtsPlaybackBar,
    list_system_voices,
    microphone_available,
    text_to_speech_available,
)
from ..ollama_model_discovery import (
    choose_ollama_embedding_model,
    discover_local_gguf_files,
    discover_ollama_model_tags,
)
from .notes_ai_keys import (
    DEFAULT_ASSISTANT_NAME,
    KEY_ANTHROPIC,
    KEY_ASSISTANT_NAME,
    KEY_GEMINI,
    KEY_GROK,
    KEY_LOCAL_GGUF_MODELS,
    KEY_OPENAI,
    KEY_SYSTEM_PROMPT,
    KEY_VOICE_PROFILE,
    OLLAMA_DEFAULT_MODELS,
    SYSTEM_PROMPT_MAX_CHARS,
    VOICE_PRESETS,
    assistant_display_name,
    ai_settings_path,
    default_voice_profile,
    keyring_available,
    load_ai_settings,
    local_gguf_models,
    model_catalog_entries,
    parse_route_key,
    save_ai_settings,
    system_prompt,
    voice_profile,
)
from . import notes_ai_memory as ai_memory
from .notes_ai_memory import INGESTED_MEMORY_NAMESPACE
from .notes_spaces import (
    DEFAULT_SPACE_ID,
    list_space_ids,
    notes_chats_dir,
    read_space_goals,
    read_space_import_source,
    read_space_pinned,
    read_space_title,
    set_space_pinned,
    update_space_meta_title,
)
from ...config import ROOT_DIR

_RAG_INDEXABLE_SUFFIXES = frozenset({".txt", ".md", ".markdown", ".pdf"})
_MENTION_TOKEN_RE = re.compile(r"(?<![\w.])@(?:\"([^\"]+)\"|'([^']+)'|([^\s,;:)\]}]+))")
_MENTION_MAX_FILES = 8
_MENTION_MAX_CHARS_PER_FILE = 6000
_MENTION_MAX_TOTAL_CHARS = 24000
_CHAT_JOBS_METADATA_KEY = "cvops_jobs"
_CHAT_ARTIFACTS_METADATA_KEY = "cvops_artifacts"
_PROJECT_LEDGER_FILE = "events_artifacts.json"
_PROJECT_EVENTS_KEY = "events"
_PROJECT_ARTIFACTS_KEY = "artifacts"
_ACTIVE_JOB_STATES = frozenset(
    {"queued", "running", "pending", "accepted", "starting", "started", "training", "scraping", "indexing"}
)
_FINAL_JOB_STATES = frozenset(
    {"done", "completed", "complete", "succeeded", "success", "error", "failed", "canceled", "cancelled", "stopped", "aborted"}
)
_ACTIVITY_FRAMES = ("|", "/", "-", "\\")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_record_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _upsert_record(
    records: list[dict[str, Any]],
    record: dict[str, Any],
    *,
    key: str,
    limit: int = 80,
) -> list[dict[str, Any]]:
    clean = {str(k): v for k, v in dict(record or {}).items() if v not in (None, "")}
    ident = str(clean.get(key) or "").strip()
    out: list[dict[str, Any]] = []
    replaced = False
    for item in _coerce_record_list(records):
        if ident and str(item.get(key) or "").strip() == ident:
            merged = dict(item)
            merged.update(clean)
            out.append(merged)
            replaced = True
        else:
            out.append(item)
    if not replaced:
        out.append(clean)
    return out[-max(1, int(limit)) :]


def _normalize_job_state(state: object) -> str:
    return str(state or "").strip().lower()


def _job_state_is_active(state: object) -> bool:
    normalized = _normalize_job_state(state)
    return normalized in _ACTIVE_JOB_STATES and normalized not in _FINAL_JOB_STATES


def _extract_file_mentions(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in _MENTION_TOKEN_RE.finditer(str(text or "")):
        raw = next((group for group in match.groups() if group), "")
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _read_text_excerpt(path: Path, *, max_chars: int = _MENTION_MAX_CHARS_PER_FILE) -> tuple[str, bool]:
    with path.open("rb") as fh:
        raw = fh.read(max(1, int(max_chars)) * 4 + 1)
    truncated = len(raw) > max(1, int(max_chars)) * 4
    text = raw.decode("utf-8", errors="replace")
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return text, truncated


def _rag_mime_has_local_paths(mime: Optional[QMimeData]) -> bool:
    if mime is None or not mime.hasUrls():
        return False
    for url in mime.urls():
        if url.isLocalFile():
            return True
    return False


def _rag_collect_local_paths_from_mime(mime: Optional[QMimeData]) -> list[str]:
    """Local files and folders (recursive) for RAG queue; folders contribute indexable files only."""
    if mime is None or not mime.hasUrls():
        return []
    out: list[str] = []
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        p = Path(url.toLocalFile())
        if p.is_file():
            out.append(str(p))
        elif p.is_dir():
            try:
                for child in sorted(p.rglob("*")):
                    if child.is_file() and child.suffix.lower() in _RAG_INDEXABLE_SUFFIXES:
                        out.append(str(child))
            except OSError:
                continue
    return out


def _collect_local_paths_from_mime(mime: Optional[QMimeData]) -> list[str]:
    if mime is None or not mime.hasUrls():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        p = Path(url.toLocalFile()).expanduser()
        try:
            key = str(p.resolve())
        except (OSError, ValueError):
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _gguf_collect_local_paths_from_mime(mime: Optional[QMimeData]) -> list[str]:
    if mime is None or not mime.hasUrls():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        p = Path(url.toLocalFile())
        candidates: list[Path] = []
        if p.is_file() and p.suffix.lower() == ".gguf":
            candidates = [p]
        elif p.is_dir():
            try:
                candidates = [child for child in sorted(p.rglob("*.gguf")) if child.is_file()]
            except OSError:
                candidates = []
        for candidate in candidates:
            try:
                key = str(candidate.expanduser().resolve())
            except (OSError, ValueError):
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def _gguf_mime_has_local_paths(mime: Optional[QMimeData]) -> bool:
    return bool(_gguf_collect_local_paths_from_mime(mime))


def _rag_dependencies_available() -> bool:
    try:
        from mlops.ChatbotAndRag.solo_rag_chat.rag_system import RAG_DEPENDENCIES_AVAILABLE

        return bool(RAG_DEPENDENCIES_AVAILABLE)
    except Exception:
        return False


def _reset_rag_singleton() -> None:
    try:
        from mlops.ChatbotAndRag.solo_rag_chat.rag_system import reset_rag_system

        reset_rag_system()
    except Exception:
        pass


def _apply_rag_config_to_space(
    *,
    space_root: Path,
    model_id: str,
    embedding_backend: str,
    embedding_model: str,
    ollama_base_url: str,
) -> None:
    from mlops.ChatbotAndRag.solo_rag_chat.rag_system import reset_rag_system, set_rag_config

    reset_rag_system()
    set_rag_config(
        model_id=model_id,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        rag_index_path=str(space_root / "rag_index"),
    )


# Directory name (under the notes vault container) for the global, vault-wide
# notes RAG index. This is the SECONDARY RAG: it shares the same engine config as
# the per-project chat RAG but keeps its own FAISS index aggregating notes from
# every project space.
NOTES_RAG_INDEX_DIRNAME = "notes_rag_index"

# Registry namespace for the global notes RAG (mirrors
# mlops.ChatbotAndRag.solo_rag_chat.rag_system.NOTES_NAMESPACE).
_NOTES_NAMESPACE = "notes"


def _notes_vault_root_from_space(space_root: Path) -> Path:
    """Notes vault container (``.../notes``) from a space root (``.../notes/spaces/<id>``)."""
    # space_root = <vault>/spaces/<id>  ->  parent.parent = <vault>
    return space_root.parent.parent


def _set_notes_rag_config(
    *,
    vault_root: Path,
    model_id: str,
    embedding_backend: str,
    embedding_model: str,
    ollama_base_url: str,
    reset: bool = False,
) -> None:
    """Point the global notes-RAG namespace at ``<vault>/notes_rag_index``.

    Engine keys (model/embeddings/URL) land in the shared engine config, so the
    notes RAG always uses the same engine as the chat RAG ("share 1"). Only the
    index path is namespace-specific.
    """
    from mlops.ChatbotAndRag.solo_rag_chat.rag_system import (
        NOTES_NAMESPACE,
        reset_rag_system,
        set_rag_config,
    )

    if reset:
        reset_rag_system(NOTES_NAMESPACE)
    set_rag_config(
        namespace=NOTES_NAMESPACE,
        model_id=model_id,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        rag_index_path=str(vault_root / NOTES_RAG_INDEX_DIRNAME),
    )


def _set_ingested_memory_rag_config(
    *,
    vault_root: Path,
    model_id: str,
    embedding_backend: str,
    embedding_model: str,
    ollama_base_url: str,
    reset: bool = False,
) -> None:
    """Point the global ingested-memory namespace at ``<vault>/ingested_memory_rag_index``.

    Shares the same engine as the chat/notes RAG; only the index path differs.
    """
    from mlops.ChatbotAndRag.solo_rag_chat.rag_system import (
        reset_rag_system,
        set_rag_config,
    )

    if reset:
        reset_rag_system(INGESTED_MEMORY_NAMESPACE)
    set_rag_config(
        namespace=INGESTED_MEMORY_NAMESPACE,
        model_id=model_id,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        ollama_base_url=ollama_base_url,
        rag_index_path=str(ai_memory.ingested_memory_index_dir(vault_root)),
    )


def _compact_tacitus_mcp_catalog() -> dict[str, Any]:
    catalog = TacitusMcpSurface.structured_json_tool_catalog()
    tools: list[dict[str, Any]] = []
    for tool in catalog.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        schema = tool.get("input_schema") if isinstance(tool.get("input_schema"), dict) else {}
        tools.append(
            {
                "name": str(tool.get("name") or ""),
                "mcp_name": str(tool.get("mcp_name") or ""),
                "description": str(tool.get("description") or ""),
                "properties": sorted((schema.get("properties") or {}).keys()) if isinstance(schema.get("properties"), dict) else [],
                "required": list(schema.get("required") or []),
            }
        )
    return {
        "tools": tools,
        "call_shape": catalog.get("call_shape") or {"tool": "provider_safe_tool_name", "arguments": {}},
    }


def _build_tacitus_mcp_prompt_block(context: dict[str, Any], catalog: dict[str, Any]) -> str:
    safe_context = {
        "active_project": context.get("active_project") if isinstance(context.get("active_project"), dict) else {},
        "active_scenario": str(context.get("active_scenario") or ""),
        "selected_dataset": str(context.get("selected_dataset") or ""),
    }
    ingested_memory = str(context.get("ingested_memory") or "").strip()
    if ingested_memory:
        safe_context["ingested_memory"] = ingested_memory
    return "\n".join(
        [
            "[Tacitus MCP tools]",
            "For CV Ops actions or CV Ops state lookup, answer with exactly one JSON object and no markdown.",
            "Use this shape: {\"tool\":\"provider_safe_tool_name\",\"arguments\":{...}}.",
            "If no tool is needed, answer normally.",
            "Never set promotion_request confirmed/manual_confirmed true unless the user explicitly confirms promotion.",
            "Current context JSON:",
            json.dumps(safe_context, ensure_ascii=True, separators=(",", ":")),
            "Available tools JSON:",
            json.dumps(catalog, ensure_ascii=True, separators=(",", ":")),
            "[/Tacitus MCP tools]",
        ]
    )


def _build_ollama_prompt(
    messages: List[dict],
    assistant_name: str = DEFAULT_ASSISTANT_NAME,
    *,
    mcp_context: Optional[dict[str, Any]] = None,
    mcp_catalog: Optional[dict[str, Any]] = None,
) -> str:
    assistant_label = str(assistant_name or "").strip() or DEFAULT_ASSISTANT_NAME
    parts: List[str] = []
    if mcp_catalog is not None:
        parts.append(_build_tacitus_mcp_prompt_block(dict(mcp_context or {}), dict(mcp_catalog or {})))
    for m in messages[-40:]:
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role in ("user", "human"):
            parts.append(f"User: {content}")
        elif role in ("assistant", "ai", "model"):
            parts.append(f"{assistant_label}: {content}")
        else:
            parts.append(f"{role}: {content}")
    if not parts:
        return f"{assistant_label}:"
    return "\n\n".join(parts) + f"\n\n{assistant_label}:"


def _extract_structured_mcp_tool_call(text: str) -> Optional[dict[str, Any]]:
    body = str(text or "").strip()
    if not body:
        return None
    fence = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", body, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        body = fence.group(1).strip()
    if not body.startswith("{"):
        return None
    try:
        payload = json.loads(body)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    source = payload.get("function") if isinstance(payload.get("function"), dict) else payload
    if not any(str(source.get(key) or payload.get(key) or "").strip() for key in ("tool", "name")):
        return None
    return payload


class _ComposerMessageEdit(QPlainTextEdit):
    """Tall composer field: Ctrl+Enter sends; Enter/Shift+Enter inserts a newline."""

    sendRequested = pyqtSignal()
    pathsDropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            self.sendRequested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if _collect_local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if _collect_local_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        paths = _collect_local_paths_from_mime(event.mimeData())
        if paths:
            self.pathsDropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)


class _RagDocListWidget(QListWidget):
    """Accepts dropped file URLs and forwards local paths to the parent RAG panel."""

    pathsDropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("notesRagDocList")
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if _rag_mime_has_local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if _rag_mime_has_local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        paths = _rag_collect_local_paths_from_mime(event.mimeData())
        if paths:
            self.pathsDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class _GgufModelListWidget(QListWidget):
    """Accepts dropped GGUF files and forwards their paths to the AI settings panel."""

    pathsDropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("notesLocalModelList")
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if _gguf_mime_has_local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if _gguf_mime_has_local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        paths = _gguf_collect_local_paths_from_mime(event.mimeData())
        if paths:
            self.pathsDropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()


class _RagTabDropHost(QWidget):
    """RAG tab root: accept file/folder drops on empty space and on child widgets (preview, output, fields)."""

    def __init__(self, workspace: "NotesAiWorkspace", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._ws = workspace
        self.setObjectName("notesRagTabSurface")
        self.setAcceptDrops(True)
        self._rag_drop_filter_installed = False

    def rag_install_child_drop_filters(self) -> None:
        if self._rag_drop_filter_installed:
            return
        for w in self.findChildren(QWidget):
            if w is self or isinstance(w, _RagDocListWidget):
                continue
            w.installEventFilter(self)
        self._rag_drop_filter_installed = True

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if _rag_mime_has_local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if _rag_mime_has_local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        paths = _rag_collect_local_paths_from_mime(event.mimeData())
        if paths:
            self._ws._rag_on_paths_dropped(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        et = event.type()
        if et == QEvent.Type.DragEnter and isinstance(event, QDragEnterEvent):
            if _rag_mime_has_local_paths(event.mimeData()):
                event.acceptProposedAction()
                return True
            return False
        if et == QEvent.Type.DragMove and isinstance(event, QDragMoveEvent):
            if _rag_mime_has_local_paths(event.mimeData()):
                event.acceptProposedAction()
                return True
            return False
        if et == QEvent.Type.Drop and isinstance(event, QDropEvent):
            paths = _rag_collect_local_paths_from_mime(event.mimeData())
            if paths:
                self._ws._rag_on_paths_dropped(paths)
                event.acceptProposedAction()
                return True
            return False
        return super().eventFilter(watched, event)


class _OverlayChatHost(QWidget):
    """Full-size transcript with the composer frame overlaid on the bottom (higher z-order)."""

    def __init__(
        self,
        chat_view: QTextBrowser,
        composer_frame: QFrame,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._chat = chat_view
        self._composer = composer_frame
        chat_view.setParent(self)
        composer_frame.setParent(self)
        chat_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._chat.installEventFilter(self)
        self._chat.viewport().installEventFilter(self)
        vbar = self._chat.verticalScrollBar()
        if vbar is not None:
            vbar.rangeChanged.connect(lambda *_: self._sync_overlay_geometry())
        hbar = self._chat.horizontalScrollBar()
        if hbar is not None:
            hbar.rangeChanged.connect(lambda *_: self._sync_overlay_geometry())

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if watched in (self._chat, self._chat.viewport()) and event.type() in (
            QEvent.Type.Resize,
            QEvent.Type.Move,
            QEvent.Type.Show,
            QEvent.Type.LayoutRequest,
        ):
            QTimer.singleShot(0, self._sync_overlay_geometry)
        return super().eventFilter(watched, event)

    def _sync_overlay_geometry(self) -> None:
        w, h = max(self.width(), 1), max(self.height(), 1)
        margin = 12
        usable_w = max(40, w - 2 * margin)
        # Keep the composer as a narrow dock so the conversation remains the
        # primary geometry instead of another giant control panel. Cap at the
        # available width (with a small floor) so it never overflows a narrow
        # card -- the previous 520px floor clipped the composer in the assistant
        # overlay; in wide windows the 72% term still dominates so this is a noop.
        cw = int(min(usable_w, max(360, min(860, int(usable_w * 0.72)))))
        lay = self._composer.layout()
        if lay is not None:
            lay.activate()
            try:
                hinted = int(lay.heightForWidth(int(cw)))
            except Exception:
                hinted = 0
            if hinted < 0:
                hinted = 0
        else:
            hinted = 0
        min_hint = int(self._composer.minimumSizeHint().height())
        sh = int(self._composer.sizeHint().height())
        tools_strip = self._composer.findChild(QFrame, "notesComposerToolsStrip")
        tools_open = bool(tools_strip is not None and tools_strip.isVisible())
        target = int(max(74 if not tools_open else 102, min(h * (0.14 if not tools_open else 0.19), 156)))
        ch = max(hinted, min_hint, sh, 64, target)
        max_ch = max(1, h - 2 * margin)
        ch = min(ch, max_ch)
        x0 = max(margin, (w - cw) // 2)
        y0 = h - ch - margin
        self._composer.setGeometry(x0, y0, cw, ch)
        self._composer.raise_()
        self._chat.setGeometry(0, 0, w, h)
        self._chat.setViewportMargins(0, 0, 0, min(h - 1, ch + margin + 8))

    def resizeEvent(self, event: QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_overlay_geometry()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._sync_overlay_geometry()


class _FullTextLabel(QLabel):
    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(str(text), parent)
        self._raw_text = str(text)
        self.setToolTip(self._raw_text)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._raw_text = str(text)
        self.setToolTip(self._raw_text)
        super().setText(self._raw_text)


class _AutoFitListWidget(QListWidget):
    """QListWidget that keeps every row's sizeHint width pinned to the current
    viewport width.

    Without this, rows are sized once at insert time and never refreshed, so a
    narrower pane leaves the overflow ("⋯") button off-screen until the user
    drags the splitter back out. Resizing the viewport now retags every item.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.viewport().installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if watched is self.viewport() and event.type() == QEvent.Type.Resize:
            self._refit_all_rows()
        return super().eventFilter(watched, event)

    def _refit_all_rows(self) -> None:
        w = max(1, self.viewport().width())
        for i in range(self.count()):
            item = self.item(i)
            if item is None:
                continue
            h = item.sizeHint().height() or 40
            item.setSizeHint(QSize(w, h))


class _NotesSidebarRowHost(QWidget):
    """Title + overflow for sidebar lists; clicks select the owning ``QListWidgetItem``."""

    def __init__(
        self,
        list_widget: QListWidget,
        item: QListWidgetItem,
        title: str,
        *,
        for_project: bool,
        workspace: "NotesAiWorkspace",
        active_jobs: int = 0,
        activity_label: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("notesSidebarRowHost")
        self._list = list_widget
        self._item = item
        self._for_project = for_project
        self._ws = workspace
        self._overflow_visible = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        if for_project:
            icon_lab = QLabel()
            icon_lab.setObjectName("notesSidebarRowIcon")
            icon_lab.setPixmap(
                workspace.style()
                .standardIcon(QStyle.StandardPixmap.SP_DirIcon)
                .pixmap(18, 18)
            )
            self._icon = icon_lab
            lay.addWidget(icon_lab, stretch=0)
        else:
            self._icon = None
        self._active_jobs = max(0, int(active_jobs or 0))
        self._activity_label = str(activity_label or "").strip()
        self._activity = QLabel()
        self._activity.setObjectName("notesSidebarActivity")
        self._activity.setFixedWidth(18)
        self._activity.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._activity.setToolTip(self._activity_label)
        self._activity.setVisible(self._active_jobs > 0)
        lay.addWidget(self._activity, stretch=0)
        self._title = _FullTextLabel(title)
        self._title.setObjectName("notesSidebarRowTitle")
        lay.addWidget(self._title, stretch=1)
        self._btn = QToolButton()
        self._btn.setObjectName("notesSidebarRowOverflow")
        self._btn.setText("⋯")
        self._btn.setAutoRaise(True)
        self._btn.setToolTip("Actions")
        self._btn.clicked.connect(self._on_overflow_clicked)
        lay.addWidget(self._btn)
        self._set_overflow_visible(False)
        self.update_activity_icon()

    def has_active_jobs(self) -> bool:
        return self._active_jobs > 0

    def update_activity_icon(self) -> None:
        if self._active_jobs <= 0:
            self._activity.setText("")
            self._activity.setVisible(False)
            return
        self._activity.setVisible(True)
        self._activity.setText(self._ws._activity_spinner_text())
        self._activity.setToolTip(
            self._activity_label
            or f"{self._active_jobs} running CV Ops job(s) attached to this chat"
        )

    def _set_overflow_visible(self, visible: bool) -> None:
        self._overflow_visible = bool(visible)
        self._btn.setVisible(self._overflow_visible)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._set_overflow_visible(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._set_overflow_visible(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self._list.setCurrentItem(self._item)
        super().mousePressEvent(event)

    def _on_overflow_clicked(self) -> None:
        self._list.setCurrentItem(self._item)
        menu = QMenu(self._btn)
        raw = self._item.data(Qt.ItemDataRole.UserRole)
        if raw is None:
            return
        ident = str(raw)
        if self._for_project:
            self._ws._fill_project_context_menu(menu, ident)
        else:
            self._ws._fill_chat_context_menu(menu, ident)
        menu.exec(self._btn.mapToGlobal(self._btn.rect().bottomLeft()))


class _AiImportWorker(QThread):
    """Background ingest of an exported ChatGPT/Claude history into a chats dir.

    Parsing a large ``conversations.json`` (ChatGPT exports can be hundreds of
    MB) is done off the GUI thread so the window never freezes during import.
    """

    finished = pyqtSignal(object)  # ImportResult | ProjectsImportResult
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # (conversations done, total) — drives ETA

    # Pace large imports: write 25 conversations, then yield the disk for 30ms.
    # This keeps a multi-thousand-chat export from saturating I/O in one burst.
    _BATCH_SIZE = 25
    _BATCH_PAUSE = 0.03

    def __init__(
        self,
        path: Path,
        chats_dir: Path,
        source: Optional[str],
        *,
        mode: str = "conversations",
        spaces_root: Optional[Path] = None,
        vault_root: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self._chats_dir = Path(chats_dir)
        self._source = source
        self._mode = mode
        self._spaces_root = spaces_root
        self._vault_root = vault_root

    def run(self) -> None:  # type: ignore[override]
        try:
            from . import notes_ai_import as ai_import

            if self._mode == "claude_projects":
                result = ai_import.ingest_claude_projects(
                    self._path,
                    self._spaces_root,
                    self._vault_root,
                    fallback_chats_dir=self._chats_dir,
                    batch_size=self._BATCH_SIZE,
                    batch_pause=self._BATCH_PAUSE,
                    on_progress=self.progress.emit,
                )
            else:
                result = ai_import.ingest_export_file(
                    self._path,
                    self._chats_dir,
                    source=self._source,
                    batch_size=self._BATCH_SIZE,
                    batch_pause=self._BATCH_PAUSE,
                    on_progress=self.progress.emit,
                )
        except Exception as exc:  # pragma: no cover - defensive thread guard
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class _MemoryBuildWorker(QThread):
    """Build ingested-memory docs (transcripts + summaries + detected memory files).

    Runs off the GUI thread because per-chat summarization can call a local
    model. Summaries are best-effort and skipped silently when unreachable.
    """

    finished = pyqtSignal(object)  # dict from ai_memory.build_memory_docs
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, int)  # (conversations done, total) — drives ETA

    def __init__(
        self,
        *,
        vault_root: Path,
        spaces_root: Path,
        source: str,
        export_path: Optional[Path],
        summarize: bool,
        model: Optional[str],
        base_url: Optional[str],
        force: bool = False,
    ) -> None:
        super().__init__()
        self._vault_root = Path(vault_root)
        self._spaces_root = Path(spaces_root)
        self._source = source
        self._export_path = export_path
        self._summarize = summarize
        self._model = model
        self._base_url = base_url
        self._force = force

    def run(self) -> None:  # type: ignore[override]
        try:
            result = ai_memory.build_memory_docs(
                vault_root=self._vault_root,
                spaces_root=self._spaces_root,
                source=self._source,
                export_path=self._export_path,
                summarize=self._summarize,
                model=self._model,
                base_url=self._base_url,
                force=self._force,
                on_progress=self.progress.emit,
            )
        except Exception as exc:  # pragma: no cover - defensive thread guard
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class _InlineTaskProgress(QFrame):
    """Non-modal progress strip embedded in the workspace.

    Ingesting a giant export and (especially) reindexing it into RAG memory can
    run for *hours* — summarizing every conversation with a local model. A modal
    dialog would lock the whole app for that whole time. This strip lives inside
    the workspace instead: the job runs on its background thread, this only
    reflects progress + ETA, and the user can keep chatting, browsing, and
    working. "Hide" tucks the strip away without stopping the job.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("inlineTaskProgress")
        self.setVisible(False)
        self.setStyleSheet(
            "#inlineTaskProgress {"
            " background: rgba(255,255,255,0.04);"
            " border-top: 1px solid rgba(255,255,255,0.08); }"
        )
        # Auto-dismiss a finished strip after a grace period; cancelled the
        # moment a new task starts so back-to-back phases never flicker away.
        self._autohide = QTimer(self)
        self._autohide.setSingleShot(True)
        self._autohide.timeout.connect(self.hide_banner)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 8, 12, 10)
        root.setSpacing(4)

        head = QHBoxLayout()
        head.setSpacing(8)
        self._title = QLabel("Working…")
        self._title.setStyleSheet("font-weight:600;")
        head.addWidget(self._title, 1)
        self._hide_btn = QToolButton()
        self._hide_btn.setText("Hide")
        self._hide_btn.setToolTip(
            "Hide this strip. The job keeps running in the background — you can "
            "keep working while it finishes."
        )
        self._hide_btn.clicked.connect(self.hide_banner)
        head.addWidget(self._hide_btn, 0)
        root.addLayout(head)

        self._bar = QProgressBar()
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        self._bar.setRange(0, 0)
        root.addWidget(self._bar)

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color: rgba(255,255,255,0.62);")
        root.addWidget(self._detail)

    def start(self, title: str, detail: str = "Starting…") -> None:
        self._autohide.stop()
        self._title.setText(title)
        self._detail.setText(detail)
        self._bar.setRange(0, 0)  # indeterminate until the first progress tick
        self.setVisible(True)
        self.raise_()

    def set_progress(self, done: int, total: int, detail: str) -> None:
        if total > 0:
            if self._bar.maximum() != total:
                self._bar.setRange(0, total)
            self._bar.setValue(max(0, min(done, total)))
        else:
            self._bar.setRange(0, 0)
        self._detail.setText(detail)
        if not self.isVisible():
            self.setVisible(True)

    def complete(self, title: str, detail: str, *, autohide_ms: int = 12000) -> None:
        self._title.setText(title)
        self._detail.setText(detail)
        self._bar.setRange(0, 1)
        self._bar.setValue(1)
        self.setVisible(True)
        self.raise_()
        if autohide_ms > 0:
            self._autohide.start(autohide_ms)

    def hide_banner(self) -> None:
        self._autohide.stop()
        self.setVisible(False)


class NotesAiWorkspace(QWidget):
    """Ollama chat + RAG console scoped to one project (``notes/spaces/<id>`` + ``notes/chats/<id>``)."""

    errorRaised = pyqtSignal(str)
    projectSelected = pyqtSignal(str)
    newProjectClicked = pyqtSignal()
    projectsMetadataChanged = pyqtSignal()
    assistantNameChanged = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("notesAiWorkspace")
        self._assistant_name = assistant_display_name()
        self._space_root: Optional[Path] = None
        self._chat_mgr: Optional[ChatManager] = None
        self._current_chat_id: Optional[str] = None
        self._chat_worker: Optional[QThread] = None
        self._rag_worker: Optional[RAGWorker] = None
        # Separate worker for the global notes RAG so background auto-indexing of
        # uploaded notes never contends with the per-project chat RAG worker.
        self._notes_rag_worker: Optional[RAGWorker] = None
        # Dedicated worker for the global ingested-memory namespace (knowledge
        # transferred from other AIs), kept separate so it never contends with
        # the per-project chat RAG or the notes RAG.
        self._memory_rag_worker: Optional[RAGWorker] = None
        # Background worker for ingesting exported chat history from other AIs.
        self._ai_import_worker: Optional[_AiImportWorker] = None
        self._ai_import_target_is_current: bool = False
        # Wall-clock start of the active long job, for computing the ETA shown in
        # the inline (non-modal) progress strip. The strip itself is built later
        # and stored on ``self._task_progress``.
        self._ai_import_started: float = 0.0
        # Background worker that turns just-ingested chats into RAG memory.
        self._memory_build_worker: Optional[_MemoryBuildWorker] = None
        # True while the active memory build is a user-triggered reindex (force).
        self._memory_build_force: bool = False
        # Live-streaming assistant reply buffer. Tokens arrive via
        # _on_chat_token and accumulate here so the message bubble renders
        # in-place (the previous approach blindly insertPlainText'd into the
        # QTextBrowser, which produced the un-themed "user: hi" wall of text).
        self._streaming_assistant: str = ""
        self._streaming_error: str = ""
        self._streaming_model_label: str = ""
        self._streaming_provider: str = ""
        self._streaming_mcp_enabled: bool = False
        self._is_streaming: bool = False
        # Whether a native system voice is available for read-aloud playback;
        # gates the per-message "Play" action in the transcript.
        self._tts_enabled: bool = text_to_speech_available()
        # Composer-level tool toggles (chip strip above the input). Currently:
        # "web" is a UI-only stub, "rag" mirrors the dedicated RAG button so
        # users can see at a glance what context the next send will carry.
        self._composer_tool_state: dict[str, bool] = {"web": False, "rag": False}
        self._composer_attachments: list[str] = []
        self._project_drop_filter_installed = False
        self._pending_open_chat_id: str = ""
        self._ollama_model_tags: list[str] = []
        self._local_gguf_paths: list[str] = local_gguf_models()
        self._job_state_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._activity_frame_index = 0
        self._activity_timer = QTimer(self)
        self._activity_timer.setInterval(180)
        self._activity_timer.timeout.connect(self._advance_activity_indicators)
        # True when embedded in the small assistant overlay card: scales down the
        # empty-state starters, header, and composer so nothing overflows.
        self._compact_overlay = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, stretch=1)

        self._tabs.addTab(self._build_chat_tab(), "AI Chat")
        self._tabs.addTab(self._build_rag_tab(), "RAG")
        self._tabs.addTab(self._build_ai_settings_tab(), "AI settings")

        # Non-modal progress strip for long imports / reindex jobs. Sits under the
        # tabs and stays hidden until a job runs, so it never steals layout space.
        self._task_progress = _InlineTaskProgress(self)
        outer.addWidget(self._task_progress, stretch=0)

        self._refresh_model_catalog()
        self._project_rows: list[tuple[str, str, str, bool]] = []

    def assistant_name(self) -> str:
        return str(getattr(self, "_assistant_name", "") or "").strip() or DEFAULT_ASSISTANT_NAME

    def _set_assistant_name(self, name: str) -> None:
        chosen = str(name or "").strip() or DEFAULT_ASSISTANT_NAME
        if chosen == self.assistant_name():
            return
        self._assistant_name = chosen
        self.assistantNameChanged.emit(chosen)
        win = self.window()
        btn = getattr(win, "_ai_assistant_btn", None)
        if isinstance(btn, QPushButton):
            btn.setText(chosen)
            btn.setToolTip(f"Open {chosen} for quick CV Ops questions.")
        if hasattr(self, "_settings_assistant_name"):
            self._settings_assistant_name.setText(chosen)
        if hasattr(self, "chat_view"):
            self._show_current_chat()
        if hasattr(self, "_chat_header_title"):
            self._update_chat_header()

    def is_ai_busy(self) -> bool:
        if self._chat_worker is not None and self._chat_worker.isRunning():
            return True
        if self._rag_worker is not None and self._rag_worker.isRunning():
            return True
        if self._notes_rag_worker is not None and self._notes_rag_worker.isRunning():
            return True
        if self._ai_import_worker is not None and self._ai_import_worker.isRunning():
            return True
        return False

    def set_space_root(self, root: Path) -> None:
        """Point RAG at ``root`` (``notes/spaces/<id>``) and chats at sibling ``notes/chats/<id>``."""
        root = root.expanduser().resolve()
        self._space_root = root
        _reset_rag_singleton()
        notes_vault = root.parent.parent
        chats_dir = notes_chats_dir(notes_vault, root.name)
        chats_dir.mkdir(parents=True, exist_ok=True)
        self._chat_mgr = ChatManager(chats_dir=chats_dir)
        self._refresh_chat_list()
        existing = self._chat_mgr.list_chats()
        if existing:
            if self._pending_open_chat_id and any(str(r.get("id") or "") == self._pending_open_chat_id for r in existing):
                self._current_chat_id = self._pending_open_chat_id
                self._pending_open_chat_id = ""
            else:
                self._current_chat_id = str(existing[0]["id"])
        else:
            self._current_chat_id = self._chat_mgr.create_chat("Notes chat")
            self._refresh_chat_list()
        self._select_chat_id(self._current_chat_id or "")
        self._show_current_chat()
        self._refresh_project_workspace(root.name)
        if hasattr(self, "rag_output"):
            self.rag_output.clear()
        self._rag_reset_queue_for_new_space()
        if hasattr(self, "_rag_tree"):
            self._populate_rag_tree()
        self._apply_singularity_chrome()
        self._refresh_model_catalog()

    def set_compact_overlay_mode(self, compact: bool) -> None:
        """Trim navigation chrome when the workspace is embedded in a small overlay."""
        compact = bool(compact)
        self._compact_overlay = compact
        if hasattr(self, "_tabs"):
            self._tabs.setCurrentIndex(0)
            self._tabs.tabBar().setVisible(not compact)
        sidebar = getattr(self, "_chat_sidebar", None)
        if sidebar is not None:
            sidebar.setVisible(not compact)
        ledger = getattr(self, "_events_artifacts_panel", None)
        if ledger is not None:
            ledger.setVisible(not compact)
        timeline = getattr(self, "_chat_timeline_panel", None)
        if timeline is not None:
            timeline.setVisible(not compact)
        splitter = getattr(self, "_chat_splitter", None)
        if splitter is not None:
            splitter.setHandleWidth(0 if compact else 4)
        # Keep the long "Project \ Chat" title to a single line in the narrow card
        # so the header does not wrap to three rows and eat vertical space.
        title = getattr(self, "_chat_header_title", None)
        if title is not None:
            title.setWordWrap(not compact)
        # Shrink the composer action buttons so the bar fits the card width.
        btn_h = 24 if compact else 28
        for attr in ("_btn_compose_rag", "_btn_compose_tools"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setFixedHeight(btn_h)
        for attr in ("_btn_compose_send", "_btn_compose_stop"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setMinimumHeight(btn_h)
        # Re-render so the empty-state starters pick up the compact sizing.
        try:
            self._show_current_chat()
        except Exception:
            pass
        if hasattr(self, "_chat_overlay_host"):
            QTimer.singleShot(0, self._chat_overlay_host._sync_overlay_geometry)

    def current_chat_id(self) -> str:
        return str(self._current_chat_id or "")

    def chat_message_count(self, chat_id: str) -> int:
        if self._chat_mgr is None:
            return 0
        cid = str(chat_id or "").strip()
        if not cid:
            return 0
        return len(self._chat_mgr.get_chat_messages(cid))

    def suggested_chat_title(self, chat_id: str, fallback: str = "Assistant chat") -> str:
        if self._chat_mgr is None:
            return fallback
        cid = str(chat_id or "").strip()
        if not cid:
            return fallback
        for msg in self._chat_mgr.get_chat_messages(cid):
            if str(msg.get("role") or "").strip().lower() != "user":
                continue
            raw = str(msg.get("content") or "").strip()
            raw = raw.split("[workspace context]", 1)[0].split("[file context]", 1)[0]
            first_line = " ".join(raw.split())
            if first_line:
                return first_line[:72]
        return fallback

    def start_scratch_chat(self, title: str = "Scratch assistant question") -> str:
        """Create and select a disposable chat in this workspace's normal chat store."""
        if self._chat_mgr is None:
            return ""
        cid = self._chat_mgr.create_chat(title or "Scratch assistant question")
        chat = self._chat_mgr.chats.get(cid)
        if chat is not None:
            chat["description"] = "Disposable CV Ops assistant question. Use Keep to retain it."
            chat.setdefault("metadata", {})["cvops_scratch"] = True
            self._chat_mgr.save_chat(cid)
        self._current_chat_id = cid
        self._refresh_chat_list()
        self._select_chat_id(cid)
        self._show_current_chat()
        return cid

    def discard_chat_without_prompt(
        self,
        chat_id: str,
        *,
        replacement_title: str = "",
    ) -> str:
        """Delete a chat directly and optionally select a fresh replacement chat."""
        if self._chat_mgr is None:
            return ""
        cid = str(chat_id or "").strip()
        if not cid:
            return ""
        was_current = cid == self._current_chat_id
        self._chat_mgr.delete_chat(cid)
        replacement_id = ""
        if was_current:
            if replacement_title:
                replacement_id = self._chat_mgr.create_chat(replacement_title)
                self._current_chat_id = replacement_id
            else:
                remaining = self._chat_mgr.list_chats()
                if remaining:
                    self._current_chat_id = str(remaining[0].get("id") or "")
                else:
                    self._current_chat_id = self._chat_mgr.create_chat("Notes chat")
        self._refresh_chat_list()
        if self._current_chat_id:
            self._select_chat_id(self._current_chat_id)
        self._show_current_chat()
        return replacement_id or str(self._current_chat_id or "")

    def keep_chat_without_prompt(self, chat_id: str, *, title: str = "") -> bool:
        """Promote a scratch chat so it remains as a normal project chat."""
        if self._chat_mgr is None:
            return False
        cid = str(chat_id or "").strip()
        if not cid:
            return False
        chat = self._chat_mgr.chats.get(cid) or self._chat_mgr.load_chat(cid)
        if chat is None:
            return False
        chosen = (title or "").strip()
        if chosen:
            chat["title"] = chosen
        chat["description"] = "Saved from the CV Ops assistant overlay."
        chat.setdefault("metadata", {})["cvops_scratch"] = False
        self._chat_mgr.save_chat(cid)
        self._refresh_chat_list()
        self._select_chat_id(cid)
        self._show_current_chat()
        return True

    def _workspace_chats_dir(self) -> Optional[Path]:
        root = self._space_root
        if root is None:
            return None
        notes_vault = root.parent.parent
        return notes_chats_dir(notes_vault, root.name)

    def _show_tab_info_sheet(self, title: str, body: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumSize(440, 280)
        outer = QVBoxLayout(dlg)
        txt = QPlainTextEdit(body)
        txt.setReadOnly(True)
        txt.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        outer.addWidget(txt, stretch=1)
        row = QHBoxLayout()
        copy_btn = QPushButton("Copy all")
        copy_btn.setProperty("buttonRole", "secondary")
        repolish(copy_btn)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(body))
        row.addWidget(copy_btn)
        row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setProperty("buttonRole", "secondary")
        repolish(close_btn)
        close_btn.clicked.connect(dlg.accept)
        row.addWidget(close_btn)
        outer.addLayout(row)
        dlg.exec()

    def _info_body_ai_chat(self) -> str:
        lines = [
            "AI Chat",
            "",
            "Pick a provider from the model field on the composer bar. Ollama uses the base URL on this tab.",
            "",
        ]
        cd = self._workspace_chats_dir()
        if cd is not None:
            lines.extend(
                [
                    "Chats directory",
                    str(cd.resolve()),
                    "",
                    "Each chat is stored under this folder.",
                ]
            )
        else:
            lines.append("Select a notes project to attach chat storage to a space.")
        if hasattr(self, "chat_ollama_url"):
            lines.extend(["", "Ollama base URL (this tab)", self.chat_ollama_url.text().strip() or "(default)"])
        return "\n".join(lines)

    def _info_body_rag(self) -> str:
        lines: list[str] = ["RAG (document queue + index + query)", ""]
        root = self._space_root
        if root is None:
            lines.append("Select a notes project to attach the RAG index to a space.")
            return "\n".join(lines)
        ridx = root / "rag_index"
        lines.extend(
            [
                "Project space (files, sessions, captures, …)",
                str(root.resolve()),
                "",
                "Vector index directory",
                str(ridx.resolve()),
                "",
                "Queue documents on this tab, then build the index. Embedding backend and Ollama URL are configured above the queue.",
            ]
        )
        if not _rag_dependencies_available():
            lines.extend(
                [
                    "",
                    "RAG Python dependencies are not installed. Install from:",
                    "mlops/ChatbotAndRag/solo_rag_chat/requirements-solo.txt",
                ]
            )
        return "\n".join(lines)

    def _info_body_ai_settings(self) -> str:
        p = ai_settings_path()
        return "\n".join(
            [
                "AI provider keys",
                "",
                "Settings file (plaintext JSON on this computer):",
                str(p.resolve()),
                "",
                "Providers with a non-empty saved key appear in the AI Chat model list.",
                "Keys are only sent to the provider you select when you send a message.",
                "",
                "The optional system prompt is prepended to every model (local and cloud)",
                "as standing instructions for tone, persona, and formatting.",
            ]
        )

    def _on_info_ai_chat(self) -> None:
        self._show_tab_info_sheet("AI Chat — info", self._info_body_ai_chat())

    def _on_info_rag(self) -> None:
        self._show_tab_info_sheet("RAG — info", self._info_body_rag())

    def _on_info_ai_settings(self) -> None:
        self._show_tab_info_sheet("AI settings — info", self._info_body_ai_settings())

    def _prepend_tab_info_row(self, layout: QVBoxLayout, on_clicked: Callable[[], None]) -> None:
        # Info buttons removed: they were a waste of header space. Kept as a
        # no-op so every existing caller stays valid without edits.
        return

    @staticmethod
    def _repopulate_editable_ollama_combo(box: QComboBox, tags: list[str]) -> None:
        prev = box.currentText().strip()
        box.blockSignals(True)
        box.clear()
        for t in tags:
            s = str(t).strip()
            if s:
                box.addItem(s)
        if prev:
            ix = box.findText(prev, Qt.MatchFlag.MatchExactly)
            if ix >= 0:
                box.setCurrentIndex(ix)
            else:
                box.setEditText(prev)
        elif box.count() > 0:
            box.setCurrentIndex(0)
        box.blockSignals(False)

    @staticmethod
    def _set_editable_combo_text(box: QComboBox, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            return
        box.blockSignals(True)
        ix = box.findText(value, Qt.MatchFlag.MatchExactly)
        if ix >= 0:
            box.setCurrentIndex(ix)
        else:
            box.setEditText(value)
        box.blockSignals(False)

    def _rag_sync_ollama_model_lists(self) -> None:
        """Fill RAG answer / embedding combos from ``ollama list`` and ``/api/tags`` for the RAG base URL."""
        if not hasattr(self, "rag_chat_model"):
            return
        url = self.rag_ollama_url.text().strip()
        installed_tags = discover_ollama_model_tags(base_url=url or None)
        self._rag_ollama_tags = list(installed_tags)
        self._rag_ollama_tags_url = url
        tags = installed_tags or list(OLLAMA_DEFAULT_MODELS)
        self._repopulate_editable_ollama_combo(self.rag_chat_model, tags)
        self._repopulate_editable_ollama_combo(self.rag_embed_model, tags)
        selected_embed = self._resolve_rag_embedding_model(
            self.rag_embed_backend.currentText().strip().lower(),
            self.rag_embed_model.currentText().strip(),
            url,
            update_combo=True,
            announce=True,
        )
        if selected_embed:
            self._set_editable_combo_text(self.rag_embed_model, selected_embed)
        if hasattr(self, "rag_output"):
            self.rag_output.append(
                f"[RAG] Ollama model list: {len(tags)} tag(s) (CLI list + HTTP /api/tags) for "
                f"{url or 'default host'}."
            )
        self._update_rag_engine_summary()

    def _rag_known_ollama_tags(self, base_url: str) -> list[str]:
        url = str(base_url or "").strip()
        cached_url = str(getattr(self, "_rag_ollama_tags_url", "") or "")
        cached = list(getattr(self, "_rag_ollama_tags", []) or [])
        if cached and cached_url == url:
            return cached
        tags = discover_ollama_model_tags(base_url=url or None)
        self._rag_ollama_tags = list(tags)
        self._rag_ollama_tags_url = url
        return tags

    def _resolve_rag_embedding_model(
        self,
        embedding_backend: str,
        embedding_model: str,
        ollama_base_url: str,
        *,
        update_combo: bool = False,
        announce: bool = False,
    ) -> str:
        backend = str(embedding_backend or "").strip().lower()
        current = str(embedding_model or "").strip()
        if backend != "ollama":
            return current
        tags = self._rag_known_ollama_tags(ollama_base_url)
        resolved = choose_ollama_embedding_model(tags, current=current)
        if tags and resolved and resolved != current:
            if update_combo and hasattr(self, "rag_embed_model"):
                self._set_editable_combo_text(self.rag_embed_model, resolved)
            if announce and hasattr(self, "rag_output"):
                self.rag_output.append(
                    f"[RAG] Embedding model switched to installed Ollama tag: {resolved}."
                )
        return resolved or current

    def _refresh_model_catalog(self) -> None:
        if not hasattr(self, "chat_model"):
            return
        settings = load_ai_settings()
        base = self.chat_ollama_url.text().strip() if hasattr(self, "chat_ollama_url") else ""
        ollama_tags = discover_ollama_model_tags(base_url=base or None)
        ggufs = local_gguf_models(settings)
        self._ollama_model_tags = list(ollama_tags)
        self._local_gguf_paths = list(ggufs)
        rows = model_catalog_entries(
            settings,
            ollama_installed=ollama_tags,
            local_gguf_installed=ggufs,
        )
        prev_route = self.chat_model.currentData()
        prev_text = self.chat_model.currentText().strip()
        self.chat_model.blockSignals(True)
        self.chat_model.clear()
        for label, route in rows:
            self.chat_model.addItem(label, route)
        ix_restored = -1
        if prev_route is not None:
            pr = str(prev_route)
            for i in range(self.chat_model.count()):
                d = self.chat_model.itemData(i, Qt.ItemDataRole.UserRole)
                if d is not None and str(d) == pr:
                    ix_restored = i
                    break
        if ix_restored >= 0:
            self.chat_model.setCurrentIndex(ix_restored)
        elif prev_text:
            self.chat_model.setEditText(prev_text)
        elif self.chat_model.count() > 0:
            self.chat_model.setCurrentIndex(0)
        self.chat_model.blockSignals(False)
        self._refresh_on_device_models_list()

    def _build_ai_settings_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self._prepend_tab_info_row(layout, self._on_info_ai_settings)
        form = QFormLayout()
        self._settings_assistant_name = QLineEdit()
        self._settings_assistant_name.setPlaceholderText(DEFAULT_ASSISTANT_NAME)
        form.addRow("Assistant name", self._settings_assistant_name)
        self._settings_key_openai = QLineEdit()
        self._settings_key_openai.setEchoMode(QLineEdit.EchoMode.Password)
        self._settings_key_openai.setPlaceholderText("sk-…")
        form.addRow("OpenAI API key", self._settings_key_openai)
        self._settings_key_anthropic = QLineEdit()
        self._settings_key_anthropic.setEchoMode(QLineEdit.EchoMode.Password)
        self._settings_key_anthropic.setPlaceholderText("sk-ant-…")
        form.addRow("Anthropic API key", self._settings_key_anthropic)
        self._settings_key_grok = QLineEdit()
        self._settings_key_grok.setEchoMode(QLineEdit.EchoMode.Password)
        self._settings_key_grok.setPlaceholderText("xai-…")
        form.addRow("Grok (xAI) API key", self._settings_key_grok)
        self._settings_key_gemini = QLineEdit()
        self._settings_key_gemini.setEchoMode(QLineEdit.EchoMode.Password)
        self._settings_key_gemini.setPlaceholderText("AIza…")
        form.addRow("Gemini API key", self._settings_key_gemini)

        # Tell the operator exactly where their keys land so nothing is a surprise.
        if keyring_available():
            key_storage_note = (
                "Keys are stored in your operating system keyring "
                "(macOS Keychain / Windows Credential Locker / Linux Secret Service), "
                "not in plaintext on disk."
            )
        else:
            key_storage_note = (
                "[NO OS KEYRING DETECTED] Keys will be saved in plaintext at "
                f"{ai_settings_path()}. Install the 'keyring' package and restart to "
                "store them in your OS keyring instead."
            )
        key_storage_caption = QLabel(key_storage_note)
        key_storage_caption.setWordWrap(True)
        key_storage_caption.setObjectName("notesKeyStorageCaption")
        form.addRow("", key_storage_caption)

        # Global system prompt: prepended to every model/provider as a system
        # message so the operator can set the assistant's standing instructions.
        self._settings_system_prompt = QPlainTextEdit()
        self._settings_system_prompt.setObjectName("notesSystemPromptEdit")
        self._settings_system_prompt.setPlaceholderText(
            "Optional. Standing instructions sent to every model as a system prompt "
            "(persona, tone, formatting rules). Leave blank for none."
        )
        self._settings_system_prompt.setMinimumHeight(120)
        self._settings_system_prompt.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        form.addRow("System prompt", self._settings_system_prompt)
        layout.addLayout(form)
        save_btn = QPushButton("Save keys and refresh model catalog")
        save_btn.clicked.connect(self._on_save_ai_settings)
        layout.addWidget(save_btn)
        layout.addWidget(self._build_local_model_section())
        layout.addWidget(self._build_voice_section())
        layout.addWidget(self._build_ai_import_section())
        layout.addStretch(1)
        self._load_ai_settings_form()
        self._refresh_on_device_models_list()
        self._load_voice_form()
        return w

    def _build_local_model_section(self) -> QWidget:
        section = QFrame()
        section.setObjectName("notesLocalModelsSection")
        col = QVBoxLayout(section)
        col.setContentsMargins(0, 12, 0, 0)
        col.setSpacing(6)
        title = QLabel("On-device models")
        title.setProperty("isTitle", True)
        repolish(title)
        col.addWidget(title)

        self._local_models_status = QLabel("")
        self._local_models_status.setProperty("muted", True)
        self._local_models_status.setWordWrap(True)
        repolish(self._local_models_status)
        col.addWidget(self._local_models_status)

        self._local_models_list = _GgufModelListWidget()
        self._local_models_list.setMinimumHeight(120)
        self._local_models_list.setToolTip("Drop .gguf files or folders here to register local model paths.")
        self._local_models_list.pathsDropped.connect(self._on_local_gguf_paths_dropped)
        col.addWidget(self._local_models_list)

        row = QHBoxLayout()
        add_btn = QPushButton("Add GGUF")
        add_btn.clicked.connect(self._pick_local_gguf_models)
        row.addWidget(add_btn)
        scan_btn = QPushButton("Rescan device")
        scan_btn.clicked.connect(self._scan_local_gguf_models)
        row.addWidget(scan_btn)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_local_gguf_models)
        row.addWidget(remove_btn)
        row.addStretch(1)
        col.addLayout(row)
        return section

    # ----------------------------------------------------------------- #
    # Voice maker: design the assistant's spoken voice (base voice + a
    # restrained ffmpeg effect chain), preview it live, and persist the
    # active profile to ai_settings.json. See notes_ai_keys.voice_profile.
    # ----------------------------------------------------------------- #

    # (slider key, label, min, max, scale) — scale maps slider int <-> profile
    # float so we can use plain QSliders. e.g. pitch -120..120 / 10 = -12..12.
    # Core character sliders, shown first.
    _VOICE_SLIDERS = (
        ("rate_wpm", "Pace (words/min)", 90, 320, 1.0),
        ("pitch_semitones", "Pitch (semitones)", -120, 120, 10.0),
        ("warmth_db", "Warmth (dB)", 0, 120, 10.0),
        ("high_cut_hz", "Soften highs (Hz, 0=off)", 0, 16000, 1.0),
        ("room", "Room (0=dry)", 0, 100, 100.0),
    )
    # Fine-tuning sliders for naturalness / smoothing, shown under a sub-heading.
    _VOICE_FINE_SLIDERS = (
        ("low_cut_hz", "Low cut (Hz, 0=off)", 0, 400, 1.0),
        ("presence_db", "Clarity (dB)", -120, 120, 10.0),
        ("air_db", "Air / breath (dB)", 0, 120, 10.0),
        ("sibilance", "De-ess (tame S)", 0, 100, 100.0),
        ("smoothing", "Leveling (even delivery)", 0, 100, 100.0),
        ("depth", "Depth / richness", 0, 100, 100.0),
    )

    @classmethod
    def _all_voice_specs(cls):
        """Every (key, label, min, max, scale) across core + fine sliders."""
        return (*cls._VOICE_SLIDERS, *cls._VOICE_FINE_SLIDERS)

    def _build_voice_section(self) -> QWidget:
        section = QFrame()
        section.setObjectName("notesVoiceSection")
        col = QVBoxLayout(section)
        col.setContentsMargins(0, 12, 0, 0)
        col.setSpacing(6)

        title = QLabel("Assistant voice")
        title.setProperty("isTitle", True)
        repolish(title)
        col.addWidget(title)

        blurb = QLabel(
            "Design the voice your assistant speaks with: pick a base system "
            "voice and shape it with subtle effects. Defaults to a calm, "
            "measured “Tacitus” profile."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("QLabel { color: rgba(255,255,255,0.62); }")
        col.addWidget(blurb)

        if not self._tts_enabled:
            warn = QLabel(
                "[NO VOICE] No system text-to-speech engine was found, so the "
                "voice cannot be previewed on this machine."
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("QLabel { color: rgba(255,200,90,0.85); }")
            col.addWidget(warn)

        form = QFormLayout()
        # True while we programmatically set widgets so change-signals don't
        # spuriously flip the preset to "Custom".
        self._voice_loading = False

        self._voice_preset_combo = QComboBox()
        self._voice_preset_combo.addItems([*VOICE_PRESETS.keys(), "Custom"])
        self._voice_preset_combo.currentTextChanged.connect(self._on_voice_preset_changed)
        form.addRow("Preset", self._voice_preset_combo)

        self._voice_base_combo = QComboBox()
        self._voice_base_combo.addItem("System default", "")
        for name, locale in list_system_voices():
            self._voice_base_combo.addItem(f"{name}  ({locale})", name)
        self._voice_base_combo.currentIndexChanged.connect(self._on_voice_edited)
        form.addRow("Base voice", self._voice_base_combo)

        self._voice_sliders: dict[str, QSlider] = {}
        self._voice_value_labels: dict[str, QLabel] = {}

        def _add_slider(key: str, label: str, lo: int, hi: int) -> None:
            row = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.valueChanged.connect(lambda _v, k=key: self._on_voice_slider(k))
            value_lbl = QLabel("")
            value_lbl.setMinimumWidth(56)
            value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_lbl.setStyleSheet("QLabel { color: rgba(255,255,255,0.62); }")
            row.addWidget(slider, 1)
            row.addWidget(value_lbl, 0)
            self._voice_sliders[key] = slider
            self._voice_value_labels[key] = value_lbl
            form.addRow(label, row)

        for key, label, lo, hi, _scale in self._VOICE_SLIDERS:
            _add_slider(key, label, lo, hi)

        fine_heading = QLabel("Fine tuning — smoothing & naturalness")
        fine_heading.setStyleSheet(
            "QLabel { color: rgba(255,255,255,0.5); font-weight: 600; padding-top: 4px; }"
        )
        form.addRow(fine_heading)
        for key, label, lo, hi, _scale in self._VOICE_FINE_SLIDERS:
            _add_slider(key, label, lo, hi)

        self._voice_comms_check = QCheckBox("Radio / helmet “comms” band-pass")
        self._voice_comms_check.toggled.connect(self._on_voice_edited)
        form.addRow("", self._voice_comms_check)
        col.addLayout(form)

        row = QHBoxLayout()
        test_btn = QPushButton("Test voice")
        test_btn.setToolTip("Speak a short sample using the current settings.")
        test_btn.clicked.connect(self._on_test_voice)
        test_btn.setEnabled(self._tts_enabled)
        row.addWidget(test_btn)
        save_btn = QPushButton("Save voice")
        save_btn.clicked.connect(self._on_save_voice_profile)
        row.addWidget(save_btn)
        row.addStretch(1)
        col.addLayout(row)
        return section

    def _voice_profile_from_form(self) -> dict:
        """Read the current designer widgets into a voice-profile dict."""
        profile = default_voice_profile()
        profile["name"] = self._voice_preset_combo.currentText() or "Custom"
        profile["base_voice"] = self._voice_base_combo.currentData() or ""
        for key, _label, _lo, _hi, scale in self._all_voice_specs():
            profile[key] = self._voice_sliders[key].value() / scale
        profile["comms_bandpass"] = self._voice_comms_check.isChecked()
        return profile

    def _apply_voice_profile_to_form(self, profile: dict) -> None:
        """Push a profile dict into the designer widgets (no change signals)."""
        self._voice_loading = True
        try:
            base = str(profile.get("base_voice") or "")
            idx = self._voice_base_combo.findData(base)
            self._voice_base_combo.setCurrentIndex(idx if idx >= 0 else 0)
            for key, _label, _lo, _hi, scale in self._all_voice_specs():
                self._voice_sliders[key].setValue(int(round(float(profile.get(key, 0)) * scale)))
            self._voice_comms_check.setChecked(bool(profile.get("comms_bandpass")))
            self._refresh_voice_value_labels()
        finally:
            self._voice_loading = False

    def _refresh_voice_value_labels(self) -> None:
        for key, _label, _lo, _hi, scale in self._all_voice_specs():
            val = self._voice_sliders[key].value() / scale
            if key.endswith("_hz"):
                text = "off" if val < 1 else f"{int(val)}"
            elif key == "rate_wpm":
                text = f"{int(val)}"
            elif key.endswith("_db"):
                text = f"{val:+.1f}"
            elif key in ("room", "sibilance", "smoothing", "depth"):
                text = "off" if val < 0.01 else f"{val:.2f}"
            else:
                text = f"{val:+.1f}" if key == "pitch_semitones" else f"{val:.1f}"
            self._voice_value_labels[key].setText(text)

    def _load_voice_form(self) -> None:
        if not hasattr(self, "_voice_preset_combo"):
            return
        profile = voice_profile(load_ai_settings())
        name = str(profile.get("name") or "")
        self._voice_loading = True
        try:
            self._voice_preset_combo.setCurrentText(name if name in VOICE_PRESETS else "Custom")
        finally:
            self._voice_loading = False
        self._apply_voice_profile_to_form(profile)
        if hasattr(self, "_tts_bar"):
            self._tts_bar.set_voice_profile(profile)

    def _on_voice_preset_changed(self, name: str) -> None:
        if self._voice_loading or name not in VOICE_PRESETS:
            return
        self._apply_voice_profile_to_form(VOICE_PRESETS[name])

    def _on_voice_slider(self, _key: str) -> None:
        self._refresh_voice_value_labels()
        self._on_voice_edited()

    def _on_voice_edited(self, *_args) -> None:
        # Any manual edit means the profile no longer matches a named preset.
        if self._voice_loading:
            return
        if self._voice_preset_combo.currentText() != "Custom":
            self._voice_loading = True
            try:
                self._voice_preset_combo.setCurrentText("Custom")
            finally:
                self._voice_loading = False

    def _on_test_voice(self) -> None:
        if not hasattr(self, "_tts_bar"):
            return
        profile = self._voice_profile_from_form()
        self._tts_bar.set_voice_profile(profile)
        self._tts_bar.speak(
            "This is how I will sound. Calm, measured, and considered."
        )

    def _on_save_voice_profile(self) -> None:
        profile = self._voice_profile_from_form()
        settings = load_ai_settings()
        settings[KEY_VOICE_PROFILE] = profile
        save_ai_settings(settings)
        if hasattr(self, "_tts_bar"):
            self._tts_bar.set_voice_profile(voice_profile(settings))
        QMessageBox.information(
            self, "Assistant voice", "Voice saved. The assistant will speak with it from now on."
        )

    @staticmethod
    def _normalize_existing_gguf_paths(paths: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in paths:
            try:
                p = Path(str(raw or "").strip()).expanduser().resolve()
            except (OSError, ValueError):
                continue
            if not (p.is_file() and p.suffix.lower() == ".gguf"):
                continue
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _save_local_gguf_models(self, paths: list[str]) -> None:
        settings = load_ai_settings()
        settings[KEY_LOCAL_GGUF_MODELS] = self._normalize_existing_gguf_paths(paths)
        save_ai_settings(settings)
        self._local_gguf_paths = local_gguf_models(settings)
        self._refresh_model_catalog()

    def _on_local_gguf_paths_dropped(self, paths: list[str]) -> None:
        incoming = self._normalize_existing_gguf_paths(paths)
        if not incoming:
            if hasattr(self, "_local_models_status"):
                self._local_models_status.setText("No .gguf model files found in the drop.")
            return
        existing = local_gguf_models(load_ai_settings())
        merged = list(dict.fromkeys([*existing, *incoming]))
        self._save_local_gguf_models(merged)
        if hasattr(self, "_local_models_status"):
            self._local_models_status.setText(f"Registered {len(incoming)} local GGUF model path(s).")

    def _pick_local_gguf_models(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add GGUF models",
            str(Path.home()),
            "GGUF models (*.gguf);;All files (*)",
        )
        if files:
            self._on_local_gguf_paths_dropped(files)

    def _scan_local_gguf_models(self) -> None:
        try:
            found = discover_local_gguf_files(repo_root=Path(ROOT_DIR), max_files=500)
        except Exception as exc:
            if hasattr(self, "_local_models_status"):
                self._local_models_status.setText(f"Model scan failed: {exc}")
            return
        existing = local_gguf_models(load_ai_settings())
        merged = list(dict.fromkeys([*existing, *found]))
        self._save_local_gguf_models(merged)
        if hasattr(self, "_local_models_status"):
            self._local_models_status.setText(f"Found {len(found)} GGUF file(s) on device.")

    def _remove_selected_local_gguf_models(self) -> None:
        if not hasattr(self, "_local_models_list"):
            return
        remove: set[str] = set()
        for item in self._local_models_list.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(data, dict) and data.get("kind") == "gguf":
                remove.add(str(data.get("path") or ""))
        if not remove:
            return
        kept = [p for p in local_gguf_models(load_ai_settings()) if p not in remove]
        self._save_local_gguf_models(kept)
        if hasattr(self, "_local_models_status"):
            self._local_models_status.setText(f"Removed {len(remove)} local GGUF model path(s).")

    def _refresh_on_device_models_list(self) -> None:
        if not hasattr(self, "_local_models_list"):
            return
        self._local_models_list.clear()
        ollama_tags = list(getattr(self, "_ollama_model_tags", []) or [])
        gguf_paths = list(getattr(self, "_local_gguf_paths", []) or [])
        for tag in ollama_tags:
            item = QListWidgetItem(f"Ollama  {tag}")
            item.setData(Qt.ItemDataRole.UserRole, {"kind": "ollama", "tag": tag})
            self._local_models_list.addItem(item)
        for path in gguf_paths:
            item = QListWidgetItem(f"GGUF    {Path(path).name}")
            item.setToolTip(path)
            item.setData(Qt.ItemDataRole.UserRole, {"kind": "gguf", "path": path})
            self._local_models_list.addItem(item)
        if not ollama_tags and not gguf_paths:
            item = QListWidgetItem("No local models listed yet. Drop .gguf files or rescan.")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._local_models_list.addItem(item)
        if hasattr(self, "_local_models_status"):
            self._local_models_status.setText(
                f"{len(ollama_tags)} Ollama tag(s), {len(gguf_paths)} GGUF path(s)."
            )

    def _build_ai_import_section(self) -> QWidget:
        """Ingest exported chat history from other AIs (ChatGPT / Claude).

        The chats land in the *currently selected* notes project, back-dated to
        their original timestamps so a scientific journey spread across several
        assistants reads as one continuous timeline.
        """
        section = QFrame()
        section.setObjectName("notesAiImportSection")
        col = QVBoxLayout(section)
        col.setContentsMargins(0, 12, 0, 0)
        col.setSpacing(6)

        heading = QLabel("Ingest chats from other AIs")
        heading.setObjectName("notesAiImportHeading")
        heading.setStyleSheet("QLabel { font-weight: 600; }")
        col.addWidget(heading)

        blurb = QLabel(
            "Drop in your exported Claude or ChatGPT data (the conversations.json "
            "or the .zip you downloaded). Conversations are added to the selected "
            "notes project and back-dated to when they actually happened."
        )
        blurb.setWordWrap(True)
        blurb.setStyleSheet("QLabel { color: rgba(255,255,255,0.62); }")
        col.addWidget(blurb)

        row = QHBoxLayout()
        row.setSpacing(8)
        self._btn_import_auto = QPushButton("Choose export file or folder…")
        self._btn_import_auto.clicked.connect(lambda: self._on_import_ai_chats(None))
        row.addWidget(self._btn_import_auto)
        self._btn_import_chatgpt = QPushButton("ChatGPT…")
        self._btn_import_chatgpt.setProperty("buttonRole", "secondary")
        self._btn_import_chatgpt.clicked.connect(
            lambda: self._on_import_ai_chats("chatgpt")
        )
        row.addWidget(self._btn_import_chatgpt)
        self._btn_import_claude = QPushButton("Claude…")
        self._btn_import_claude.setProperty("buttonRole", "secondary")
        self._btn_import_claude.clicked.connect(
            lambda: self._on_import_ai_chats("claude")
        )
        row.addWidget(self._btn_import_claude)
        self._btn_import_claude_projects = QPushButton("Claude projects…")
        self._btn_import_claude_projects.setProperty("buttonRole", "secondary")
        self._btn_import_claude_projects.setToolTip(
            "Respawn Claude projects as notes projects: same name, instructions, "
            "knowledge files, and their conversations (back-dated)."
        )
        self._btn_import_claude_projects.clicked.connect(self._on_import_claude_projects)
        row.addWidget(self._btn_import_claude_projects)
        row.addStretch(1)
        col.addLayout(row)

        manage_row = QHBoxLayout()
        manage_row.setSpacing(8)
        self._btn_manage_imported = QPushButton("Manage ingested data…")
        self._btn_manage_imported.setProperty("buttonRole", "secondary")
        self._btn_manage_imported.setToolTip(
            "See and cleanse ingested chats/projects so the interface shows only "
            "what was created on this baseline system."
        )
        self._btn_manage_imported.clicked.connect(self._on_manage_imported)
        manage_row.addWidget(self._btn_manage_imported)
        manage_row.addStretch(1)
        col.addLayout(manage_row)

        for b in (
            self._btn_import_auto,
            self._btn_import_chatgpt,
            self._btn_import_claude,
            self._btn_import_claude_projects,
            self._btn_manage_imported,
        ):
            repolish(b)
        return section

    def _import_target_choices(self) -> list[tuple[str, str, Path]]:
        """``(space_id, title, chats_dir)`` for every notes project, current first."""
        spaces_root = self._spaces_root()
        if spaces_root is None or self._space_root is None:
            return []
        vault = self._space_root.parent.parent
        current_sid = self._space_root.name
        ordered: list[str] = [current_sid]
        for sid in list_space_ids(spaces_root):
            if sid not in ordered:
                ordered.append(sid)
        out: list[tuple[str, str, Path]] = []
        for sid in ordered:
            out.append((sid, self._project_title_for_space(sid), notes_chats_dir(vault, sid)))
        return out

    def _on_import_ai_chats(self, source: Optional[str]) -> None:
        """Pick an export (+ target project) and ingest its conversations off-thread."""
        if self._ai_import_worker is not None and self._ai_import_worker.isRunning():
            QMessageBox.information(
                self, "Ingest chats", "An import is already running — let it finish first."
            )
            return

        current_chats_dir = self._workspace_chats_dir()
        if current_chats_dir is None:
            QMessageBox.information(
                self,
                "Ingest chats",
                "Open a notes project first — imported conversations are added to "
                "a project's chat history.",
            )
            return

        label = {"chatgpt": "ChatGPT", "claude": "Claude"}.get(source or "", "AI")
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose {label} export (conversations.json or .zip)",
            "",
            "Chat exports (*.json *.zip);;All files (*)",
        )
        if not path_str:
            # Fall back to a folder picker so a user can point at an unzipped export dir.
            path_str = QFileDialog.getExistingDirectory(self, f"Choose {label} export folder")
        if not path_str:
            return

        # Optional target-project picker: default to the current project, but let
        # the user redirect the import to any other notes project at import time.
        target_dir = current_chats_dir
        target_title = self._project_title_for_space(self._space_root.name)
        target_is_current = True
        choices = self._import_target_choices()
        if len(choices) > 1:
            titles = [f"{title}  ({sid})" for sid, title, _ in choices]
            picked, ok = QInputDialog.getItem(
                self,
                "Ingest chats — target project",
                "Add the imported conversations to:",
                titles,
                0,
                False,
            )
            if not ok:
                return
            idx = titles.index(picked) if picked in titles else 0
            sid, target_title, target_dir = choices[idx]
            target_is_current = sid == self._space_root.name

        if not self._confirm_ingest_size(Path(path_str), target_dir, target_title):
            return

        self._ai_import_target_is_current = target_is_current
        self._set_ai_import_buttons_enabled(False)
        worker = _AiImportWorker(Path(path_str), target_dir, source)
        worker.progress.connect(self._on_ai_import_progress)
        worker.finished.connect(self._on_ai_import_finished)
        worker.failed.connect(self._on_ai_import_failed)
        self._ai_import_worker = worker
        self._begin_ai_import_progress()
        worker.start()

    def _on_import_claude_projects(self) -> None:
        """Respawn Claude projects (instructions + knowledge + chats) as notes projects."""
        if self._ai_import_worker is not None and self._ai_import_worker.isRunning():
            QMessageBox.information(
                self, "Ingest chats", "An import is already running — let it finish first."
            )
            return
        spaces_root = self._spaces_root()
        if spaces_root is None or self._space_root is None:
            QMessageBox.information(
                self,
                "Ingest Claude projects",
                "Open a notes project first so the importer knows which vault to "
                "respawn the projects into.",
            )
            return

        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Claude data export (.zip or folder with projects.json)",
            "",
            "Claude export (*.zip *.json);;All files (*)",
        )
        if not path_str:
            path_str = QFileDialog.getExistingDirectory(self, "Choose Claude export folder")
        if not path_str:
            return

        vault = self._space_root.parent.parent
        fallback_dir = self._workspace_chats_dir()
        if not self._confirm_ingest_size(
            Path(path_str),
            spaces_root,
            "your notes vault",
            member_names=["conversations.json", "projects.json"],
            intro=(
                "This will respawn your Claude projects here — recreating each "
                "project's name, instructions, knowledge files, and conversations "
                "(back-dated). It will ingest the data below into your notes vault."
            ),
        ):
            return

        # Project respawn touches the spaces sidebar; the open chat manager only
        # needs a reload for unlinked chats that land in the current project.
        self._ai_import_target_is_current = True
        self._set_ai_import_buttons_enabled(False)
        worker = _AiImportWorker(
            Path(path_str),
            fallback_dir if fallback_dir is not None else (vault / "chats" / self._space_root.name),
            "claude",
            mode="claude_projects",
            spaces_root=spaces_root,
            vault_root=vault,
        )
        worker.progress.connect(self._on_ai_import_progress)
        worker.finished.connect(self._on_ai_import_finished)
        worker.failed.connect(self._on_ai_import_failed)
        self._ai_import_worker = worker
        self._begin_ai_import_progress()
        worker.start()

    def _on_manage_imported(self) -> None:
        """Inventory + cleanse ingested chats/projects (keep only baseline data)."""
        from . import notes_ai_import as ai_import

        spaces_root = self._spaces_root()
        if spaces_root is None or self._space_root is None:
            QMessageBox.information(
                self, "Manage ingested data", "Open a notes project first."
            )
            return
        vault = self._space_root.parent.parent
        inv = ai_import.scan_imported(spaces_root, vault)
        if inv.is_empty():
            QMessageBox.information(
                self,
                "Manage ingested data",
                "No ingested chats or projects found — everything here was created "
                "on this system.",
            )
            return

        def _src_label(s: str) -> str:
            return {"chatgpt": "ChatGPT", "claude": "Claude"}.get(s, s or "unknown")

        dlg = QDialog(self)
        dlg.setWindowTitle("Manage ingested data")
        dlg.setMinimumWidth(460)
        lay = QVBoxLayout(dlg)

        summary = [f"Ingested chats: {inv.total_chats}"]
        for src, n in sorted(inv.chats_by_source.items()):
            summary.append(f"   • {_src_label(src)}: {n}")
        if inv.total_projects:
            summary.append(f"Respawned projects: {inv.total_projects}")
            for src, n in sorted(inv.projects_by_source.items()):
                summary.append(f"   • {_src_label(src)}: {n}")
        mem_total = ai_memory.memory_doc_count(vault)
        if mem_total:
            summary.append(f"Transferred memory docs: {mem_total}")
        head = QLabel("\n".join(summary))
        head.setWordWrap(True)
        lay.addWidget(head)

        note = QLabel(
            "Cleansing removes only ingested items. Chats and projects you created "
            "on this baseline system are never touched. Removing projects also "
            "deletes their instructions, knowledge files, and chat history.\n\n"
            "When you remove ingested chats you can choose to keep the memory "
            "Tacitus learned from them — so the knowledge survives without storing "
            "the raw conversations."
        )
        note.setWordWrap(True)
        note.setStyleSheet("QLabel { color: rgba(255,255,255,0.62); }")
        lay.addWidget(note)

        # Reindex: rebuild transferable memory from chats already on disk (e.g.
        # ingested before this corpus existed, or after changing the model).
        def _run_reindex() -> None:
            dlg.accept()
            self._start_memory_build(source="all", export_path=None, force=True)

        reindex_btn = QPushButton(f"Reindex memory from chats ({inv.total_chats})")
        reindex_btn.setToolTip(
            "Regenerate transcripts/summaries for every ingested chat and rebuild "
            "the memory index. Runs in batches with an estimated time remaining."
        )
        reindex_btn.clicked.connect(_run_reindex)
        lay.addWidget(reindex_btn)

        # Buttons: per-source chat cleanse, all chats, and (destructive) projects.
        def _run_delete_chats(source: Optional[str]) -> None:
            label = "all ingested chats" if source is None else f"{_src_label(source)} ingested chats"
            if QMessageBox.question(
                dlg, "Confirm cleanse", f"Permanently remove {label}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return

            # Decouple chat storage from transferred knowledge: when memory exists
            # for these chats, ask whether to keep it (Tacitus retains the
            # knowledge) or purge it along with the chats.
            drop_memory = False
            if ai_memory.memory_doc_count(vault, source=source):
                box = QMessageBox(dlg)
                box.setWindowTitle("Keep transferred memory?")
                box.setIcon(QMessageBox.Icon.Question)
                box.setText(
                    "Tacitus has memory learned from these chats. Keep that memory "
                    "(recommended) so the knowledge survives, or delete it too?"
                )
                keep_btn = box.addButton("Keep memory", QMessageBox.ButtonRole.AcceptRole)
                drop_btn = box.addButton("Delete memory too", QMessageBox.ButtonRole.DestructiveRole)
                cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
                box.setDefaultButton(keep_btn)
                box.exec()
                clicked = box.clickedButton()
                if clicked is cancel_btn:
                    return
                drop_memory = clicked is drop_btn

            n = ai_import.delete_imported_chats(spaces_root, vault, source=source)
            if drop_memory:
                removed = ai_memory.delete_memory_docs(vault, source=source)
                self._rebuild_ingested_memory_index()
                self._after_cleanse(
                    dlg, f"Removed {n} ingested chat(s) and {removed} memory doc(s)."
                )
            else:
                ai_memory.mark_chats_deleted(vault, source=source)
                self._after_cleanse(
                    dlg, f"Removed {n} ingested chat(s). Transferred memory kept."
                )

        def _run_delete_projects() -> None:
            if QMessageBox.warning(
                dlg, "Confirm project removal",
                f"Permanently delete {inv.total_projects} respawned project(s), "
                "including their instructions, knowledge files, and chats?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ) != QMessageBox.StandardButton.Yes:
                return
            n = ai_import.delete_imported_projects(spaces_root, vault)
            # Imported projects own ingested chats too; refresh both views.
            self._after_cleanse(dlg, f"Removed {n} respawned project(s).", projects=True)

        for src in sorted(inv.chats_by_source):
            b = QPushButton(f"Remove {_src_label(src)} chats ({inv.chats_by_source[src]})")
            b.setProperty("buttonRole", "secondary")
            repolish(b)
            b.clicked.connect(lambda _=False, s=src: _run_delete_chats(s))
            lay.addWidget(b)
        if len(inv.chats_by_source) > 1:
            b_all = QPushButton(f"Remove ALL ingested chats ({inv.total_chats})")
            b_all.clicked.connect(lambda: _run_delete_chats(None))
            lay.addWidget(b_all)
        if inv.total_projects:
            b_proj = QPushButton(f"Remove respawned projects ({inv.total_projects})")
            b_proj.clicked.connect(lambda: _run_delete_projects())
            lay.addWidget(b_proj)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setProperty("buttonRole", "secondary")
        repolish(close_btn)
        close_btn.clicked.connect(dlg.accept)
        close_row.addWidget(close_btn)
        lay.addLayout(close_row)
        dlg.exec()

    def _after_cleanse(self, dlg: QDialog, message: str, *, projects: bool = False) -> None:
        # Reload the open chat manager and rebuild the sidebars to drop removed rows.
        if self._chat_mgr is not None:
            self._chat_mgr.load_all_chats()
            # The open chat may have just been cleansed; fall back to a live one.
            if self._current_chat_id not in self._chat_mgr.chats:
                remaining = self._chat_mgr.list_chats()
                self._current_chat_id = (
                    str(remaining[0].get("id") or "")
                    if remaining
                    else self._chat_mgr.create_chat("Notes chat")
                )
                self._select_chat_id(self._current_chat_id or "")
        if projects:
            self.projectsMetadataChanged.emit()
        self._refresh_chat_list()
        self._show_current_chat()
        QMessageBox.information(self, "Manage ingested data", message)
        dlg.accept()

    def _confirm_ingest_size(
        self,
        path: Path,
        target_dir: Path,
        target_title: str,
        *,
        member_names: Optional[list[str]] = None,
        intro: Optional[str] = None,
    ) -> bool:
        """Show how big the import is vs. free disk before writing anything.

        Guards against accidentally dropping a giant (or zip-bombed) export into
        the assistant's memory: archives are measured by their UNCOMPRESSED size,
        and the import is refused outright when it would not fit on disk.
        """
        from . import notes_ai_import as ai_import

        info = ai_import.measure_export(path, member_names)
        if info.resolved_path is None:
            QMessageBox.warning(
                self,
                "Ingest chats",
                "Could not find a conversations.json in the selected file or folder.",
            )
            return False

        hb = ai_import.human_bytes
        free = ai_import.free_space_bytes(target_dir)
        src_drive = ai_import.volume_label(info.resolved_path) or "this drive"
        dst_drive = ai_import.volume_label(target_dir) or "this drive"

        # Hard stop: not enough room on the volume for the uncompressed data.
        if info.payload_bytes and free and info.payload_bytes > free:
            QMessageBox.critical(
                self,
                "Ingest chats — not enough space",
                f"This export expands to {hb(info.payload_bytes)} but only "
                f"{hb(free)} is free on “{dst_drive}”.\n\nImport cancelled.",
            )
            return False

        # Safety level for the indicator dot (hard-stop above already covers >free).
        risky_archive = info.is_archive and (
            info.compression_ratio >= 50 or info.payload_bytes >= 2 * 1024**3
        )
        if not free:
            color, verdict = "#f1c40f", "[CAUTION] Could not read free space on the destination."
        elif info.payload_bytes > free * 0.9 or risky_archive:
            color, verdict = "#e74c3c", "[RISK] This import is close to filling the disk."
        elif info.payload_bytes > free * 0.5 or (info.is_archive and info.compression_ratio >= 10):
            color, verdict = "#f1c40f", "[CAUTION] Large import relative to free space."
        else:
            color, verdict = "#2ecc71", "[SAFE] Comfortably within free space."

        esc = html.escape
        intro_line = intro or (
            f"This will ingest {hb(info.payload_bytes)} of conversation data into "
            f"“{target_title}” assistant memory."
        )
        gray = "color:#9aa0a6;"
        rows = [
            esc(intro_line),
            "",
            f"Selected file on disk: {hb(info.source_bytes)} "
            f"<span style='{gray}'>({esc(src_drive)})</span>",
            f"Estimated space it will take up: ~{hb(info.payload_bytes)} "
            f"<span style='{gray}'>on {esc(dst_drive)}</span>",
            f"Free space available: {hb(free)} "
            f"<span style='{gray}'>({esc(dst_drive)})</span>",
        ]
        # Surface an unusual expansion ratio (classic zip-bomb signature).
        if info.is_archive and info.compression_ratio >= 2:
            caution = (
                f"Uncompressed: {hb(info.payload_bytes)} "
                f"({info.compression_ratio:.0f}x larger than the archive)"
            )
            rows.append(f"<span style='{gray}'>{esc(caution)}</span>")
        dot = f"<span style='color:{color}; font-size:15px;'>&#9679;</span>"
        rows.extend(
            [
                "",
                f"{dot} {esc(verdict)}",
                "",
                "Continue?",
            ]
        )

        box = QMessageBox(self)
        box.setWindowTitle("Ingest chats")
        box.setIcon(QMessageBox.Icon.Question)
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText("<br>".join(rows))
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
        return box.exec() == QMessageBox.StandardButton.Yes

    def _set_ai_import_buttons_enabled(self, enabled: bool) -> None:
        for attr in (
            "_btn_import_auto",
            "_btn_import_chatgpt",
            "_btn_import_claude",
            "_btn_import_claude_projects",
        ):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(enabled)
        if not enabled and hasattr(self, "_btn_import_auto"):
            self._btn_import_auto.setText("Importing…")
        elif hasattr(self, "_btn_import_auto"):
            self._btn_import_auto.setText("Choose export file or folder…")

    def _begin_ai_import_progress(self) -> None:
        """Show the inline (non-modal) progress strip for the batched ingest."""
        self._ai_import_started = time.monotonic()
        self._task_progress.start("Ingesting chats", "Preparing import…")

    @staticmethod
    def _format_eta(seconds: float) -> str:
        secs = max(0, int(round(seconds)))
        if secs < 60:
            return f"{secs}s"
        mins, secs = divmod(secs, 60)
        if mins < 60:
            return f"{mins}m {secs:02d}s"
        hours, mins = divmod(mins, 60)
        return f"{hours}h {mins:02d}m"

    def _eta_suffix(self, done: int, total: int) -> str:
        """`` — ~Xh Ym remaining`` from the average rate so far (or empty)."""
        elapsed = time.monotonic() - self._ai_import_started
        if done > 0 and done < total and elapsed > 0.25:
            rate = done / elapsed  # items per second
            if rate > 0:
                return f" — ~{self._format_eta((total - done) / rate)} remaining"
        return ""

    def _on_ai_import_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self._task_progress.set_progress(0, 0, "Scanning export…")
            return
        self._task_progress.set_progress(
            done,
            total,
            f"Ingesting {done:,} of {total:,} conversations{self._eta_suffix(done, total)}",
        )

    def _cleanup_ai_import_worker(self) -> None:
        self._set_ai_import_buttons_enabled(True)
        worker = self._ai_import_worker
        self._ai_import_worker = None
        if worker is not None:
            worker.deleteLater()

    def _on_ai_import_finished(self, result: object) -> None:
        from .notes_ai_import import ImportResult, ProjectsImportResult

        # Capture the export details from the worker before it is torn down so we
        # can build the ingested-memory corpus afterwards.
        worker = self._ai_import_worker
        export_path = getattr(worker, "_path", None)
        worker_source = getattr(worker, "_source", None)

        self._cleanup_ai_import_worker()

        is_projects = isinstance(result, ProjectsImportResult)
        if not isinstance(result, (ImportResult, ProjectsImportResult)):
            return

        nothing_done = (not is_projects and result.chats_written == 0) or (
            is_projects and result.projects_created == 0 and result.projects_reused == 0
        )
        if result.errors and nothing_done:
            self._task_progress.hide_banner()
            QMessageBox.warning(self, "Ingest chats", "\n".join(result.errors))
            return

        # New/updated projects change the spaces sidebar; rebuild it from disk.
        if is_projects:
            self.projectsMetadataChanged.emit()
        # Refresh the chat sidebar (it rebuilds from every project) and reload the
        # open chat manager when chats may have landed in the current project.
        if self._ai_import_target_is_current and self._chat_mgr is not None:
            self._chat_mgr.load_all_chats()
        self._refresh_chat_list()

        # Report the import result inline (non-modal). The strip then transitions
        # straight into the memory-build phase below, which is the long one.
        msg = result.summary_line()
        if result.errors:
            msg += " Some items were skipped (see RAG log)."
            if hasattr(self, "rag_output"):
                for line in result.errors[:5]:
                    self.rag_output.append(f"[INGEST][SKIP] {line}")
        self._task_progress.set_progress(1, 1, msg + " Preparing memory…")

        # Knowledge transfer: organize the freshly ingested chats into durable
        # RAG memory so Tacitus can use them across every project.
        source = result.source if isinstance(result, ImportResult) and result.source else (
            str(worker_source or "") or "claude"
        )
        self._start_memory_build(source=source, export_path=export_path)

    def _start_memory_build(
        self, *, source: str, export_path: object, force: bool = False
    ) -> None:
        """Kick off building (or reindexing) the ingested-memory corpus.

        ``force`` regenerates docs for chats already on disk — this is the
        "reindex already-uploaded chats" path. When set, the whole FAISS
        namespace is rebuilt on completion rather than appended to.
        """
        if self._space_root is None:
            return
        if self._memory_build_worker is not None and self._memory_build_worker.isRunning():
            QMessageBox.information(
                self,
                "Reindex chats",
                "Memory is already being built — let it finish first.",
            )
            return
        vault_root = _notes_vault_root_from_space(self._space_root)
        spaces_root = self._spaces_root()
        if spaces_root is None:
            return
        params = self._notes_rag_engine_params()
        worker = _MemoryBuildWorker(
            vault_root=vault_root,
            spaces_root=spaces_root,
            source=source,
            export_path=Path(export_path) if export_path else None,
            summarize=True,
            model=params.get("model_id"),
            base_url=params.get("ollama_base_url"),
            force=force,
        )
        self._memory_build_force = force
        worker.progress.connect(self._on_memory_build_progress)
        worker.finished.connect(self._on_memory_build_finished)
        worker.failed.connect(self._on_memory_build_failed)
        self._memory_build_worker = worker
        # Always drive the inline strip: the per-chat summarization here is the
        # genuinely long phase (can run for hours), so the user must be able to
        # keep working while it runs and watch the ETA without a blocking dialog.
        self._ai_import_started = time.monotonic()
        verb = "Reindexing chats" if force else "Building memory"
        self._task_progress.start(
            verb, "Scanning ingested chats…" if force else "Organizing ingested chats…"
        )
        if hasattr(self, "rag_output"):
            log_verb = "Reindexing" if force else "Organizing"
            self.rag_output.append(f"[MEMORY] {log_verb} ingested chats into transferable memory…")
        worker.start()

    def _on_memory_build_progress(self, done: int, total: int) -> None:
        force = getattr(self, "_memory_build_force", False)
        verb = "Reindexing" if force else "Building memory for"
        if total <= 0:
            self._task_progress.set_progress(0, 0, "Scanning ingested chats…")
            return
        self._task_progress.set_progress(
            done,
            total,
            f"{verb} {done:,} of {total:,} conversations{self._eta_suffix(done, total)}",
        )

    def _on_memory_build_finished(self, result: object) -> None:
        worker = self._memory_build_worker
        self._memory_build_worker = None
        force = getattr(self, "_memory_build_force", False)
        self._memory_build_force = False
        if worker is not None:
            worker.deleteLater()
        if not isinstance(result, dict):
            self._task_progress.hide_banner()
            return
        doc_paths = result.get("doc_paths") or []
        summary = (
            f"{result.get('transcripts', 0)} transcript(s), "
            f"{result.get('summaries', 0)} summary(ies), "
            f"{result.get('memory_files', 0)} memory file(s)"
        )
        if hasattr(self, "rag_output"):
            self.rag_output.append(f"[MEMORY] Built {summary}.")
        # A reindex rebuilds the whole namespace so removed/changed docs drop out;
        # a fresh import only appends the new docs. The FAISS (re)build runs in the
        # background, so the strip reports the memory build as done and notes that
        # indexing continues.
        if force:
            self._rebuild_ingested_memory_index()
            title = "Reindex complete"
            if hasattr(self, "rag_output"):
                self.rag_output.append("[MEMORY] Reindex complete — memory index rebuilt.")
        else:
            if doc_paths:
                self._index_ingested_memory(doc_paths)
            title = "Memory ready"
        self._task_progress.complete(title, f"Built {summary}. Indexing in the background.")

    def _on_memory_build_failed(self, message: str) -> None:
        worker = self._memory_build_worker
        self._memory_build_worker = None
        self._memory_build_force = False
        if worker is not None:
            worker.deleteLater()
        self._task_progress.complete(
            "Memory build failed", message, autohide_ms=0
        )
        if hasattr(self, "rag_output"):
            self.rag_output.append(f"[MEMORY][ERROR] {message}")

    def _on_ai_import_failed(self, message: str) -> None:
        self._cleanup_ai_import_worker()
        self._task_progress.hide_banner()
        QMessageBox.critical(self, "Ingest chats", f"Import failed:\n{message}")

    def _load_ai_settings_form(self) -> None:
        if not hasattr(self, "_settings_key_openai"):
            return
        s = load_ai_settings()
        self._settings_assistant_name.setText(assistant_display_name(s))
        self._settings_key_openai.setText(s.get(KEY_OPENAI, ""))
        self._settings_key_anthropic.setText(s.get(KEY_ANTHROPIC, ""))
        self._settings_key_grok.setText(s.get(KEY_GROK, ""))
        self._settings_key_gemini.setText(s.get(KEY_GEMINI, ""))
        if hasattr(self, "_settings_system_prompt"):
            self._settings_system_prompt.setPlainText(system_prompt(s))
        self._local_gguf_paths = local_gguf_models(s)
        self._set_assistant_name(assistant_display_name(s))
        self._refresh_on_device_models_list()

    def _on_save_ai_settings(self) -> None:
        existing = load_ai_settings()
        settings = {
            KEY_ASSISTANT_NAME: self._settings_assistant_name.text(),
            KEY_OPENAI: self._settings_key_openai.text(),
            KEY_ANTHROPIC: self._settings_key_anthropic.text(),
            KEY_GROK: self._settings_key_grok.text(),
            KEY_GEMINI: self._settings_key_gemini.text(),
            KEY_LOCAL_GGUF_MODELS: local_gguf_models(existing),
            KEY_SYSTEM_PROMPT: self._settings_system_prompt.toPlainText()[:SYSTEM_PROMPT_MAX_CHARS],
            KEY_VOICE_PROFILE: existing.get(KEY_VOICE_PROFILE),
        }
        save_ai_settings(settings)
        self._set_assistant_name(assistant_display_name(settings))
        self._refresh_model_catalog()
        QMessageBox.information(
            self,
            "AI settings",
            f"Saved. {self.assistant_name()} is listed as this workspace assistant.",
        )

    def _apply_singularity_chrome(self) -> None:
        """Cv Ops palette on the composer bar (dark chrome, blue RAG chip)."""
        if not hasattr(self, "_composer_frame"):
            return
        from PyQt6.QtGui import QPalette

        bg = cvops_color("bg_panel")
        composer_fill = cvops_rgba("bg_panel", 0.96)
        # Free-form, fewer boxes: a subtle border for input affordance only.
        # Large filled containers drop their border entirely below.
        border = cvops_rgba("line_light", 0.45)
        iron = cvops_color("text_iron")
        bright = cvops_color("text_bright")
        void_ = cvops_color("bg_void")
        act = cvops_color("accent_active")
        selected_fill = cvops_rgba("selection_active", 0.88)
        selected_fill_strong = cvops_rgba("selection_active", 0.96)
        selected_text = cvops_color("selection_text")

        self._composer_frame.setStyleSheet(
            f"QFrame#notesSingularityComposer {{ background-color: {composer_fill}; "
            f"border: 1px solid {cvops_rgba('line_light', 0.28)}; border-radius: 0px; }}"
        )
        if hasattr(self, "chat_view"):
            self.chat_view.setStyleSheet(
                f"QTextBrowser {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                f"stop:0 {void_}, stop:0.18 {cvops_rgba('bg_panel', 0.10)}, "
                f"stop:0.50 {cvops_rgba('bg_panel', 0.18)}, stop:0.82 {cvops_rgba('bg_panel', 0.10)}, "
                f"stop:1 {void_}); border: none; "
                f"border-radius: 0px; padding: 14px 16px; color: {bright}; }}"
            )
        if hasattr(self, "_chat_timeline_panel"):
            self._chat_timeline_panel.setStyleSheet(
                f"QFrame#notesChatTimelinePanel {{ background-color: {cvops_rgba('bg_panel', 0.34)}; "
                f"border: none; border-left: 1px solid {cvops_rgba('line_light', 0.16)}; }}"
                f"QListWidget#notesChatTimelineList {{ background-color: transparent; border: none; "
                f"outline: none; color: {iron}; font-size: 10px; }}"
                f"QListWidget#notesChatTimelineList::item {{ padding: 4px 1px; border-left: 2px solid {cvops_rgba('line_light', 0.32)}; }}"
                f"QListWidget#notesChatTimelineList::item:hover {{ background: {cvops_rgba('text_signal', 0.08)}; }}"
                f"QListWidget#notesChatTimelineList::item:selected {{ background: {selected_fill}; "
                f"color: {selected_text}; border-left: 2px solid {selected_text}; }}"
            )
        header_ss = (
            f"QLabel {{ color: {iron}; font-size: 10px; font-weight: 600; "
            "letter-spacing: 0.08em; text-transform: uppercase; "
            "padding: 4px 2px 2px 2px; border: none; background: transparent; }"
        )
        for attr in (
            "_chat_pinned_header",
            "_chat_recent_header",
            "_project_pinned_header",
            "_project_recent_header",
        ):
            w = getattr(self, attr, None)
            if w is not None:
                w.setStyleSheet(header_ss)
        if hasattr(self, "chat_ollama_url"):
            self.chat_ollama_url.setStyleSheet(
                f"QLineEdit {{ background-color: {void_}; border: 1px solid {border}; "
                f"border-radius: 0px; color: {bright}; padding: 4px 8px; }}"
            )
        self.chat_input.setStyleSheet(
            f"QPlainTextEdit#notesComposerMessage {{ background-color: {composer_fill}; "
            f"border: 1px solid {cvops_rgba('line_light', 0.24)}; border-radius: 0px; "
            f"color: {bright}; padding: 10px 12px; }}"
        )
        pal_in = self.chat_input.palette()
        pal_in.setColor(QPalette.ColorRole.PlaceholderText, QColor(iron))
        self.chat_input.setPalette(pal_in)

        self.chat_model.setStyleSheet(
            f"QComboBox {{ background-color: {composer_fill}; border: 1px solid {border}; border-radius: 0px; "
            f"color: {bright}; padding: 4px 10px; min-height: 26px; }}"
        )
        le = self.chat_model.lineEdit()
        if le is not None:
            ple = le.palette()
            ple.setColor(QPalette.ColorRole.PlaceholderText, QColor(iron))
            le.setPalette(ple)

        if hasattr(self, "_btn_compose_rag"):
            self._btn_compose_rag.setStyleSheet(
                f"QPushButton#notesSingularityRagChip {{ background-color: transparent; border: 1px solid {border}; "
                f"border-radius: 0px; color: {bright}; font-weight: 600; padding: 0 12px; }}"
                f"QPushButton#notesSingularityRagChip:hover {{ border-color: {act}; color: {act}; }}"
            )
        if hasattr(self, "_btn_compose_tools"):
            self._btn_compose_tools.setStyleSheet(
                f"QPushButton#notesSingularityTools {{ background-color: transparent; border: 1px solid {border}; "
                f"border-radius: 0px; color: {bright}; padding: 0 12px; }}"
                f"QPushButton#notesSingularityTools:checked {{ background-color: {selected_fill}; color: {selected_text}; }}"
            )
        if hasattr(self, "_btn_compose_send"):
            self._btn_compose_send.setStyleSheet(
                f"QPushButton#notesSingularitySend {{ border-radius: 0px; padding: 0 14px; }}"
            )
        if hasattr(self, "_btn_compose_stop"):
            self._btn_compose_stop.setStyleSheet(
                f"QPushButton#notesSingularityStop {{ background-color: transparent; border: 1px solid {border}; "
                f"border-radius: 0px; color: {bright}; padding: 0 12px; }}"
            )
        if hasattr(self, "_btn_compose_dictate"):
            self._btn_compose_dictate.setStyleSheet(
                f"QPushButton#notesSingularityDictate {{ background-color: transparent; border: 1px solid {border}; "
                f"border-radius: 0px; color: {bright}; padding: 0 8px; }}"
                f"QPushButton#notesSingularityDictate:hover {{ border-color: {act}; color: {act}; }}"
                f"QPushButton#notesSingularityDictate:checked {{ background-color: {act}; color: {void_}; "
                f"border-color: {act}; }}"
                f"QPushButton#notesSingularityDictate:disabled {{ color: {iron}; }}"
            )
        if hasattr(self, "_tts_bar"):
            self._tts_bar.setStyleSheet(
                f"QWidget#notesTtsPlaybackBar {{ background-color: {composer_fill}; "
                f"border: 1px solid {border}; border-radius: 0px; }}"
                f"QWidget#notesTtsPlaybackBar QLabel {{ color: {bright}; font-size: 11px; }}"
                f"QPushButton#notesTtsPlay, QPushButton#notesTtsClose {{ background-color: transparent; "
                f"border: 1px solid {border}; border-radius: 0px; color: {bright}; padding: 2px 4px; }}"
                f"QPushButton#notesTtsPlay:hover, QPushButton#notesTtsClose:hover {{ "
                f"border-color: {act}; color: {act}; }}"
                f"QSlider#notesTtsSeek::groove:horizontal {{ height: 4px; background: {border}; }}"
                f"QSlider#notesTtsSeek::handle:horizontal {{ width: 10px; margin: -5px 0; "
                f"background: {act}; border-radius: 5px; }}"
                f"QSlider#notesTtsSeek::sub-page:horizontal {{ background: {act}; }}"
            )
        if hasattr(self, "_tools_strip"):
            self._tools_strip.setStyleSheet(
                f"QFrame#notesComposerToolsStrip {{ background-color: transparent; border: none; }}"
                f"QPushButton {{ background-color: transparent; border: 1px solid {border}; color: {bright}; padding: 4px 10px; }}"
                f"QPushButton:checked {{ background-color: {selected_fill}; color: {selected_text}; }}"
                f"QLabel#notesModelSpeedBadge {{ border: 1px solid {border}; background: {void_}; color: {bright}; padding: 4px 8px; }}"
            )
        if hasattr(self, "chat_list_pinned"):
            glass = cvops_rgba("text_signal", 0.08)
            selected = selected_fill
            sidebar_list_ids = (
                "notesChatSidebarPinned",
                "notesChatSidebarRecent",
                "notesProjectSidebarPinned",
                "notesProjectSidebarRecent",
            )
            sel = (
                ", ".join(
                    f"QWidget#notesAiWorkspace QListWidget#{oid}::item:selected" for oid in sidebar_list_ids
                )
            )
            hover = ", ".join(
                f"QWidget#notesAiWorkspace QListWidget#{oid}::item:hover" for oid in sidebar_list_ids
            )
            items = ", ".join(
                f"QWidget#notesAiWorkspace QListWidget#{oid}::item" for oid in sidebar_list_ids
            )
            roots = ", ".join(f"QWidget#notesAiWorkspace QListWidget#{oid}" for oid in sidebar_list_ids)
            catalog_ss = (
                f"{roots} {{ border: none; outline: none; background-color: transparent; color: {bright}; }}"
                f"{items} {{ border: none; padding: 0px; margin: 0px; background: transparent; }}"
                f"{hover} {{ background: {glass}; }}"
                f"{sel} {{ background: {selected}; color: {selected_text}; }}"
            )
            self.chat_list_pinned.setStyleSheet(catalog_ss)
            self.chat_list_recent.setStyleSheet(catalog_ss)
            self.project_list_pinned.setStyleSheet(catalog_ss)
            self.project_list_recent.setStyleSheet(catalog_ss)
        if hasattr(self, "_btn_sidebar_create"):
            self._btn_sidebar_create.setStyleSheet(
                f"QToolButton#notesSidebarCreate {{ background-color: transparent; border: 1px solid {border}; "
                f"border-radius: 0px; color: {bright}; padding: 2px 10px; font-size: 16px; font-weight: 600; }}"
                f"QToolButton#notesSidebarCreate:hover {{ border-color: {act}; color: {act}; }}"
            )
        if hasattr(self, "_events_artifacts_panel"):
            self._events_artifacts_panel.setStyleSheet(
                f"QFrame#notesEventsArtifactsPanel {{ background-color: {cvops_rgba('bg_panel', 0.72)}; "
                f"border: none; border-left: 1px solid {cvops_rgba('line_light', 0.18)}; }}"
                f"QListWidget#notesChatJobsList, QListWidget#notesChatArtifactsList, "
                f"QListWidget#notesProjectEventsList {{ background-color: transparent; border: none; "
                f"outline: none; color: {bright}; }}"
                f"QListWidget#notesChatJobsList::item, QListWidget#notesChatArtifactsList::item, "
                f"QListWidget#notesProjectEventsList::item {{ padding: 4px 2px; }}"
                f"QListWidget#notesChatJobsList::item:selected, QListWidget#notesChatArtifactsList::item:selected, "
                f"QListWidget#notesProjectEventsList::item:selected {{ background: {selected_fill}; "
                f"color: {selected_text}; }}"
                f"QToolButton {{ background-color: transparent; border: 1px solid {border}; "
                f"color: {bright}; padding: 2px 6px; }}"
            )
        row_chunk = (
            f"QWidget#notesSidebarRowHost {{ background: transparent; }}"
            f" QWidget#notesSidebarRowHost QLabel#notesSidebarRowTitle {{ border: none; background: transparent; "
            f"text-decoration: none; }}"
            f" QListWidget#notesChatSidebarPinned QWidget#notesSidebarRowHost QLabel#notesSidebarRowTitle, "
            f"QListWidget#notesProjectSidebarPinned QWidget#notesSidebarRowHost QLabel#notesSidebarRowTitle {{ "
            f"color: {bright}; font-weight: 600; }}"
            f" QListWidget#notesChatSidebarRecent QWidget#notesSidebarRowHost QLabel#notesSidebarRowTitle, "
            f"QListWidget#notesProjectSidebarRecent QWidget#notesSidebarRowHost QLabel#notesSidebarRowTitle {{ "
            f"color: {cvops_rgba('text_bright', 0.70)}; font-weight: 500; }}"
            f" QWidget#notesSidebarRowHost QLabel#notesSidebarRowIcon {{ color: {cvops_rgba('text_bright', 0.70)}; }}"
            f" QWidget#notesSidebarRowHost QLabel#notesSidebarActivity {{ color: {act}; "
            f"border: none; background: transparent; font-weight: 700; }}"
            f" QToolButton#notesSidebarRowOverflow {{ background-color: transparent; border: none; "
            f"color: {cvops_rgba('text_iron', 0.92)}; padding: 0px 4px; }}"
        )
        _row_key = "/*notes_sidebar_row_chrome*/"
        base_ss = (self.styleSheet() or "").strip()
        if _row_key in base_ss:
            base_ss = base_ss.split(_row_key)[0].rstrip()
        self.setStyleSheet(
            f"{base_ss}\n{_row_key}\n{row_chunk}".strip() if base_ss else f"{_row_key}\n{row_chunk}".strip()
        )
        if hasattr(self, "_btn_mode_chats"):
            mode_btn_ss = (
                f"QPushButton {{ background-color: transparent; border: none; border-bottom: 1px solid {border}; "
                f"border-radius: 0px; color: {iron}; padding: 6px 2px; font-weight: 600; }}"
                f"QPushButton:checked {{ background-color: {selected_fill}; color: {selected_text}; "
                f"border-bottom: 2px solid {cvops_rgba('selection_edge', 0.92)}; }}"
            )
            self._btn_mode_chats.setStyleSheet(mode_btn_ss)
            self._btn_mode_projects.setStyleSheet(mode_btn_ss)
        if hasattr(self, "_rag_doc_list"):
            self._rag_doc_list.setStyleSheet(
                f"QListWidget#notesRagDocList {{ border: 1px solid {border}; border-radius: 0px; "
                f"background-color: {void_}; color: {bright}; }}"
            )
        if hasattr(self, "rag_preview"):
            self.rag_preview.setStyleSheet(
                f"QPlainTextEdit#notesRagPreview {{ border: 1px solid {border}; border-radius: 0px; "
                f"background-color: {void_}; color: {bright}; padding: 8px; }}"
            )
        if hasattr(self, "rag_path_entry"):
            self.rag_path_entry.setStyleSheet(
                f"QLineEdit {{ background-color: {void_}; border: 1px solid {border}; border-radius: 0px; "
                f"color: {bright}; padding: 4px 8px; }}"
            )
        _rag_combo_ss = (
            f"QComboBox {{ background-color: {void_}; border: 1px solid {border}; border-radius: 0px; "
            f"color: {bright}; padding: 4px 8px; min-height: 24px; }}"
        )
        if hasattr(self, "rag_chat_model"):
            self.rag_chat_model.setStyleSheet(_rag_combo_ss)
        if hasattr(self, "rag_embed_model"):
            self.rag_embed_model.setStyleSheet(_rag_combo_ss)
        if hasattr(self, "_chat_header"):
            self._chat_header.setStyleSheet(
                f"QFrame#notesChatHeader {{ background-color: {bg}; "
                f"border: none; border-bottom: 1px solid {cvops_rgba('line_light', 0.20)}; "
                f"border-radius: 0px; }}"
                f"QLabel#notesChatHeaderTitle {{ color: {bright}; font-weight: 600; "
                f"font-size: 12px; letter-spacing: 0.015em; }}"
                f"QLabel#notesChatHeaderModel {{ color: {cvops_rgba('text_bright', 0.56)}; font-size: 10px; "
                f"font-family: 'JetBrains Mono', 'Menlo', monospace; "
                f"border: none; padding: 0px 2px; }}"
                f"QLabel#notesChatHeaderStatus[statusKind=\"streaming\"] {{ "
                f"color: {act}; background-color: transparent; padding: 0px 2px; "
                f"font-size: 10px; font-weight: 600; }}"
                f"QLabel#notesChatHeaderStatus[statusKind=\"error\"] {{ "
                f"color: {cvops_color('accent_alert')}; background-color: transparent; "
                f"padding: 0px 2px; font-size: 10px; font-weight: 600; }}"
                f"QLabel#notesChatHeaderStatus[statusKind=\"idle\"] {{ "
                f"color: {cvops_rgba('text_bright', 0.40)}; padding: 0px 2px; font-size: 10px; }}"
                f"QToolButton#notesChatHeaderSetup, QToolButton#notesChatHeaderMore, "
                f"QToolButton#notesChatHeaderArtifacts {{ "
                f"background-color: transparent; border: none; "
                f"color: {cvops_rgba('text_bright', 0.62)}; padding: 2px 6px; }}"
                f"QToolButton#notesChatHeaderSetup:checked, QToolButton#notesChatHeaderArtifacts:checked {{ "
                f"background-color: {selected_fill}; color: {selected_text}; }}"
                f"QToolButton#notesChatHeaderSetup:hover, QToolButton#notesChatHeaderMore:hover, "
                f"QToolButton#notesChatHeaderArtifacts:hover {{ "
                f"color: {bright}; background-color: {cvops_rgba('bg_panel', 0.20)}; }}"
            )
        if hasattr(self, "_chat_setup_bar"):
            self._chat_setup_bar.setStyleSheet(
                f"QFrame#notesChatSetupBar {{ background-color: {bg}; border: none; border-bottom: 1px solid {border}; }}"
            )
        if hasattr(self, "_composer_footer"):
            self._composer_footer.setStyleSheet(
                f"QFrame#notesComposerFooter {{ background-color: transparent; border: none; "
                f"border-top: 1px solid {cvops_rgba('line_light', 0.14)}; padding-top: 3px; }}"
            )
        if hasattr(self, "_btn_chat_sync_ollama_models"):
            self._btn_chat_sync_ollama_models.setStyleSheet(
                f"QPushButton {{ background-color: transparent; border: 1px solid {border}; "
                f"border-radius: 0px; color: {bright}; padding: 4px 10px; }}"
                f"QPushButton:hover {{ border-color: {act}; color: {act}; }}"
            )
        if hasattr(self, "_sidebar_search"):
            self._sidebar_search.setStyleSheet(
                f"QLineEdit#notesSidebarSearch {{ background-color: {void_}; "
                f"border: 1px solid {border}; border-radius: 0px; color: {bright}; "
                f"padding: 6px 8px; }}"
            )
        if hasattr(self, "_tools_strip"):
            self._tools_strip.setStyleSheet(
                f"QFrame#notesComposerToolsStrip {{ background-color: transparent; "
                f"border: none; border-radius: 0px; }}"
                f"QPushButton {{ background-color: transparent; border: 1px solid {cvops_rgba('line_light', 0.22)}; "
                f"color: {cvops_rgba('text_bright', 0.78)}; padding: 2px 8px; min-height: 20px; }}"
                f"QPushButton:checked {{ background-color: {selected_fill_strong}; color: {selected_text}; font-weight: bold; }}"
                f"QLabel {{ color: {iron}; font-size: 10px; }}"
            )
        if hasattr(self, "_btn_compose_tools"):
            self._btn_compose_tools.setStyleSheet(
                f"QPushButton#notesSingularityTools {{ background-color: transparent; "
                f"border: 1px solid {cvops_rgba('line_light', 0.22)}; border-radius: 0px; color: {cvops_rgba('text_bright', 0.80)}; "
                f"padding: 0 10px; }}"
                f"QPushButton#notesSingularityTools:checked {{ background-color: {selected_fill}; color: {selected_text}; }}"
            )
        if hasattr(self, "_model_speed_badge"):
            self._model_speed_badge.setStyleSheet(
                f"QLabel#notesModelSpeedBadge {{ color: {iron}; "
                f"font-family: 'JetBrains Mono', 'Menlo', monospace; font-size: 10px; "
                f"font-weight: 600; letter-spacing: 0.08em; "
                f"border: none; background-color: transparent; padding: 2px 2px; "
                f"text-transform: uppercase; }}"
            )
        if hasattr(self, "_btn_compose_stop"):
            self._btn_compose_stop.setStyleSheet(
                f"QPushButton#notesSingularityStop {{ background-color: transparent; "
                f"border: none; border-radius: 0px; "
                f"color: {cvops_rgba('text_bright', 0.44)}; padding: 0 8px; }}"
                f"QPushButton#notesSingularityStop:enabled {{ color: {cvops_color('accent_alert')}; }}"
                f"QPushButton#notesSingularityStop:disabled {{ color: {iron}; "
                f"border: none; }}"
            )

    def sync_project_list(
        self,
        entries: list[Union[tuple[str, str, str], tuple[str, str, str, bool]]],
        current_space_id: str,
    ) -> None:
        """Populate the Projects sidebar (``entries`` are ``(space_id, title, goals[, pinned])``)."""
        if not hasattr(self, "project_list_recent"):
            return
        rows: list[tuple[str, str, str, bool]] = []
        for e in entries:
            if len(e) >= 4:
                sid, title, goals, pinned = e[0], e[1], e[2], bool(e[3])
            else:
                sid, title, goals, pinned = e[0], e[1], e[2], False
            rows.append((str(sid), str(title), str(goals), pinned))
        rows.sort(key=lambda r: (0 if r[3] else 1, (r[1] or r[0]).lower()))
        self._project_rows = rows
        pinned_rows = [r for r in rows if r[3]]
        recent_rows = [r for r in rows if not r[3]]
        show_pinned = bool(pinned_rows)
        self._project_pinned_header.setVisible(show_pinned)
        self.project_list_pinned.setVisible(show_pinned)
        for lw in (self.project_list_pinned, self.project_list_recent):
            lw.blockSignals(True)
            lw.clear()
        for sid, title, goals, _p in pinned_rows:
            self._add_project_list_row(self.project_list_pinned, sid, title, goals)
        for sid, title, goals, _p in recent_rows:
            self._add_project_list_row(self.project_list_recent, sid, title, goals)
        for lw in (self.project_list_pinned, self.project_list_recent):
            lw.blockSignals(False)
        self._sync_pinned_list_height(self.project_list_pinned, max_rows=3)
        self._select_project_row(current_space_id)
        if hasattr(self, "chat_list_pinned") and self._space_root is not None:
            self._refresh_chat_list()

    def set_project_list_selection(self, space_id: str) -> None:
        """Highlight ``space_id`` in the Projects list (no ``projectSelected`` emit)."""
        if not hasattr(self, "project_list_recent"):
            return
        for lw in (self.project_list_pinned, self.project_list_recent):
            lw.blockSignals(True)
        self._select_project_row(space_id)
        for lw in (self.project_list_pinned, self.project_list_recent):
            lw.blockSignals(False)

    def focus_chats_sidebar_mode(self) -> None:
        """Show the Chats sidebar (e.g. after creating a new project)."""
        if not hasattr(self, "_btn_mode_chats"):
            return
        self._btn_mode_chats.setChecked(True)
        self._on_sidebar_mode_changed(0)

    def _on_sidebar_mode_changed(self, idx: int) -> None:
        mode_idx = int(idx)
        if hasattr(self, "_sidebar_stack"):
            self._sidebar_stack.setCurrentIndex(mode_idx)
        self._set_main_view_mode(mode_idx == 1)
        self._sync_sidebar_create_button()

    def _handle_sidebar_create_action(self) -> None:
        if bool(getattr(self, "_btn_mode_projects", None) and self._btn_mode_projects.isChecked()):
            self.newProjectClicked.emit()
            return
        self._new_chat()

    def _sync_sidebar_create_button(self) -> None:
        btn = getattr(self, "_btn_sidebar_create", None)
        if btn is None:
            return
        projects_mode = bool(getattr(self, "_btn_mode_projects", None) and self._btn_mode_projects.isChecked())
        btn.setToolTip("New project" if projects_mode else "New chat")
        if hasattr(self, "_sidebar_search"):
            self._sidebar_search.setPlaceholderText("Search projects…" if projects_mode else "Search chats…")

    def _select_project_row(self, space_id: str) -> None:
        for listw in (self.project_list_pinned, self.project_list_recent):
            for i in range(listw.count()):
                it = listw.item(i)
                if it is not None and str(it.data(Qt.ItemDataRole.UserRole) or "") == space_id:
                    listw.setCurrentItem(it)
                    return

    def _sync_pinned_list_height(self, listw: Optional[QListWidget], *, max_rows: int = 3) -> None:
        """Cap pinned list height to visible content with at most ``max_rows`` rows."""
        if listw is None:
            return
        heights: list[int] = []
        for i in range(listw.count()):
            item = listw.item(i)
            if item is None or item.isHidden():
                continue
            h = int(listw.sizeHintForRow(i))
            if h <= 0:
                h = int(item.sizeHint().height() or 0)
            heights.append(max(1, h))
        if not heights:
            listw.setMinimumHeight(0)
            listw.setMaximumHeight(0)
            return
        visible = heights[: max(1, int(max_rows))]
        margins = listw.contentsMargins()
        frame = max(0, int(listw.frameWidth()) * 2)
        target = int(sum(visible) + margins.top() + margins.bottom() + frame + 2)
        listw.setMinimumHeight(target)
        listw.setMaximumHeight(target)
        listw.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def _add_project_list_row(self, listw: QListWidget, sid: str, title: str, goals: str) -> None:
        label = str(title or sid)
        sdir = self._space_dir(sid)
        import_source = read_space_import_source(sdir) if sdir is not None else ""
        tips: list[str] = []
        if import_source:
            label = f"{self._ingested_tag(import_source)}{label}"
            tips.append(
                f"[INGESTED] Respawned from a {import_source} project export — "
                "not created on this system."
            )
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, sid)
        if goals:
            tips.append(goals)
        if tips:
            item.setToolTip("\n".join(tips))
        listw.addItem(item)
        host = _NotesSidebarRowHost(
            listw, item, label, for_project=True, workspace=self, parent=listw
        )
        listw.setItemWidget(item, host)
        # Pin sizeHint width to the current viewport. _AutoFitListWidget will
        # keep it in sync on subsequent resizes; this is just the initial value.
        w = max(1, listw.viewport().width())
        h = max(22, host.sizeHint().height())
        item.setSizeHint(QSize(w, h))

    def _build_events_artifacts_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("notesEventsArtifactsPanel")
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(420)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        head = QHBoxLayout()
        title = QLabel("Events & artifacts")
        title.setProperty("isTitle", True)
        repolish(title)
        head.addWidget(title, stretch=1)
        refresh_btn = QToolButton()
        refresh_btn.setText("↻")
        refresh_btn.setToolTip("Refresh events and artifacts")
        refresh_btn.clicked.connect(self._refresh_events_artifacts_panel)
        head.addWidget(refresh_btn)
        lay.addLayout(head)

        self._events_artifacts_scope = QLabel("")
        self._events_artifacts_scope.setProperty("muted", True)
        self._events_artifacts_scope.setWordWrap(True)
        repolish(self._events_artifacts_scope)
        lay.addWidget(self._events_artifacts_scope)

        self._chat_jobs_header = QLabel("Chat jobs")
        self._chat_jobs_header.setProperty("muted", True)
        repolish(self._chat_jobs_header)
        lay.addWidget(self._chat_jobs_header)
        self._chat_jobs_list = QListWidget()
        self._chat_jobs_list.setObjectName("notesChatJobsList")
        self._chat_jobs_list.setFrameShape(QFrame.Shape.NoFrame)
        self._chat_jobs_list.itemDoubleClicked.connect(self._open_events_artifacts_item)
        lay.addWidget(self._chat_jobs_list, stretch=2)

        self._chat_artifacts_header = QLabel("Chat artifacts")
        self._chat_artifacts_header.setProperty("muted", True)
        repolish(self._chat_artifacts_header)
        lay.addWidget(self._chat_artifacts_header)
        self._chat_artifacts_list = QListWidget()
        self._chat_artifacts_list.setObjectName("notesChatArtifactsList")
        self._chat_artifacts_list.setFrameShape(QFrame.Shape.NoFrame)
        self._chat_artifacts_list.itemDoubleClicked.connect(self._open_events_artifacts_item)
        lay.addWidget(self._chat_artifacts_list, stretch=2)

        self._project_events_header = QLabel("Project events/artifacts")
        self._project_events_header.setProperty("muted", True)
        repolish(self._project_events_header)
        lay.addWidget(self._project_events_header)
        self._project_events_list = QListWidget()
        self._project_events_list.setObjectName("notesProjectEventsList")
        self._project_events_list.setFrameShape(QFrame.Shape.NoFrame)
        self._project_events_list.itemDoubleClicked.connect(self._open_events_artifacts_item)
        lay.addWidget(self._project_events_list, stretch=3)
        return panel

    def _build_chat_timeline_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("notesChatTimelinePanel")
        panel.setFixedWidth(58)
        panel.setToolTip("Message timeline")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 6, 4, 6)
        lay.setSpacing(0)

        self._chat_timeline_list = QListWidget()
        self._chat_timeline_list.setObjectName("notesChatTimelineList")
        self._chat_timeline_list.setFrameShape(QFrame.Shape.NoFrame)
        self._chat_timeline_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._chat_timeline_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._chat_timeline_list.setSpacing(2)
        self._chat_timeline_list.setUniformItemSizes(True)
        self._chat_timeline_list.currentItemChanged.connect(self._on_chat_timeline_changed)
        lay.addWidget(self._chat_timeline_list, stretch=1)
        return panel

    def _build_chat_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        self._prepend_tab_info_row(layout, self._on_info_ai_chat)

        split = QSplitter(Qt.Orientation.Horizontal)
        self._chat_splitter = split
        split.setChildrenCollapsible(False)
        split.setHandleWidth(4)

        left_sidebar = QWidget()
        self._chat_sidebar = left_sidebar
        left_ll = QVBoxLayout(left_sidebar)
        left_ll.setContentsMargins(0, 0, 0, 0)
        left_ll.setSpacing(4)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        self._btn_mode_chats = QPushButton("Chats")
        self._btn_mode_projects = QPushButton("Projects")
        for b in (self._btn_mode_chats, self._btn_mode_projects):
            b.setCheckable(True)
        self._sidebar_mode_group = QButtonGroup(self)
        self._sidebar_mode_group.setExclusive(True)
        self._sidebar_mode_group.addButton(self._btn_mode_chats, 0)
        self._sidebar_mode_group.addButton(self._btn_mode_projects, 1)
        self._btn_mode_chats.setChecked(True)
        mode_row.addWidget(self._btn_mode_chats, stretch=1)
        mode_row.addWidget(self._btn_mode_projects, stretch=1)
        self._btn_sidebar_create = QToolButton()
        self._btn_sidebar_create.setObjectName("notesSidebarCreate")
        self._btn_sidebar_create.setText("+")
        self._btn_sidebar_create.setToolTip("New chat")
        self._btn_sidebar_create.clicked.connect(self._handle_sidebar_create_action)
        mode_row.addWidget(self._btn_sidebar_create)
        left_ll.addLayout(mode_row)

        self._sidebar_search = QLineEdit()
        self._sidebar_search.setObjectName("notesSidebarSearch")
        self._sidebar_search.setPlaceholderText("Search chats…")
        self._sidebar_search.setClearButtonEnabled(True)
        self._sidebar_search.textChanged.connect(self._on_sidebar_search_changed)
        left_ll.addWidget(self._sidebar_search)

        self._sidebar_stack = QStackedWidget()
        chat_page = QWidget()
        chat_page_l = QVBoxLayout(chat_page)
        chat_page_l.setContentsMargins(0, 0, 0, 0)
        chat_page_l.setSpacing(6)
        self._chat_pinned_header = QLabel("Pinned")
        self._chat_pinned_header.setProperty("muted", True)
        repolish(self._chat_pinned_header)
        chat_page_l.addWidget(self._chat_pinned_header)
        self.chat_list_pinned = _AutoFitListWidget()
        self.chat_list_pinned.setObjectName("notesChatSidebarPinned")
        self.chat_list_pinned.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_list_pinned.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list_pinned.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chat_list_pinned.currentItemChanged.connect(self._on_chat_pinned_changed)
        chat_page_l.addWidget(self.chat_list_pinned, stretch=0)
        self._chat_recent_header = QLabel("Recents")
        self._chat_recent_header.setProperty("muted", True)
        repolish(self._chat_recent_header)
        chat_page_l.addWidget(self._chat_recent_header)
        self.chat_list_recent = _AutoFitListWidget()
        self.chat_list_recent.setObjectName("notesChatSidebarRecent")
        self.chat_list_recent.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_list_recent.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_list_recent.currentItemChanged.connect(self._on_chat_recent_changed)
        chat_page_l.addWidget(self.chat_list_recent, stretch=1)
        self._sidebar_stack.addWidget(chat_page)

        proj_page = QWidget()
        proj_page_l = QVBoxLayout(proj_page)
        proj_page_l.setContentsMargins(0, 0, 0, 0)
        proj_page_l.setSpacing(6)
        self._project_pinned_header = QLabel("Pinned")
        self._project_pinned_header.setProperty("muted", True)
        repolish(self._project_pinned_header)
        proj_page_l.addWidget(self._project_pinned_header)
        self.project_list_pinned = _AutoFitListWidget()
        self.project_list_pinned.setObjectName("notesProjectSidebarPinned")
        self.project_list_pinned.setFrameShape(QFrame.Shape.NoFrame)
        self.project_list_pinned.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.project_list_pinned.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.project_list_pinned.currentItemChanged.connect(self._on_project_pinned_changed)
        proj_page_l.addWidget(self.project_list_pinned, stretch=0)
        self._project_recent_header = QLabel("Recents")
        self._project_recent_header.setProperty("muted", True)
        repolish(self._project_recent_header)
        proj_page_l.addWidget(self._project_recent_header)
        self.project_list_recent = _AutoFitListWidget()
        self.project_list_recent.setObjectName("notesProjectSidebarRecent")
        self.project_list_recent.setFrameShape(QFrame.Shape.NoFrame)
        self.project_list_recent.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.project_list_recent.currentItemChanged.connect(self._on_project_recent_changed)
        proj_page_l.addWidget(self.project_list_recent, stretch=1)
        self._sidebar_stack.addWidget(proj_page)

        self._sidebar_mode_group.idClicked.connect(self._on_sidebar_mode_changed)
        left_ll.addWidget(self._sidebar_stack, stretch=1)
        left_sidebar.setMinimumWidth(216)
        split.addWidget(left_sidebar)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        self._chat_header = QFrame()
        self._chat_header.setObjectName("notesChatHeader")
        ch_lay = QHBoxLayout(self._chat_header)
        ch_lay.setContentsMargins(10, 4, 10, 4)
        ch_lay.setSpacing(8)
        self._chat_header_title = _FullTextLabel("New chat")
        self._chat_header_title.setObjectName("notesChatHeaderTitle")
        ch_lay.addWidget(self._chat_header_title, stretch=1)
        self._chat_header_model = QLabel("—")
        self._chat_header_model.setObjectName("notesChatHeaderModel")
        self._chat_header_model.setToolTip("Active model route")
        ch_lay.addWidget(self._chat_header_model)
        self._chat_header_status = QLabel("Idle")
        self._chat_header_status.setObjectName("notesChatHeaderStatus")
        ch_lay.addWidget(self._chat_header_status)
        self._btn_chat_header_artifacts = QToolButton()
        self._btn_chat_header_artifacts.setObjectName("notesChatHeaderArtifacts")
        self._btn_chat_header_artifacts.setText("▦")
        self._btn_chat_header_artifacts.setCheckable(True)
        self._btn_chat_header_artifacts.setChecked(True)
        self._btn_chat_header_artifacts.setToolTip("Toggle the events & artifacts panel")
        self._btn_chat_header_artifacts.toggled.connect(self._toggle_events_artifacts_panel)
        ch_lay.addWidget(self._btn_chat_header_artifacts)
        self._btn_chat_header_setup = QToolButton()
        self._btn_chat_header_setup.setObjectName("notesChatHeaderSetup")
        self._btn_chat_header_setup.setText("AI")
        self._btn_chat_header_setup.setCheckable(True)
        self._btn_chat_header_setup.setToolTip("Show AI connection controls")
        self._btn_chat_header_setup.toggled.connect(self._toggle_chat_setup_bar)
        ch_lay.addWidget(self._btn_chat_header_setup)
        self._btn_chat_header_more = QToolButton()
        self._btn_chat_header_more.setObjectName("notesChatHeaderMore")
        self._btn_chat_header_more.setText("⋯")
        self._btn_chat_header_more.setToolTip("Chat actions")
        self._btn_chat_header_more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._btn_chat_header_more.clicked.connect(self._open_chat_header_menu)
        ch_lay.addWidget(self._btn_chat_header_more)
        rv.addWidget(self._chat_header)

        self._chat_setup_open = False
        self._chat_setup_bar = QFrame()
        self._chat_setup_bar.setObjectName("notesChatSetupBar")
        self._chat_setup_bar.setVisible(False)
        setup_lay = QHBoxLayout(self._chat_setup_bar)
        setup_lay.setContentsMargins(12, 8, 12, 8)
        setup_lay.setSpacing(8)
        self.chat_ollama_url = QLineEdit("http://localhost:11434")
        self.chat_ollama_url.setMinimumWidth(240)
        self.chat_ollama_url.setPlaceholderText("Ollama base URL")
        setup_lay.addWidget(self.chat_ollama_url, stretch=1)
        self._btn_chat_sync_ollama_models = QPushButton("Sync models")
        self._btn_chat_sync_ollama_models.setToolTip(
            "Reload Ollama tags from the `ollama list` CLI and GET /api/tags using the base URL above."
        )
        self._btn_chat_sync_ollama_models.clicked.connect(self._refresh_model_catalog)
        setup_lay.addWidget(self._btn_chat_sync_ollama_models)
        rv.addWidget(self._chat_setup_bar)

        self.chat_view = QTextBrowser()
        # We intercept all anchor clicks so cvops-action:// URLs run our
        # per-message handlers and external URLs open in the system browser.
        self.chat_view.setOpenExternalLinks(False)
        self.chat_view.setOpenLinks(False)
        self.chat_view.anchorClicked.connect(self._on_chat_anchor_clicked)
        self.chat_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self._composer_frame = QFrame()
        self._composer_frame.setObjectName("notesSingularityComposer")
        composer_root = QVBoxLayout(self._composer_frame)
        composer_root.setContentsMargins(10, 8, 10, 8)
        composer_root.setSpacing(6)
        composer_root.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)

        # Native text-to-speech transport for reading a model message aloud.
        # Hidden until "Play" is clicked under an assistant bubble.
        self._tts_bar = TtsPlaybackBar()
        # Speak with the saved voice profile (the "Tacitus" voice by default).
        self._tts_bar.set_voice_profile(voice_profile(load_ai_settings()))
        self._tts_bar.errorRaised.connect(self._set_chat_status)
        self._tts_bar.visibilityChanged.connect(self._resync_composer_overlay)
        composer_root.addWidget(self._tts_bar, stretch=0)

        # Push-to-talk dictation controller (mic -> local ASR -> composer text).
        self._dictation = SpeechDictationController(self)
        self._dictation.transcribed.connect(self._on_dictation_text)
        self._dictation.stateChanged.connect(self._on_dictation_state)
        self._dictation.errorRaised.connect(self._on_dictation_error)

        self.chat_input = _ComposerMessageEdit()
        self.chat_input.setObjectName("notesComposerMessage")
        self.chat_input.setPlaceholderText(
            "Write a message… use @file for context  (Ctrl+Enter to send, Enter/Shift+Enter for newline)"
        )
        self.chat_input.sendRequested.connect(self._send_chat)
        self.chat_input.pathsDropped.connect(self._on_composer_paths_dropped)
        self.chat_input.textChanged.connect(self._on_composer_text_height_changed)
        self.chat_input.setMinimumHeight(44)
        self.chat_input.setMaximumHeight(180)
        self.chat_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chat_input.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.chat_input.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        composer_root.addWidget(self.chat_input, stretch=0)

        self._composer_footer = QFrame()
        self._composer_footer.setObjectName("notesComposerFooter")
        footer_row = QHBoxLayout(self._composer_footer)
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(6)

        self._tools_strip = QFrame()
        self._tools_strip.setObjectName("notesComposerToolsStrip")
        self._tools_strip.setVisible(False)
        tools_row = QHBoxLayout(self._tools_strip)
        tools_row.setContentsMargins(0, 0, 0, 0)
        tools_row.setSpacing(6)
        self._tool_web_btn = QPushButton("Web")
        self._tool_web_btn.setCheckable(True)
        self._tool_web_btn.clicked.connect(lambda checked: self._set_tool_toggle("web", checked))
        tools_row.addWidget(self._tool_web_btn)
        self._tool_rag_btn = QPushButton("RAG")
        self._tool_rag_btn.setCheckable(True)
        self._tool_rag_btn.clicked.connect(lambda checked: self._set_tool_toggle("rag", checked))
        tools_row.addWidget(self._tool_rag_btn)
        self._tool_files_btn = QPushButton("Files")
        self._tool_files_btn.clicked.connect(self._on_tool_files_pick)
        tools_row.addWidget(self._tool_files_btn)
        self._tool_files_count = QLabel("0 file(s)")
        self._tool_files_count.setProperty("muted", True)
        repolish(self._tool_files_count)
        tools_row.addWidget(self._tool_files_count)
        self._tool_files_clear_btn = QPushButton("Clear files")
        self._tool_files_clear_btn.clicked.connect(self._on_tool_files_clear)
        tools_row.addWidget(self._tool_files_clear_btn)

        self._btn_compose_rag = QPushButton("RAG")
        self._btn_compose_rag.setObjectName("notesSingularityRagChip")
        self._btn_compose_rag.setFixedHeight(28)
        self._btn_compose_rag.setToolTip("Open the RAG tab for this project")
        self._btn_compose_rag.clicked.connect(lambda: self._tabs.setCurrentIndex(1))
        footer_row.addWidget(self._btn_compose_rag)

        self._btn_compose_tools = QPushButton("Tools")
        self._btn_compose_tools.setObjectName("notesSingularityTools")
        self._btn_compose_tools.setCheckable(True)
        self._btn_compose_tools.setFixedHeight(28)
        self._btn_compose_tools.clicked.connect(self._toggle_tools_strip)
        footer_row.addWidget(self._btn_compose_tools)

        footer_row.addWidget(self._tools_strip, stretch=0)
        footer_row.addStretch(1)

        self.chat_model = QComboBox()
        self.chat_model.setObjectName("notesSingularityModel")
        self.chat_model.setEditable(True)
        self.chat_model.setMinimumWidth(156)
        mle = self.chat_model.lineEdit()
        if mle is not None:
            mle.setPlaceholderText("Model")
        footer_row.addWidget(self.chat_model)
        self.chat_model.currentIndexChanged.connect(self._update_model_speed_badge)
        self.chat_model.currentTextChanged.connect(self._update_model_speed_badge)

        self._model_speed_badge = QLabel("Balanced")
        self._model_speed_badge.setObjectName("notesModelSpeedBadge")
        self._model_speed_badge.setMinimumWidth(54)
        self._model_speed_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_row.addWidget(self._model_speed_badge)

        # Native push-to-talk dictation: records the mic and transcribes with
        # the local ASR engine, inserting text into the composer.
        self._btn_compose_dictate = QPushButton("\U0001F3A4")  # microphone
        self._btn_compose_dictate.setObjectName("notesSingularityDictate")
        self._btn_compose_dictate.setCheckable(True)
        self._btn_compose_dictate.setFixedHeight(28)
        self._btn_compose_dictate.setMinimumWidth(34)
        self._btn_compose_dictate.setToolTip("Dictate: hold a short message, click again to stop and transcribe")
        self._btn_compose_dictate.clicked.connect(self._on_dictate_clicked)
        self._btn_compose_dictate.setEnabled(microphone_available())
        footer_row.addWidget(self._btn_compose_dictate)

        self._btn_compose_send = QPushButton("Send")
        self._btn_compose_send.setObjectName("notesSingularitySend")
        self._btn_compose_send.setProperty("isPrimary", True)
        repolish(self._btn_compose_send)
        self._btn_compose_send.setMinimumHeight(28)
        self._btn_compose_send.setMinimumWidth(68)
        self._btn_compose_send.clicked.connect(self._send_chat)
        footer_row.addWidget(self._btn_compose_send)

        self._btn_compose_stop = QPushButton("Stop")
        self._btn_compose_stop.setObjectName("notesSingularityStop")
        self._btn_compose_stop.setMinimumHeight(28)
        self._btn_compose_stop.setMinimumWidth(40)
        self._btn_compose_stop.setEnabled(False)
        self._btn_compose_stop.clicked.connect(self._stop_chat_generation)
        footer_row.addWidget(self._btn_compose_stop)

        composer_root.addWidget(self._composer_footer, stretch=0)

        self._main_stack = QStackedWidget()
        rv.addWidget(self._main_stack, stretch=1)

        chat_page = QWidget()
        chat_page_l = QVBoxLayout(chat_page)
        chat_page_l.setContentsMargins(0, 0, 0, 0)
        chat_page_l.setSpacing(0)
        chat_body = QWidget()
        chat_body_l = QHBoxLayout(chat_body)
        chat_body_l.setContentsMargins(0, 0, 0, 0)
        chat_body_l.setSpacing(0)
        self._chat_overlay_host = _OverlayChatHost(self.chat_view, self._composer_frame, chat_body)
        chat_body_l.addWidget(self._chat_overlay_host, stretch=1)
        self._chat_timeline_panel = self._build_chat_timeline_panel()
        chat_body_l.addWidget(self._chat_timeline_panel)
        chat_page_l.addWidget(chat_body, stretch=1)
        self._main_stack.addWidget(chat_page)

        project_page = QWidget()
        pp_l = QHBoxLayout(project_page)
        pp_l.setContentsMargins(0, 0, 0, 0)
        pp_l.setSpacing(8)
        project_split = QSplitter(Qt.Orientation.Horizontal)
        project_split.setChildrenCollapsible(False)
        project_split.setHandleWidth(4)
        pp_l.addWidget(project_split, stretch=1)

        chats_col = QFrame()
        chats_col.setObjectName("notesProjectChatsCard")
        chats_l = QVBoxLayout(chats_col)
        chats_l.setContentsMargins(8, 8, 8, 8)
        chats_l.setSpacing(6)
        self._project_chats_title = QLabel("Project chats")
        self._project_chats_title.setProperty("isTitle", True)
        chats_l.addWidget(self._project_chats_title)
        self._project_chats_list = _AutoFitListWidget()
        self._project_chats_list.itemDoubleClicked.connect(self._open_project_chat_from_workspace)
        chats_l.addWidget(self._project_chats_list, stretch=1)
        c_btn_row = QHBoxLayout()
        self._project_open_chat_btn = QPushButton("Open selected")
        self._project_open_chat_btn.clicked.connect(self._open_selected_project_chat_from_workspace)
        c_btn_row.addWidget(self._project_open_chat_btn)
        self._project_new_chat_btn = QPushButton("New chat")
        self._project_new_chat_btn.clicked.connect(self._new_chat_in_selected_project)
        c_btn_row.addWidget(self._project_new_chat_btn)
        c_btn_row.addStretch(1)
        chats_l.addLayout(c_btn_row)
        project_split.addWidget(chats_col)

        mem_col = QFrame()
        mem_col.setObjectName("notesProjectMemoryCard")
        mem_col.setAcceptDrops(True)
        self._project_mem_card = mem_col
        mem_l = QVBoxLayout(mem_col)
        mem_l.setContentsMargins(8, 8, 8, 8)
        mem_l.setSpacing(6)
        self._project_memory_title = QLabel("Memory & RAG files")
        self._project_memory_title.setProperty("isTitle", True)
        mem_l.addWidget(self._project_memory_title)
        mem_l.addWidget(QLabel("Instructions"))
        self._project_memory_edit = QPlainTextEdit()
        self._project_memory_edit.setPlaceholderText("Project-specific memory/instructions for this workspace.")
        self._project_memory_edit.setMinimumHeight(120)
        mem_l.addWidget(self._project_memory_edit, stretch=1)
        m_btn_row = QHBoxLayout()
        self._project_memory_save_btn = QPushButton("Save memory")
        self._project_memory_save_btn.clicked.connect(self._save_project_memory)
        m_btn_row.addWidget(self._project_memory_save_btn)
        m_btn_row.addStretch(1)
        mem_l.addLayout(m_btn_row)
        mem_l.addWidget(QLabel("Files (used by project RAG)"))
        self._project_files_list = _AutoFitListWidget()
        mem_l.addWidget(self._project_files_list, stretch=1)
        f_btn_row = QHBoxLayout()
        self._project_files_add_btn = QPushButton("Upload files")
        self._project_files_add_btn.clicked.connect(self._upload_project_files)
        f_btn_row.addWidget(self._project_files_add_btn)
        self._project_files_refresh_btn = QPushButton("Refresh")
        self._project_files_refresh_btn.clicked.connect(self._refresh_project_workspace)
        f_btn_row.addWidget(self._project_files_refresh_btn)
        f_btn_row.addStretch(1)
        mem_l.addLayout(f_btn_row)
        project_split.addWidget(mem_col)
        project_split.setSizes([420, 520])
        self._main_stack.addWidget(project_page)
        self._install_project_drop_filters()

        split.addWidget(right)
        self._events_artifacts_panel = self._build_events_artifacts_panel()
        split.addWidget(self._events_artifacts_panel)
        split.setSizes([220, 760, 300])
        layout.addWidget(split, stretch=1)

        self._apply_singularity_chrome()
        self._sync_sidebar_create_button()
        self._update_tools_strip()
        self._update_model_speed_badge()
        self._sync_chat_busy_ui()
        self._refresh_events_artifacts_panel()
        self._on_composer_text_height_changed()
        if hasattr(self, "_chat_overlay_host"):
            self._chat_overlay_host._sync_overlay_geometry()
            QTimer.singleShot(0, self._chat_overlay_host._sync_overlay_geometry)
        return w

    def _on_composer_text_height_changed(self) -> None:
        if hasattr(self, "chat_input"):
            doc_h = 0.0
            try:
                block = self.chat_input.document().firstBlock()
                while block.isValid():
                    doc_h += max(1.0, float(self.chat_input.blockBoundingRect(block).height()))
                    block = block.next()
            except Exception:
                doc_h = 0.0
            if doc_h <= 1.0:
                doc_h = float(self.chat_input.fontMetrics().lineSpacing())
            margins = self.chat_input.contentsMargins()
            target_h = int(doc_h + margins.top() + margins.bottom() + 4)
            target_h = max(44, min(180, target_h))
            self.chat_input.setFixedHeight(target_h)
        host = getattr(self, "_chat_overlay_host", None)
        if host is not None:
            host._sync_overlay_geometry()

    def _toggle_tools_strip(self, checked: bool) -> None:
        if hasattr(self, "_tools_strip"):
            self._tools_strip.setVisible(bool(checked))
        self._update_tools_strip()
        self._on_composer_text_height_changed()

    def _set_tool_toggle(self, key: str, enabled: bool) -> None:
        self._composer_tool_state[str(key)] = bool(enabled)
        self._update_tools_strip()

    def _on_tool_files_pick(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Attach files",
            str(self._space_root) if self._space_root is not None else "",
            "All Files (*)",
        )
        if not files:
            return
        existing = set(self._composer_attachments)
        for p in files:
            s = str(p or "").strip()
            if s and s not in existing:
                self._composer_attachments.append(s)
                existing.add(s)
        self._update_tools_strip()

    def _on_composer_paths_dropped(self, paths: list[str]) -> None:
        existing = set(self._composer_attachments)
        added = 0
        for raw in paths:
            s = str(raw or "").strip()
            if not s or s in existing:
                continue
            self._composer_attachments.append(s)
            existing.add(s)
            added += 1
        if added:
            self._btn_compose_tools.setChecked(True)
            self._update_tools_strip()
            self._set_chat_status(f"Attached {added} path(s). Use @dataset with run/train to launch a controlled CV Ops job.")

    def _on_tool_files_clear(self) -> None:
        self._composer_attachments = []
        self._update_tools_strip()

    def _mention_search_roots(self) -> list[tuple[str, Path]]:
        root = self._space_root
        if root is None:
            return []
        return [
            ("files", root / "files"),
            ("sessions", root / "sessions"),
        ]

    @staticmethod
    def _mention_aliases(path: Path, root: Path, label: str) -> set[str]:
        aliases = {path.name, path.stem}
        try:
            rel = path.relative_to(root)
            rel_text = rel.as_posix()
            aliases.add(rel_text)
            aliases.add(f"{label}/{rel_text}")
        except ValueError:
            pass
        return {a.lower() for a in aliases if a}

    def _mention_candidate_files(self) -> list[tuple[Path, set[str]]]:
        out: list[tuple[Path, set[str]]] = []
        seen: set[str] = set()
        for label, root in self._mention_search_roots():
            if not root.is_dir():
                continue
            try:
                files = sorted(p for p in root.rglob("*") if p.is_file())
            except OSError:
                continue
            for path in files:
                try:
                    key = str(path.resolve())
                except (OSError, ValueError):
                    key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                out.append((path, self._mention_aliases(path, root, label)))
        return out

    def _resolve_file_mention(self, raw: str) -> Optional[Path]:
        token = str(raw or "").strip().rstrip(".,!?")
        if not token:
            return None
        direct = Path(token).expanduser()
        direct_candidates: list[Path] = []
        if direct.is_absolute():
            direct_candidates.append(direct)
        else:
            if self._space_root is not None:
                direct_candidates.extend(
                    [
                        self._space_root / token,
                        self._space_root / "files" / token,
                        self._space_root / "sessions" / token,
                    ]
                )
            direct_candidates.append(Path.cwd() / token)
        for candidate in direct_candidates:
            try:
                resolved = candidate.resolve()
            except (OSError, ValueError):
                resolved = candidate
            if resolved.is_file():
                return resolved

        key = token.lower()
        matches = [path for path, aliases in self._mention_candidate_files() if key in aliases]
        if len(matches) == 1:
            return matches[0]
        exact_name = [path for path in matches if path.name.lower() == key]
        if len(exact_name) == 1:
            return exact_name[0]
        return None

    def _mentioned_file_context(self, text: str) -> tuple[list[str], str]:
        mentions = _extract_file_mentions(text)
        if not mentions:
            return [], ""
        blocks: list[str] = []
        used_labels: list[str] = []
        total = 0
        for raw in mentions[:_MENTION_MAX_FILES]:
            path = self._resolve_file_mention(raw)
            if path is None:
                continue
            try:
                excerpt, truncated = _read_text_excerpt(path)
            except OSError:
                continue
            remaining = _MENTION_MAX_TOTAL_CHARS - total
            if remaining <= 0:
                break
            if len(excerpt) > remaining:
                excerpt = excerpt[:remaining]
                truncated = True
            total += len(excerpt)
            label = raw
            try:
                if self._space_root is not None:
                    label = path.relative_to(self._space_root).as_posix()
            except ValueError:
                label = path.name
            used_labels.append(label)
            suffix = "\n[truncated]" if truncated else ""
            blocks.append(f"### @{label}\npath: {path}\n\n{excerpt}{suffix}")
        if not blocks:
            return [], ""
        return used_labels, "\n\n".join(blocks)

    def _update_tools_strip(self) -> None:
        if hasattr(self, "_tool_web_btn"):
            self._tool_web_btn.setChecked(bool(self._composer_tool_state.get("web", False)))
        if hasattr(self, "_tool_rag_btn"):
            self._tool_rag_btn.setChecked(bool(self._composer_tool_state.get("rag", False)))
        if hasattr(self, "_tool_files_count"):
            self._tool_files_count.setText(f"{len(self._composer_attachments)} file(s)")

    def _current_chat_route_parts(self) -> tuple[str, str, str]:
        route_obj = self.chat_model.currentData(Qt.ItemDataRole.UserRole)
        route_text = self.chat_model.currentText().strip()
        raw = str(route_obj) if route_obj is not None and str(route_obj).strip() else route_text
        if ":" in raw:
            provider, model = parse_route_key(raw)
        else:
            provider, model = "ollama", raw
        model_name = str(model or "").strip()
        provider_name = str(provider or "").strip().lower()
        if route_text:
            label = route_text
        elif model_name:
            label = model_name if provider_name == "ollama" else f"{model_name} · {provider_name}"
        else:
            label = provider_name or "model"
        return provider_name, model_name, label

    def _model_speed_hint(self) -> str:
        provider, model, _label = self._current_chat_route_parts()
        m = str(model or "").lower()
        p = str(provider or "").lower()
        if any(k in m for k in ("mini", "flash", "haiku", "8b", "3b", "1b")):
            return "Fast"
        if any(k in m for k in ("pro", "opus", "reason", "70b", "90b", "405b")):
            return "Deep"
        if p in {"anthropic", "openai", "gemini", "grok"}:
            return "Cloud"
        return "Balanced"

    def _update_model_speed_badge(self) -> None:
        if hasattr(self, "_model_speed_badge"):
            self._model_speed_badge.setText(self._model_speed_hint())

    def _build_rag_tab(self) -> QWidget:
        if not _rag_dependencies_available():
            w = QWidget()
            layout = QVBoxLayout(w)
            self._prepend_tab_info_row(layout, self._on_info_rag)
            layout.addWidget(
                QLabel(
                    "RAG Python dependencies are not installed. "
                    "Install with: pip install -r mlops/ChatbotAndRag/solo_rag_chat/requirements-solo.txt"
                )
            )
            self.rag_output = QTextBrowser()
            layout.addWidget(self.rag_output)
            return w

        w = _RagTabDropHost(self)
        layout = QVBoxLayout(w)
        self._prepend_tab_info_row(layout, self._on_info_rag)

        layout.setSpacing(8)

        # Active target tracked from the tree selection.
        self._rag_active_target: Optional[dict] = None
        self._rag_file_paths: List[str] = []

        # ---- Engine settings: collapsed by default, with an always-on summary ----
        engine_section = CollapsibleSection("Engine settings", expanded=False)
        eform = QFormLayout()
        eform.setContentsMargins(0, 0, 0, 0)
        eform.setVerticalSpacing(6)
        eform.setHorizontalSpacing(10)
        rag_url_wrap = QWidget()
        rag_url_row = QHBoxLayout(rag_url_wrap)
        rag_url_row.setContentsMargins(0, 0, 0, 0)
        rag_url_row.setSpacing(8)
        self.rag_ollama_url = QLineEdit("http://localhost:11434")
        self.rag_ollama_url.setMinimumWidth(160)
        rag_url_row.addWidget(self.rag_ollama_url, stretch=1)
        self._btn_rag_sync_ollama_models = QPushButton("Sync models")
        self._btn_rag_sync_ollama_models.setToolTip(
            "Reload Ollama tags from the `ollama list` CLI and GET /api/tags using this base URL."
        )
        self._btn_rag_sync_ollama_models.clicked.connect(self._rag_sync_ollama_model_lists)
        rag_url_row.addWidget(self._btn_rag_sync_ollama_models)
        eform.addRow("Ollama URL", rag_url_wrap)
        self.rag_chat_model = QComboBox()
        self.rag_chat_model.setObjectName("notesRagAnswerModel")
        self.rag_chat_model.setEditable(True)
        self.rag_chat_model.setMinimumWidth(200)
        self.rag_chat_model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        le_am = self.rag_chat_model.lineEdit()
        if le_am is not None:
            le_am.setPlaceholderText("Answer / chat model tag (e.g. gemma3:4b)")
        eform.addRow("Answer model", self.rag_chat_model)
        self.rag_embed_backend = QComboBox()
        self.rag_embed_backend.addItems(["ollama", "huggingface"])
        eform.addRow("Embedding backend", self.rag_embed_backend)
        self.rag_embed_model = QComboBox()
        self.rag_embed_model.setObjectName("notesRagEmbedModel")
        self.rag_embed_model.setEditable(True)
        self.rag_embed_model.setMinimumWidth(200)
        self.rag_embed_model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        le_em = self.rag_embed_model.lineEdit()
        if le_em is not None:
            le_em.setPlaceholderText("Embedding model (Ollama tag or HF id when backend is huggingface)")
        eform.addRow("Embedding model", self.rag_embed_model)
        # Compact, uniform input heights so the form reads as one tidy block.
        for _inp in (self.rag_ollama_url, self.rag_chat_model, self.rag_embed_backend, self.rag_embed_model):
            _inp.setMaximumHeight(28)
        engine_section.body_layout().addLayout(eform)
        layout.addWidget(engine_section)

        self._rag_engine_summary = QLabel("")
        self._rag_engine_summary.setObjectName("notesRagEngineSummary")
        self._rag_engine_summary.setProperty("muted", True)
        self._rag_engine_summary.setWordWrap(True)
        repolish(self._rag_engine_summary)
        layout.addWidget(self._rag_engine_summary)
        for _combo in (self.rag_chat_model, self.rag_embed_backend, self.rag_embed_model):
            _combo.currentTextChanged.connect(self._update_rag_engine_summary)
        self.rag_ollama_url.textChanged.connect(self._update_rag_engine_summary)

        # ---- Status strip: one digestible line of current state ----
        self._rag_status = QLabel("[IDLE]  Ready — drop a PDF/TXT/MD, pick a target, then Index.")
        self._rag_status.setObjectName("notesRagStatus")
        self._rag_status.setWordWrap(True)
        # Font-only emphasis (no colour) keeps the status strip readable across
        # every colour scheme without touching the semantic accent palette.
        self._rag_status.setStyleSheet(
            "QLabel#notesRagStatus { font-weight: 600; padding: 3px 2px; }"
        )
        layout.addWidget(self._rag_status)

        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.setChildrenCollapsible(False)

        # ---- Left: the index tree (mirrors the Database God-View browser) ----
        tree_col = QWidget()
        tcl = QVBoxLayout(tree_col)
        tcl.setContentsMargins(0, 0, 0, 0)
        tcl.setSpacing(6)
        tree_btn_row = QHBoxLayout()
        tree_lab = QLabel("Indexes")
        tree_lab.setProperty("isTitle", True)
        repolish(tree_lab)
        tree_btn_row.addWidget(tree_lab)
        tree_btn_row.addStretch(1)
        rag_refresh_btn = QPushButton("Refresh")
        rag_refresh_btn.setToolTip("Rescan the vault for the global notes index and every project index.")
        rag_refresh_btn.clicked.connect(self._populate_rag_tree)
        tree_btn_row.addWidget(rag_refresh_btn)
        tcl.addLayout(tree_btn_row)

        self._rag_tree = QTreeWidget()
        self._rag_tree.setObjectName("notesRagIndexTree")
        self._rag_tree.setHeaderLabels(["Index / Source", "Info"])
        self._rag_tree.setRootIsDecorated(True)
        self._rag_tree.setUniformRowHeights(True)
        self._rag_tree.setMinimumWidth(280)
        self._rag_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._rag_tree.itemSelectionChanged.connect(self._on_rag_tree_selection)
        self._rag_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._rag_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._rag_tree.header().setStretchLastSection(False)
        tcl.addWidget(self._rag_tree, stretch=1)
        main_split.addWidget(tree_col)

        # ---- Right: staged files + actions + query for the selected target ----
        right_col = QWidget()
        rc = QVBoxLayout(right_col)
        rc.setContentsMargins(0, 0, 0, 0)
        rc.setSpacing(6)

        target_title = QLabel("Target")
        target_title.setProperty("isTitle", True)
        repolish(target_title)
        rc.addWidget(target_title)
        self._rag_target_label = QLabel("Select an index on the left.")
        self._rag_target_label.setObjectName("notesRagTargetLabel")
        self._rag_target_label.setWordWrap(True)
        rc.addWidget(self._rag_target_label)

        stage_lab = QLabel("Staged files — drop documents anywhere on this tab, or add a path")
        stage_lab.setProperty("muted", True)
        repolish(stage_lab)
        rc.addWidget(stage_lab)

        path_row = QHBoxLayout()
        self.rag_path_entry = QLineEdit()
        self.rag_path_entry.setMaximumHeight(28)
        self.rag_path_entry.setPlaceholderText(
            "Absolute path or project-relative path — Enter or Add"
        )
        self.rag_path_entry.returnPressed.connect(self._rag_add_path_from_entry)
        path_row.addWidget(self.rag_path_entry, stretch=1)
        btn_add_path = QPushButton("Add path")
        btn_add_path.clicked.connect(self._rag_add_path_from_entry)
        path_row.addWidget(btn_add_path)
        rc.addLayout(path_row)

        stage_tools = QHBoxLayout()
        btn_proj = QPushButton("Stage target sources")
        btn_proj.setToolTip(
            "Stage every .txt, .md, and .pdf the selected target already draws from "
            "(its files/ and sessions/ folders)."
        )
        btn_proj.clicked.connect(self._rag_stage_target_sources)
        stage_tools.addWidget(btn_proj)
        btn_rem = QPushButton("Remove selected")
        btn_rem.clicked.connect(self._rag_remove_selected_docs)
        stage_tools.addWidget(btn_rem)
        btn_clr = QPushButton("Clear staged")
        btn_clr.clicked.connect(self._rag_clear_doc_queue)
        stage_tools.addWidget(btn_clr)
        stage_tools.addStretch(1)
        self._rag_doc_count_label = QLabel("0 staged")
        self._rag_doc_count_label.setProperty("muted", True)
        repolish(self._rag_doc_count_label)
        stage_tools.addWidget(self._rag_doc_count_label)
        rc.addLayout(stage_tools)

        stage_split = QSplitter(Qt.Orientation.Horizontal)
        stage_split.setChildrenCollapsible(False)
        self._rag_doc_list = _RagDocListWidget(self)
        self._rag_doc_list.pathsDropped.connect(self._rag_on_paths_dropped)
        self._rag_doc_list.currentItemChanged.connect(self._rag_on_doc_item_changed)
        stage_split.addWidget(self._rag_doc_list)
        self.rag_preview = QPlainTextEdit()
        self.rag_preview.setReadOnly(True)
        self.rag_preview.setObjectName("notesRagPreview")
        self.rag_preview.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.rag_preview.setPlaceholderText("Select a staged document to preview it here.")
        stage_split.addWidget(self.rag_preview)
        stage_split.setSizes([260, 320])
        rc.addWidget(stage_split, stretch=1)

        # Primary action emphasized; secondary maintenance actions on the next row.
        action_row = QHBoxLayout()
        index_btn = QPushButton("Index staged → target")
        index_btn.setObjectName("notesRagPrimaryAction")
        index_btn.setStyleSheet("QPushButton#notesRagPrimaryAction { font-weight: 600; }")
        index_btn.setToolTip("Rebuild the selected target's index from the staged files.")
        index_btn.clicked.connect(self._rag_index_into_target)
        action_row.addWidget(index_btn)
        add_btn = QPushButton("Add staged → target")
        add_btn.setToolTip("Append the staged files to the selected target's existing index.")
        add_btn.clicked.connect(self._rag_add_into_target)
        action_row.addWidget(add_btn)
        action_row.addStretch(1)
        rc.addLayout(action_row)

        action_row2 = QHBoxLayout()
        reindex_btn = QPushButton("Reindex from sources")
        reindex_btn.setToolTip("Rebuild the selected target's index from all of its source notes.")
        reindex_btn.clicked.connect(self._rag_reindex_target)
        action_row2.addWidget(reindex_btn)
        clear_btn = QPushButton("Clear index")
        clear_btn.clicked.connect(self._rag_clear_target)
        action_row2.addWidget(clear_btn)
        status_btn = QPushButton("Status")
        status_btn.clicked.connect(self._rag_status_target)
        action_row2.addWidget(status_btn)
        action_row2.addStretch(1)
        rc.addLayout(action_row2)

        ask_title = QLabel("Ask the selected index")
        ask_title.setProperty("isTitle", True)
        repolish(ask_title)
        rc.addWidget(ask_title)
        q_row = QHBoxLayout()
        self.rag_question = QLineEdit()
        self.rag_question.setMaximumHeight(28)
        self.rag_question.setPlaceholderText("Question for the selected target index")
        self.rag_question.returnPressed.connect(self._rag_query_target)
        q_row.addWidget(self.rag_question, stretch=1)
        q_btn = QPushButton("Query target")
        q_btn.clicked.connect(self._rag_query_target)
        q_row.addWidget(q_btn)
        rc.addLayout(q_row)

        main_split.addWidget(right_col)
        main_split.setSizes([320, 560])
        layout.addWidget(main_split, stretch=1)

        # Notes-RAG quick query keeps working programmatically even though the
        # tree is now the primary surface; methods guard on hasattr.
        self.notes_rag_question = self.rag_question

        # ---- Activity log: secondary, with a Clear control ----
        log_row = QHBoxLayout()
        log_title = QLabel("Activity log")
        log_title.setProperty("isTitle", True)
        repolish(log_title)
        log_row.addWidget(log_title)
        log_row.addStretch(1)
        log_clear_btn = QPushButton("Clear log")
        log_clear_btn.clicked.connect(lambda: self.rag_output.clear())
        log_row.addWidget(log_clear_btn)
        layout.addLayout(log_row)
        self.rag_output = QTextBrowser()
        self.rag_output.setObjectName("notesRagLog")
        self.rag_output.setMaximumHeight(150)
        layout.addWidget(self.rag_output)

        w.rag_install_child_drop_filters()
        self.rag_chat_model.setEditText("gemma3:4b")
        self.rag_embed_model.setEditText("nomic-embed-text")
        self._update_rag_engine_summary()
        QTimer.singleShot(0, self._rag_sync_ollama_model_lists)
        QTimer.singleShot(0, self._populate_rag_tree)
        return w

    def _update_rag_engine_summary(self, *_args) -> None:
        """Keep the always-visible one-line engine summary in sync with the form."""
        if not hasattr(self, "_rag_engine_summary"):
            return
        url = self.rag_ollama_url.text().strip() if hasattr(self, "rag_ollama_url") else ""
        answer = self.rag_chat_model.currentText().strip() if hasattr(self, "rag_chat_model") else ""
        backend = self.rag_embed_backend.currentText().strip() if hasattr(self, "rag_embed_backend") else ""
        embed = self.rag_embed_model.currentText().strip() if hasattr(self, "rag_embed_model") else ""
        self._rag_engine_summary.setText(
            f"Engine   answer: {answer or '—'}   ·   embed: {backend or '—'}:{embed or '—'}   ·   {url or '—'}"
        )

    def _set_rag_status(self, state: str, message: str) -> None:
        """Update the single-line status strip (state is a bracket tag, no emoji)."""
        if hasattr(self, "_rag_status"):
            self._rag_status.setText(f"[{state}]  {message}")

    def _rag_reset_queue_for_new_space(self) -> None:
        if not hasattr(self, "_rag_doc_list"):
            return
        self._rag_doc_list.clear()
        self._rag_file_paths = []
        if hasattr(self, "rag_preview"):
            self.rag_preview.clear()
        if hasattr(self, "_rag_doc_count_label"):
            self._rag_update_count_label()

    def _rag_paths_from_list(self) -> List[str]:
        if not hasattr(self, "_rag_doc_list"):
            return []
        out: List[str] = []
        for i in range(self._rag_doc_list.count()):
            it = self._rag_doc_list.item(i)
            if it is None:
                continue
            d = it.data(Qt.ItemDataRole.UserRole)
            if d:
                out.append(str(d))
        return out

    def _rag_sync_paths_from_list(self) -> None:
        self._rag_file_paths = self._rag_paths_from_list()

    def _rag_update_count_label(self) -> None:
        if hasattr(self, "_rag_doc_count_label") and hasattr(self, "_rag_doc_list"):
            self._rag_doc_count_label.setText(f"{self._rag_doc_list.count()} document(s) staged")

    def _rag_display_rel(self, path: Path) -> str:
        root = self._space_root
        if root is not None:
            try:
                return str(path.resolve().relative_to(root.resolve()))
            except ValueError:
                pass
        return path.name

    def _rag_add_paths_merged(self, raw_paths: List[str]) -> int:
        """Append unique indexable files; returns count of newly added rows."""
        if not hasattr(self, "_rag_doc_list"):
            return 0
        existing = set(self._rag_paths_from_list())
        added = 0
        for raw in raw_paths:
            raw = str(raw or "").strip()
            if not raw:
                continue
            try:
                p = Path(raw).expanduser()
                if not p.is_absolute() and self._space_root is not None:
                    cand = (self._space_root / raw).resolve()
                    if cand.is_file():
                        p = cand
                    else:
                        p = p.resolve()
                else:
                    p = p.resolve()
            except Exception:
                continue
            if not p.is_file():
                continue
            if p.suffix.lower() not in _RAG_INDEXABLE_SUFFIXES:
                continue
            ps = str(p)
            if ps in existing:
                continue
            existing.add(ps)
            it = QListWidgetItem(self._rag_display_rel(p))
            it.setData(Qt.ItemDataRole.UserRole, ps)
            it.setToolTip(ps)
            self._rag_doc_list.addItem(it)
            added += 1
        self._rag_sync_paths_from_list()
        self._rag_update_count_label()
        return added

    def _rag_replace_queue_with_paths(self, paths: List[str]) -> None:
        if not hasattr(self, "_rag_doc_list"):
            return
        self._rag_doc_list.clear()
        self._rag_file_paths = []
        if hasattr(self, "rag_preview"):
            self.rag_preview.clear()
        self._rag_add_paths_merged(list(paths))

    def _rag_remove_selected_docs(self) -> None:
        if not hasattr(self, "_rag_doc_list"):
            return
        rows = sorted({ix.row() for ix in self._rag_doc_list.selectedIndexes()}, reverse=True)
        for r in rows:
            self._rag_doc_list.takeItem(r)
        self._rag_sync_paths_from_list()
        self._rag_update_count_label()
        self._rag_on_doc_item_changed(self._rag_doc_list.currentItem(), None)

    def _rag_clear_doc_queue(self) -> None:
        if not hasattr(self, "_rag_doc_list"):
            return
        self._rag_doc_list.clear()
        self._rag_file_paths = []
        self._rag_update_count_label()
        if hasattr(self, "rag_preview"):
            self.rag_preview.clear()
        if hasattr(self, "rag_output"):
            self.rag_output.append("[RAG] Cleared the document queue.")

    def _rag_add_from_project_library(self) -> None:
        if not hasattr(self, "rag_output"):
            return
        paths = self._collect_space_index_paths()
        if not paths:
            self.rag_output.append("[RAG] No .txt / .md / .pdf under files/ or sessions/.")
            return
        n = self._rag_add_paths_merged(paths)
        self.rag_output.append(
            f"[RAG] Merged project library: {n} new file(s) added ({len(paths)} scanned)."
        )

    def _rag_add_path_from_entry(self) -> None:
        if not hasattr(self, "rag_path_entry") or not hasattr(self, "rag_output"):
            return
        txt = self.rag_path_entry.text().strip()
        if not txt:
            return
        n = self._rag_add_paths_merged([txt])
        if n:
            self.rag_path_entry.clear()
            self.rag_output.append("[RAG] Added path to the queue.")
        else:
            self.rag_output.append(
                f"[RAG] Could not add {txt!r}. Use an existing .txt, .md, or .pdf "
                "(absolute path or path relative to the project folder)."
            )

    def _rag_on_paths_dropped(self, paths: List[str]) -> None:
        if not hasattr(self, "rag_output"):
            return
        n = self._rag_add_paths_merged(paths)
        self.rag_output.append(f"[RAG] Drop accepted: {n} new file(s) added to the queue.")

    def _rag_on_doc_item_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        if not hasattr(self, "rag_preview"):
            return
        if not current:
            self.rag_preview.clear()
            return
        raw = current.data(Qt.ItemDataRole.UserRole)
        if not raw:
            self.rag_preview.clear()
            return
        self._rag_fill_preview(Path(str(raw)))

    def _rag_pdf_preview_text(self, path: Path) -> str:
        try:
            import pymupdf
        except Exception:
            try:
                import fitz as pymupdf  # type: ignore[no-redef,import-not-found]
            except Exception:
                return "PDF preview needs PyMuPDF (pymupdf), included with the RAG dependencies."
        try:
            doc = pymupdf.open(path)
            try:
                if doc.page_count < 1:
                    return "(PDF has no pages)"
                text = doc.load_page(0).get_text()
                return text if text.strip() else "(No extractable text on page 1)"
            finally:
                doc.close()
        except Exception as exc:
            return f"Could not read PDF: {exc}"

    def _rag_fill_preview(self, path: Path) -> None:
        if not hasattr(self, "rag_preview"):
            return
        self.rag_preview.clear()
        if not path.is_file():
            self.rag_preview.setPlainText("(file not found)")
            return
        suf = path.suffix.lower()
        try:
            if suf in (".txt", ".md", ".markdown"):
                raw = path.read_text(encoding="utf-8", errors="replace")
                if len(raw) > 200_000:
                    raw = raw[:200_000] + "\n\n[Preview truncated]"
                self.rag_preview.setPlainText(raw)
            elif suf == ".pdf":
                body = self._rag_pdf_preview_text(path)
                if len(body) > 120_000:
                    body = body[:120_000] + "\n\n[Preview truncated]"
                self.rag_preview.setPlainText(body)
            else:
                self.rag_preview.setPlainText("(no preview for this file type)")
        except Exception as exc:
            self.rag_preview.setPlainText(f"Preview error: {exc}")

    def _space_or_warn(self) -> Optional[Path]:
        if self._space_root is None:
            QMessageBox.warning(self, "Notes AI", "No active notes project.")
            return None
        return self._space_root

    def _apply_rag_config(self) -> None:
        root = self._space_or_warn()
        if root is None:
            return
        _apply_rag_config_to_space(space_root=root, **self._notes_rag_engine_params())

    def _collect_space_index_paths(self) -> List[str]:
        root = self._space_root
        if root is None:
            return []
        out: List[str] = []
        for sub in ("files", "sessions"):
            base = root / sub
            if not base.is_dir():
                continue
            try:
                for p in sorted(base.rglob("*")):
                    if p.is_file() and p.suffix.lower() in _RAG_INDEXABLE_SUFFIXES:
                        out.append(str(p))
            except Exception:
                continue
        return out

    def _rag_index_this_space(self) -> None:
        if not _rag_dependencies_available():
            return
        paths = self._collect_space_index_paths()
        if not paths:
            if hasattr(self, "rag_output"):
                self.rag_output.append(
                    "[RAG] No indexable files under this project's files/ or sessions/."
                )
            return
        self._rag_replace_queue_with_paths(paths)
        if hasattr(self, "rag_output"):
            self.rag_output.append(
                f"[RAG] Queue replaced with {len(paths)} project document(s); building index…"
            )
        self._rag_build()

    def _rag_build(self) -> None:
        if not _rag_dependencies_available():
            return
        self._rag_sync_paths_from_list()
        if not self._rag_file_paths:
            if hasattr(self, "rag_output"):
                self.rag_output.append(
                    "[RAG] Add documents to the queue (library, path, or drop) before building."
                )
            return
        self._apply_rag_config()
        self.rag_output.append("[RAG] Building index…")
        self._start_rag_worker(RAGWorker("build", files=self._rag_file_paths))

    def _rag_clear(self) -> None:
        if not _rag_dependencies_available():
            return
        self._apply_rag_config()
        self._start_rag_worker(RAGWorker("clear"))

    def _rag_status(self) -> None:
        if not _rag_dependencies_available():
            return
        self._apply_rag_config()
        self._start_rag_worker(RAGWorker("status"))

    def _rag_query(self) -> None:
        if not _rag_dependencies_available():
            return
        q = self.rag_question.text().strip()
        if not q:
            return
        self._apply_rag_config()
        self._start_rag_worker(
            RAGWorker("query", question=q, k=4, return_sources=True),
            on_query=True,
        )

    def _start_rag_worker(self, worker: RAGWorker, *, on_query: bool = False) -> None:
        if self._rag_worker is not None and self._rag_worker.isRunning():
            if hasattr(self, "rag_output"):
                self.rag_output.append("[RAG] Wait for the current RAG task to finish.")
            return
        self._rag_worker = worker
        if on_query:
            worker.finished.connect(self._on_rag_query_finished)
        else:
            worker.finished.connect(self._on_rag_finished)
        worker.progress.connect(self._on_rag_progress)
        worker.error.connect(self._on_rag_error)
        worker.start()

    def _on_rag_progress(self, message: str) -> None:
        self._set_rag_status("WORKING", message)
        if hasattr(self, "rag_output"):
            self.rag_output.append(f"[RAG] {message}")

    def _on_rag_finished(self, payload: dict) -> None:
        msg = payload.get("message") if isinstance(payload, dict) else None
        self._set_rag_status("DONE", str(msg) if msg else "Operation complete.")
        self.rag_output.append(f"[RAG] {msg}" if msg else str(payload))
        # An index may have just been created/cleared — refresh the tree status.
        if hasattr(self, "_rag_tree"):
            self._populate_rag_tree()

    def _on_rag_query_finished(self, payload: dict) -> None:
        ans = payload.get("answer", "")
        self._set_rag_status("DONE", "Answer ready.")
        self.rag_output.append(f"Answer:\n{ans}\n")
        if payload.get("sources"):
            self.rag_output.append(f"Sources: {payload.get('sources')}\n")

    def _on_rag_error(self, msg: str) -> None:
        self._set_rag_status("ERROR", str(msg))
        self.rag_output.append(f"[ERROR] {msg}")
        self.errorRaised.emit(f"Notes RAG: {msg}")

    # -- Global notes RAG (secondary, vault-wide) ----------------------------

    def _notes_rag_engine_params(self) -> dict:
        """Engine settings shared with the chat RAG, read from the RAG tab combos."""
        model_id = "gemma3:4b"
        embedding_backend = "ollama"
        embedding_model = ""
        ollama_base_url = "http://localhost:11434"
        if hasattr(self, "rag_chat_model"):
            model_id = self.rag_chat_model.currentText().strip() or model_id
        if hasattr(self, "rag_embed_backend"):
            embedding_backend = self.rag_embed_backend.currentText().strip().lower() or embedding_backend
        if hasattr(self, "rag_embed_model"):
            embedding_model = self.rag_embed_model.currentText().strip() or embedding_model
        if hasattr(self, "rag_ollama_url"):
            ollama_base_url = self.rag_ollama_url.text().strip() or ollama_base_url
        embedding_model = self._resolve_rag_embedding_model(
            embedding_backend,
            embedding_model,
            ollama_base_url,
            update_combo=True,
            announce=True,
        )
        if embedding_backend != "ollama" and not embedding_model:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        return {
            "model_id": model_id,
            "embedding_backend": embedding_backend,
            "embedding_model": embedding_model,
            "ollama_base_url": ollama_base_url,
        }

    def _ensure_notes_rag_config(self, *, reset: bool = False) -> Optional[Path]:
        """Configure the notes-RAG namespace against the vault root. Returns the vault root or None."""
        if self._space_root is None:
            return None
        vault_root = _notes_vault_root_from_space(self._space_root)
        _set_notes_rag_config(
            vault_root=vault_root,
            reset=reset,
            **self._notes_rag_engine_params(),
        )
        return vault_root

    def _collect_vault_notes_paths(self) -> List[str]:
        """Every indexable note (files/ + sessions/) across all project spaces in the vault."""
        if self._space_root is None:
            return []
        spaces_root = self._space_root.parent
        out: List[str] = []
        seen: set[str] = set()
        try:
            space_dirs = [p for p in spaces_root.iterdir() if p.is_dir()]
        except Exception:
            return []
        for space in space_dirs:
            for sub in ("files", "sessions"):
                base = space / sub
                if not base.is_dir():
                    continue
                try:
                    for p in sorted(base.rglob("*")):
                        if not (p.is_file() and p.suffix.lower() in _RAG_INDEXABLE_SUFFIXES):
                            continue
                        key = str(p.resolve())
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(str(p))
                except Exception:
                    continue
        return out

    def notes_rag_add_paths(self, paths: List[str]) -> None:
        """Auto-index hook: append freshly uploaded notes to the global notes RAG index."""
        if not _rag_dependencies_available():
            return
        indexable = [
            str(Path(p))
            for p in (paths or [])
            if str(p).strip() and Path(p).suffix.lower() in _RAG_INDEXABLE_SUFFIXES
        ]
        if not indexable:
            return
        if self._ensure_notes_rag_config() is None:
            return
        self._start_notes_rag_worker(
            RAGWorker("add", namespace=_NOTES_NAMESPACE, files=indexable)
        )

    def notes_rag_reindex(self) -> None:
        """Rebuild the global notes RAG index from every note in the vault."""
        if not _rag_dependencies_available():
            return
        paths = self._collect_vault_notes_paths()
        if not paths:
            if hasattr(self, "rag_output"):
                self.rag_output.append("[NOTES RAG] No indexable notes found across the vault.")
            return
        if self._ensure_notes_rag_config(reset=True) is None:
            if hasattr(self, "rag_output"):
                self.rag_output.append("[NOTES RAG] Open a project first.")
            return
        if hasattr(self, "rag_output"):
            self.rag_output.append(
                f"[NOTES RAG] Rebuilding global notes index from {len(paths)} note(s)…"
            )
        self._start_notes_rag_worker(
            RAGWorker("build", namespace=_NOTES_NAMESPACE, files=paths)
        )

    def _notes_rag_query(self) -> None:
        if not _rag_dependencies_available():
            return
        if not hasattr(self, "notes_rag_question"):
            return
        q = self.notes_rag_question.text().strip()
        if not q:
            return
        if self._ensure_notes_rag_config() is None:
            self.rag_output.append("[NOTES RAG] Open a project first.")
            return
        self.rag_output.append(f"[NOTES RAG] Q: {q}")
        self._start_notes_rag_worker(
            RAGWorker("query", namespace=_NOTES_NAMESPACE, question=q, k=4, return_sources=True),
            on_query=True,
        )

    def _start_notes_rag_worker(self, worker: RAGWorker, *, on_query: bool = False) -> None:
        if self._notes_rag_worker is not None and self._notes_rag_worker.isRunning():
            if hasattr(self, "rag_output"):
                self.rag_output.append("[NOTES RAG] Wait for the current notes-RAG task to finish.")
            return
        self._notes_rag_worker = worker
        if on_query:
            worker.finished.connect(self._on_notes_rag_query_finished)
        else:
            worker.finished.connect(self._on_notes_rag_finished)
        worker.progress.connect(self._on_notes_rag_progress)
        worker.error.connect(self._on_notes_rag_error)
        worker.start()

    def _on_notes_rag_progress(self, message: str) -> None:
        self._set_rag_status("WORKING", f"notes · {message}")
        if hasattr(self, "rag_output"):
            self.rag_output.append(f"[NOTES RAG] {message}")

    def _on_notes_rag_finished(self, payload: dict) -> None:
        msg = payload.get("message", payload) if isinstance(payload, dict) else payload
        self._set_rag_status("DONE", f"notes · {msg}")
        if hasattr(self, "rag_output"):
            self.rag_output.append(f"[NOTES RAG] {msg}")
        if hasattr(self, "_rag_tree"):
            self._populate_rag_tree()

    def _on_notes_rag_query_finished(self, payload: dict) -> None:
        if not hasattr(self, "rag_output"):
            return
        ans = payload.get("answer", "")
        self.rag_output.append(f"[NOTES RAG] Answer:\n{ans}\n")
        if payload.get("sources"):
            self.rag_output.append(f"[NOTES RAG] Sources: {payload.get('sources')}\n")

    def _on_notes_rag_error(self, msg: str) -> None:
        if hasattr(self, "rag_output"):
            self.rag_output.append(f"[NOTES RAG][ERROR] {msg}")
        self.errorRaised.emit(f"Notes RAG (global): {msg}")

    # -- Ingested memory (knowledge transferred from other AIs) --------------

    def _ensure_ingested_memory_rag_config(self, *, reset: bool = False) -> Optional[Path]:
        """Configure the ingested-memory namespace against the vault root."""
        if self._space_root is None:
            return None
        vault_root = _notes_vault_root_from_space(self._space_root)
        _set_ingested_memory_rag_config(
            vault_root=vault_root,
            reset=reset,
            **self._notes_rag_engine_params(),
        )
        return vault_root

    def _start_memory_rag_worker(self, worker: RAGWorker) -> None:
        if self._memory_rag_worker is not None and self._memory_rag_worker.isRunning():
            if hasattr(self, "rag_output"):
                self.rag_output.append("[MEMORY RAG] Wait for the current memory task to finish.")
            return
        self._memory_rag_worker = worker
        worker.finished.connect(self._on_memory_rag_finished)
        worker.progress.connect(lambda m: self._set_rag_status("WORKING", f"memory · {m}"))
        worker.error.connect(self._on_memory_rag_error)
        worker.start()

    def _on_memory_rag_finished(self, payload: dict) -> None:
        msg = payload.get("message", payload) if isinstance(payload, dict) else payload
        self._set_rag_status("DONE", f"memory · {msg}")
        if hasattr(self, "rag_output"):
            self.rag_output.append(f"[MEMORY RAG] {msg}")

    def _on_memory_rag_error(self, msg: str) -> None:
        if hasattr(self, "rag_output"):
            self.rag_output.append(f"[MEMORY RAG][ERROR] {msg}")
        self.errorRaised.emit(f"Ingested memory RAG: {msg}")

    def _index_ingested_memory(self, doc_paths: List[str], *, rebuild: bool = False) -> None:
        """Add fresh memory docs (or rebuild the whole namespace) into FAISS."""
        if not _rag_dependencies_available():
            return
        if self._ensure_ingested_memory_rag_config(reset=rebuild) is None:
            return
        if rebuild:
            paths = [p for p in doc_paths if str(p).strip()]
            if not paths:
                self._start_memory_rag_worker(
                    RAGWorker("clear", namespace=INGESTED_MEMORY_NAMESPACE)
                )
            else:
                self._start_memory_rag_worker(
                    RAGWorker("build", namespace=INGESTED_MEMORY_NAMESPACE, files=paths)
                )
            return
        indexable = [str(p) for p in (doc_paths or []) if str(p).strip()]
        if not indexable:
            return
        self._start_memory_rag_worker(
            RAGWorker("add", namespace=INGESTED_MEMORY_NAMESPACE, files=indexable)
        )

    def _rebuild_ingested_memory_index(self) -> None:
        """Rebuild the ingested-memory namespace from whatever docs remain on disk."""
        if self._space_root is None:
            return
        vault_root = _notes_vault_root_from_space(self._space_root)
        self._index_ingested_memory(ai_memory.memory_doc_paths(vault_root), rebuild=True)

    def _ingested_memory_context_block(self, query: str) -> str:
        """Best-effort retrieval of transferred knowledge for the outbound message.

        Returns a short ``[transferred knowledge]`` block (or "") to splice into
        the prompt. Retrieval-only (no LLM call); guarded so a missing index or
        any failure simply yields no extra context.
        """
        q = str(query or "").strip()
        if not q or self._space_root is None or not _rag_dependencies_available():
            return ""
        vault_root = _notes_vault_root_from_space(self._space_root)
        if not ai_memory.ingested_memory_index_dir(vault_root).exists():
            return ""
        if self._ensure_ingested_memory_rag_config() is None:
            return ""
        try:
            import asyncio

            from mlops.ChatbotAndRag.solo_rag_chat.rag_system import get_rag_system

            async def _retrieve() -> list[str]:
                rag = await get_rag_system(INGESTED_MEMORY_NAMESPACE)
                if rag is None:
                    return []
                if getattr(rag, "vectorstore", None) is None:
                    await rag.load_index()
                if getattr(rag, "vectorstore", None) is None:
                    return []
                docs = rag.vectorstore.similarity_search(q, k=3)
                out: list[str] = []
                for d in docs:
                    txt = str(getattr(d, "page_content", "") or "").strip()
                    if txt:
                        out.append(txt[:800])
                return out

            snippets = asyncio.run(_retrieve())
        except Exception:
            return ""
        if not snippets:
            return ""
        return "[transferred knowledge]\n" + "\n---\n".join(snippets)

    # -- RAG database tree (every index in the vault) ------------------------

    def _rag_targets(self) -> List[dict]:
        """Discover every RAG index in the vault: the global notes index + one per project."""
        targets: List[dict] = []
        if self._space_root is None:
            return targets
        spaces_root = self._space_root.parent
        vault_root = _notes_vault_root_from_space(self._space_root)
        active_id = self._space_root.name

        try:
            space_ids = list_space_ids(spaces_root)
        except Exception:
            space_ids = []
        if not space_ids:
            space_ids = [active_id]

        # Global notes RAG (secondary, vault-wide): sources are every project's notes.
        notes_sources: List[Path] = []
        for sid in space_ids:
            sdir = spaces_root / sid
            notes_sources.extend([sdir / "files", sdir / "sessions"])
        targets.append({
            "key": "notes",
            "label": "Global notes RAG",
            "kind": "notes",
            "namespace": _NOTES_NAMESPACE,
            "index_path": vault_root / NOTES_RAG_INDEX_DIRNAME,
            "source_dirs": notes_sources,
            "space_id": None,
        })

        # Per-project RAG indexes.
        for sid in space_ids:
            sdir = spaces_root / sid
            try:
                title = read_space_title(sdir, sid)
            except Exception:
                title = sid
            label = f"{title}" if title else sid
            if sid == active_id:
                label += "  [active]"
            targets.append({
                "key": f"project:{sid}",
                "label": label,
                "kind": "project",
                "namespace": f"project:{sid}",
                "index_path": sdir / "rag_index",
                "source_dirs": [sdir / "files", sdir / "sessions"],
                "space_id": sid,
            })
        return targets

    @staticmethod
    def _rag_index_exists(index_path: Path) -> bool:
        try:
            return (index_path / "index.faiss").is_file() or (index_path / "index.pkl").is_file()
        except Exception:
            return False

    def _rag_collect_sources(self, source_dirs: List[Path]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for base in source_dirs:
            if not base.is_dir():
                continue
            try:
                for p in sorted(base.rglob("*")):
                    if not (p.is_file() and p.suffix.lower() in _RAG_INDEXABLE_SUFFIXES):
                        continue
                    key = str(p.resolve())
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(str(p))
            except Exception:
                continue
        return out

    def _populate_rag_tree(self) -> None:
        if not hasattr(self, "_rag_tree"):
            return
        prior_key = None
        if self._rag_active_target:
            prior_key = self._rag_active_target.get("key")
        self._rag_tree.blockSignals(True)
        self._rag_tree.clear()
        targets = self._rag_targets()

        reselect_item: Optional[QTreeWidgetItem] = None
        projects_group: Optional[QTreeWidgetItem] = None
        proj_indexed = 0
        proj_total = 0

        for tgt in targets:
            sources = self._rag_collect_sources(tgt["source_dirs"])
            exists = self._rag_index_exists(tgt["index_path"])
            info = f"[INDEXED] {len(sources)} docs" if exists else f"[EMPTY] {len(sources)} docs"
            item = QTreeWidgetItem([tgt["label"], info])
            item.setData(0, Qt.ItemDataRole.UserRole, tgt)
            item.setToolTip(0, str(tgt["index_path"]))

            if tgt["kind"] == "notes":
                # Global notes index sits at the top level (it is the secondary RAG).
                self._rag_tree.addTopLevelItem(item)
            else:
                if projects_group is None:
                    projects_group = QTreeWidgetItem(["Projects", ""])
                    projects_group.setFirstColumnSpanned(True)
                    self._rag_tree.addTopLevelItem(projects_group)
                projects_group.addChild(item)
                proj_total += 1
                proj_indexed += 1 if exists else 0

            # Source-document leaves so the full data structure is visible.
            for sp in sources:
                suffix = Path(sp).suffix.lower().lstrip(".") or "file"
                leaf = QTreeWidgetItem([Path(sp).name, suffix.upper()])
                leaf.setToolTip(0, sp)
                item.addChild(leaf)
            if prior_key is not None and tgt["key"] == prior_key:
                reselect_item = item

        if projects_group is not None:
            projects_group.setText(1, f"{proj_indexed}/{proj_total} indexed")
            projects_group.setExpanded(True)
        self._rag_tree.blockSignals(False)
        if reselect_item is not None:
            reselect_item.setSelected(True)
            self._rag_tree.setCurrentItem(reselect_item)
        self._on_rag_tree_selection()

    def _rag_selected_target(self) -> Optional[dict]:
        items = self._rag_tree.selectedItems() if hasattr(self, "_rag_tree") else []
        for it in items:
            data = it.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, dict):
                return data
            # Allow selecting a source leaf: fall back to its parent target.
            parent = it.parent()
            if parent is not None:
                pdata = parent.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(pdata, dict):
                    return pdata
        return None

    def _on_rag_tree_selection(self) -> None:
        target = self._rag_selected_target()
        self._rag_active_target = target
        if not hasattr(self, "_rag_target_label"):
            return
        if target is None:
            self._rag_target_label.setText("Select an index on the left.")
            return
        kind = "Global notes RAG" if target["kind"] == "notes" else "Project RAG"
        self._rag_target_label.setText(
            f"{target['label']}  ·  {kind}\n{target['index_path']}"
        )

    def _rag_current_target_or_warn(self) -> Optional[dict]:
        target = self._rag_active_target or self._rag_selected_target()
        if target is None:
            if hasattr(self, "rag_output"):
                self.rag_output.append("[RAG] Select a target index in the tree first.")
            return None
        return target

    def _apply_target_rag_config(self, target: dict, *, reset: bool = False) -> None:
        """Point ``target['namespace']`` at its index path; engine config is shared."""
        from mlops.ChatbotAndRag.solo_rag_chat.rag_system import (
            reset_rag_system,
            set_rag_config,
        )

        if reset:
            reset_rag_system(target["namespace"])
        set_rag_config(
            namespace=target["namespace"],
            rag_index_path=str(target["index_path"]),
            **self._notes_rag_engine_params(),
        )

    def _rag_stage_target_sources(self) -> None:
        target = self._rag_current_target_or_warn()
        if target is None:
            return
        paths = self._rag_collect_sources(target["source_dirs"])
        if not paths:
            self.rag_output.append(f"[RAG] {target['label']}: no indexable source notes found.")
            return
        self._rag_replace_queue_with_paths(paths)
        self.rag_output.append(f"[RAG] Staged {len(paths)} source doc(s) from {target['label']}.")

    def _rag_index_into_target(self) -> None:
        self._rag_run_into_target("build")

    def _rag_add_into_target(self) -> None:
        self._rag_run_into_target("add")

    def _rag_run_into_target(self, action: str) -> None:
        if not _rag_dependencies_available():
            return
        target = self._rag_current_target_or_warn()
        if target is None:
            return
        self._rag_sync_paths_from_list()
        if not self._rag_file_paths:
            self.rag_output.append("[RAG] Stage documents (drop, path, or sources) before indexing.")
            return
        self._apply_target_rag_config(target, reset=(action == "build"))
        verb = "Building" if action == "build" else "Adding to"
        self.rag_output.append(f"[RAG] {verb} index for {target['label']}…")
        self._start_rag_worker(
            RAGWorker(action, namespace=target["namespace"], files=list(self._rag_file_paths))
        )

    def _rag_reindex_target(self) -> None:
        if not _rag_dependencies_available():
            return
        target = self._rag_current_target_or_warn()
        if target is None:
            return
        paths = self._rag_collect_sources(target["source_dirs"])
        if not paths:
            self.rag_output.append(f"[RAG] {target['label']}: no source notes to reindex.")
            return
        self._apply_target_rag_config(target, reset=True)
        self.rag_output.append(f"[RAG] Reindexing {target['label']} from {len(paths)} source doc(s)…")
        self._start_rag_worker(RAGWorker("build", namespace=target["namespace"], files=paths))

    def _rag_clear_target(self) -> None:
        if not _rag_dependencies_available():
            return
        target = self._rag_current_target_or_warn()
        if target is None:
            return
        self._apply_target_rag_config(target)
        self.rag_output.append(f"[RAG] Clearing index for {target['label']}…")
        self._start_rag_worker(RAGWorker("clear", namespace=target["namespace"]))

    def _rag_status_target(self) -> None:
        if not _rag_dependencies_available():
            return
        target = self._rag_current_target_or_warn()
        if target is None:
            return
        self._apply_target_rag_config(target)
        self._start_rag_worker(RAGWorker("status", namespace=target["namespace"]))

    def _rag_query_target(self) -> None:
        if not _rag_dependencies_available():
            return
        target = self._rag_current_target_or_warn()
        if target is None:
            return
        q = self.rag_question.text().strip()
        if not q:
            return
        self._apply_target_rag_config(target)
        self.rag_output.append(f"[RAG] {target['label']} Q: {q}")
        self._start_rag_worker(
            RAGWorker("query", namespace=target["namespace"], question=q, k=4, return_sources=True),
            on_query=True,
        )

    def _activity_spinner_text(self) -> str:
        return _ACTIVITY_FRAMES[self._activity_frame_index % len(_ACTIVITY_FRAMES)]

    def _advance_activity_indicators(self) -> None:
        self._activity_frame_index = (self._activity_frame_index + 1) % len(_ACTIVITY_FRAMES)
        active = False
        for listw in (
            getattr(self, "chat_list_pinned", None),
            getattr(self, "chat_list_recent", None),
            getattr(self, "_chat_jobs_list", None),
            getattr(self, "_project_events_list", None),
        ):
            if listw is None:
                continue
            for i in range(listw.count()):
                item = listw.item(i)
                if item is None:
                    continue
                host = listw.itemWidget(item)
                if hasattr(host, "has_active_jobs") and host.has_active_jobs():
                    host.update_activity_icon()
                    active = True
                    continue
                data = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(data, dict) and bool(data.get("activity_active")):
                    base = str(data.get("activity_base_text") or item.text()).strip()
                    item.setText(f"{self._activity_spinner_text()} {base}")
                    active = True
        if not active and self._activity_timer.isActive():
            self._activity_timer.stop()

    def _sync_activity_timer(self) -> None:
        active = False
        for listw in (
            getattr(self, "chat_list_pinned", None),
            getattr(self, "chat_list_recent", None),
            getattr(self, "_chat_jobs_list", None),
            getattr(self, "_project_events_list", None),
        ):
            if listw is None:
                continue
            for i in range(listw.count()):
                item = listw.item(i)
                if item is None:
                    continue
                host = listw.itemWidget(item)
                if hasattr(host, "has_active_jobs") and host.has_active_jobs():
                    host.update_activity_icon()
                    active = True
                    continue
                data = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(data, dict) and bool(data.get("activity_active")):
                    base = str(data.get("activity_base_text") or item.text()).strip()
                    item.setText(f"{self._activity_spinner_text()} {base}")
                    active = True
        if active and not self._activity_timer.isActive():
            self._activity_timer.start()
        elif not active and self._activity_timer.isActive():
            self._activity_timer.stop()

    def _chat_meta_jobs(self, meta: dict[str, Any]) -> list[dict[str, Any]]:
        direct = meta.get(_CHAT_JOBS_METADATA_KEY)
        if direct:
            return _coerce_record_list(direct)
        nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        return _coerce_record_list(nested.get(_CHAT_JOBS_METADATA_KEY))

    def _measure_job_record(self, record: dict[str, Any]) -> dict[str, Any]:
        out = dict(record or {})
        job_id = str(out.get("job_id") or "").strip()
        if not job_id or not _job_state_is_active(out.get("state")):
            return out
        now = time.monotonic()
        cached = self._job_state_cache.get(job_id)
        if cached is not None and now - float(cached[0]) < 0.75:
            measured = dict(cached[1])
        else:
            try:
                measured = self._cvops_http_json("GET", f"/jobs/{job_id}")
                self._job_state_cache[job_id] = (now, dict(measured))
            except Exception:
                measured = {}
        if measured:
            for key in ("state", "scenario", "job_type", "source"):
                value = str(measured.get(key) or "").strip()
                if value:
                    out[key] = value
            out["measured_at"] = _utc_now_iso()
        return out

    def _measured_chat_jobs(self, chat: dict[str, Any], mgr: ChatManager, chat_id: str) -> list[dict[str, Any]]:
        meta = chat.setdefault("metadata", {})
        if not isinstance(meta, dict):
            meta = {}
            chat["metadata"] = meta
        jobs = _coerce_record_list(meta.get(_CHAT_JOBS_METADATA_KEY))
        if not jobs:
            return []
        measured = [self._measure_job_record(item) for item in jobs]
        if measured != jobs:
            meta[_CHAT_JOBS_METADATA_KEY] = measured
            try:
                mgr.save_chat(chat_id)
            except Exception:
                pass
        return measured

    def _chat_activity_summary(self, meta: dict[str, Any]) -> tuple[int, str]:
        jobs = self._chat_meta_jobs(meta)
        active_jobs = [item for item in jobs if _job_state_is_active(item.get("state"))]
        count = len(active_jobs)
        cid = str(meta.get("id") or "").strip()
        if cid and cid == str(self._current_chat_id or "") and self._is_streaming:
            count += 1
        if count <= 0:
            return 0, ""
        labels = []
        for item in active_jobs[:3]:
            job_id = str(item.get("job_id") or "").strip()
            state = str(item.get("state") or "running").strip()
            scenario = str(item.get("scenario") or "").strip()
            labels.append(" ".join(part for part in (job_id, scenario, state) if part))
        if cid == str(self._current_chat_id or "") and self._is_streaming:
            labels.append("assistant reply streaming")
        return count, "\n".join(labels) or f"{count} active operation(s)"

    def _refresh_chat_list(self) -> None:
        if not hasattr(self, "chat_list_recent"):
            return
        for lw in (self.chat_list_pinned, self.chat_list_recent):
            lw.blockSignals(True)
            lw.clear()
        if self._space_root is None:
            for lw in (self.chat_list_pinned, self.chat_list_recent):
                lw.blockSignals(False)
            self._chat_pinned_header.setVisible(False)
            self.chat_list_pinned.setVisible(False)
            return
        spaces_root = self._spaces_root()
        vault = self._space_root.parent.parent
        pinned_projects = [(r[0], r[1], r[2]) for r in self._project_rows if r[3]]
        pinned_meta: list[tuple[str, str, dict]] = []
        recent_meta: list[tuple[str, str, dict]] = []
        if spaces_root is not None:
            for sid in list_space_ids(spaces_root):
                mgr = ChatManager(chats_dir=notes_chats_dir(vault, sid))
                ptitle = self._project_title_for_space(sid)
                for meta in mgr.list_chats():
                    meta = dict(meta)
                    cid = str(meta.get("id") or "").strip()
                    chat = mgr.chats.get(cid) if cid else None
                    if isinstance(chat, dict):
                        meta["metadata"] = dict(chat.get("metadata") or {})
                        meta[_CHAT_JOBS_METADATA_KEY] = self._measured_chat_jobs(chat, mgr, cid)
                    row = (sid, ptitle, meta)
                    if meta.get("pinned"):
                        pinned_meta.append(row)
                    else:
                        recent_meta.append(row)
        show_pinned = bool(pinned_projects or pinned_meta)
        self._chat_pinned_header.setVisible(show_pinned)
        self.chat_list_pinned.setVisible(show_pinned)
        for sid, title, goals in pinned_projects:
            self._add_project_list_row(self.chat_list_pinned, sid, title, goals)
        for sid, ptitle, meta in pinned_meta:
            self._add_chat_list_row(self.chat_list_pinned, sid, ptitle, meta)
        for sid, ptitle, meta in recent_meta:
            self._add_chat_list_row(self.chat_list_recent, sid, ptitle, meta)
        for lw in (self.chat_list_pinned, self.chat_list_recent):
            lw.blockSignals(False)
        self._sync_pinned_list_height(self.chat_list_pinned, max_rows=3)
        self._sync_activity_timer()

    def _project_title_for_space(self, sid: str) -> str:
        for row_sid, row_title, _goals, _pin in self._project_rows:
            if str(row_sid) == str(sid):
                return str(row_title or sid)
        sdir = self._space_dir(sid)
        if sdir is not None:
            return read_space_title(sdir, sid)
        return sid

    @staticmethod
    def _encode_chat_ref(space_id: str, chat_id: str) -> str:
        return f"{space_id}::{chat_id}"

    @staticmethod
    def _decode_chat_ref(raw: object) -> tuple[str, str]:
        text = str(raw or "")
        if "::" not in text:
            return "", text.strip()
        sid, cid = text.split("::", 1)
        return sid.strip(), cid.strip()

    @staticmethod
    def _ingested_tag(import_source: object) -> str:
        """Bracket tag marking a row as ingested from another AI (never created here)."""
        src = str(import_source or "").strip().lower()
        label = {"chatgpt": "CHATGPT", "claude": "CLAUDE"}.get(src, "INGESTED")
        return f"[{label}] "

    def _add_chat_list_row(self, listw: QListWidget, space_id: str, project_title: str, meta: dict) -> None:
        cid = str(meta.get("id", "")).strip()
        if not cid:
            return
        title = str(meta.get("title", cid))
        tag = self._ingested_tag(meta.get("import_source")) if meta.get("imported") else ""
        shown = f"{tag}[{project_title}] {title}" if tag else f"[{project_title}] {title}"
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, self._encode_chat_ref(space_id, cid))
        tips = []
        if meta.get("imported"):
            src = str(meta.get("import_source") or "").strip() or "another AI"
            tips.append(f"[INGESTED] Imported from {src} export — not created on this system.")
        desc = meta.get("description")
        if desc:
            tips.append(str(desc))
        if tips:
            item.setToolTip("\n".join(tips))
        listw.addItem(item)
        active_jobs, activity_label = self._chat_activity_summary(meta)
        host = _NotesSidebarRowHost(
            listw,
            item,
            shown,
            for_project=False,
            workspace=self,
            active_jobs=active_jobs,
            activity_label=activity_label,
            parent=listw,
        )
        listw.setItemWidget(item, host)
        # Pin sizeHint width to the current viewport. _AutoFitListWidget will
        # keep it in sync on subsequent resizes; this is just the initial value.
        w = max(1, listw.viewport().width())
        h = max(22, host.sizeHint().height())
        item.setSizeHint(QSize(w, h))

    def _on_chat_pinned_changed(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            return
        self.chat_list_recent.blockSignals(True)
        self.chat_list_recent.clearSelection()
        self.chat_list_recent.blockSignals(False)
        raw = current.data(Qt.ItemDataRole.UserRole)
        text = str(raw or "")
        if "::" in text:
            self._apply_chat_selection_from_item(current)
            return
        sid = text.strip()
        if sid:
            self.projectSelected.emit(sid)
            self._refresh_project_workspace(sid)

    def _on_chat_recent_changed(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            return
        self.chat_list_pinned.blockSignals(True)
        self.chat_list_pinned.clearSelection()
        self.chat_list_pinned.blockSignals(False)
        self._apply_chat_selection_from_item(current)

    def _on_project_pinned_changed(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            return
        self.project_list_recent.blockSignals(True)
        self.project_list_recent.clearSelection()
        self.project_list_recent.blockSignals(False)
        raw = current.data(Qt.ItemDataRole.UserRole)
        if raw:
            self.projectSelected.emit(str(raw))
            self._refresh_project_workspace(str(raw))

    def _on_project_recent_changed(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            return
        self.project_list_pinned.blockSignals(True)
        self.project_list_pinned.clearSelection()
        self.project_list_pinned.blockSignals(False)
        raw = current.data(Qt.ItemDataRole.UserRole)
        if raw:
            self.projectSelected.emit(str(raw))
            self._refresh_project_workspace(str(raw))

    def _apply_chat_selection_from_item(self, item: QListWidgetItem) -> None:
        raw = item.data(Qt.ItemDataRole.UserRole)
        if raw:
            sid, cid = self._decode_chat_ref(raw)
            if sid and self._space_root is not None and sid != self._space_root.name:
                self._pending_open_chat_id = cid
                self.projectSelected.emit(sid)
                return
            self._current_chat_id = cid
            self._show_current_chat()

    def _set_main_view_mode(self, projects_mode: bool) -> None:
        if hasattr(self, "_main_stack"):
            self._main_stack.setCurrentIndex(1 if projects_mode else 0)
        if hasattr(self, "_chat_header"):
            self._chat_header.setVisible(not projects_mode)
        if hasattr(self, "_chat_setup_bar"):
            show_setup = (not projects_mode) and bool(getattr(self, "_chat_setup_open", False))
            self._chat_setup_bar.setVisible(show_setup)
        if projects_mode and hasattr(self, "_btn_chat_header_setup"):
            self._chat_setup_open = False
            self._btn_chat_header_setup.blockSignals(True)
            self._btn_chat_header_setup.setChecked(False)
            self._btn_chat_header_setup.blockSignals(False)
        self._sync_sidebar_create_button()
        self._refresh_events_artifacts_panel()

    def _refresh_project_workspace(self, space_id: Optional[str] = None) -> None:
        sid = str(space_id or self._selected_project_space_id() or (self._space_root.name if self._space_root else "")).strip()
        if not sid:
            return
        self._refresh_project_chats_list(sid)
        self._refresh_project_memory_editor(sid)
        self._refresh_project_files_list(sid)
        self._refresh_events_artifacts_panel()

    def _install_project_drop_filters(self) -> None:
        if self._project_drop_filter_installed:
            return
        host = getattr(self, "_project_mem_card", None)
        if host is None:
            return
        host.installEventFilter(self)
        for w in host.findChildren(QWidget):
            w.installEventFilter(self)
        self._project_drop_filter_installed = True

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        host = getattr(self, "_project_mem_card", None)
        if host is not None and isinstance(watched, QWidget):
            if watched is host or host.isAncestorOf(watched):
                et = event.type()
                if et == QEvent.Type.DragEnter and isinstance(event, QDragEnterEvent):
                    paths = _rag_collect_local_paths_from_mime(event.mimeData())
                    if paths:
                        event.acceptProposedAction()
                        return True
                if et == QEvent.Type.DragMove and isinstance(event, QDragMoveEvent):
                    paths = _rag_collect_local_paths_from_mime(event.mimeData())
                    if paths:
                        event.acceptProposedAction()
                        return True
                if et == QEvent.Type.Drop and isinstance(event, QDropEvent):
                    paths = _rag_collect_local_paths_from_mime(event.mimeData())
                    if paths:
                        self._upload_project_paths(paths)
                        event.acceptProposedAction()
                        return True
        return super().eventFilter(watched, event)

    def _refresh_project_chats_list(self, space_id: str) -> None:
        if not hasattr(self, "_project_chats_list"):
            return
        self._project_chats_list.clear()
        if self._space_root is None:
            return
        vault = self._space_root.parent.parent
        mgr = ChatManager(chats_dir=notes_chats_dir(vault, space_id))
        for meta in mgr.list_chats():
            cid = str(meta.get("id") or "").strip()
            if not cid:
                continue
            title = str(meta.get("title") or cid)
            it = QListWidgetItem(f"[{space_id}] {title}")
            it.setData(Qt.ItemDataRole.UserRole, self._encode_chat_ref(space_id, cid))
            self._project_chats_list.addItem(it)
        self._project_chats_title.setText(f"Project chats ({space_id})")

    def _refresh_project_memory_editor(self, space_id: str) -> None:
        sdir = self._space_dir(space_id)
        if sdir is None:
            return
        self._project_memory_edit.setPlainText(read_space_goals(sdir))

    def _refresh_project_files_list(self, space_id: str) -> None:
        if not hasattr(self, "_project_files_list"):
            return
        self._project_files_list.clear()
        sdir = self._space_dir(space_id)
        if sdir is None:
            return
        files_dir = sdir / "files"
        if not files_dir.is_dir():
            return
        for p in sorted(files_dir.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(files_dir))
            it = QListWidgetItem(rel)
            it.setToolTip(str(p))
            self._project_files_list.addItem(it)

    def _save_project_memory(self) -> None:
        sid = str(self._selected_project_space_id() or (self._space_root.name if self._space_root else "")).strip()
        if not sid:
            return
        sdir = self._space_dir(sid)
        if sdir is None:
            return
        meta = sdir / "meta.json"
        try:
            raw = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
            if not isinstance(raw, dict):
                raw = {}
        except Exception:
            raw = {}
        raw["goals"] = self._project_memory_edit.toPlainText().strip()
        raw.setdefault("id", sid)
        raw.setdefault("title", read_space_title(sdir, sid))
        meta.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    def _upload_project_files(self) -> None:
        sid = str(self._selected_project_space_id() or (self._space_root.name if self._space_root else "")).strip()
        if not sid:
            return
        sdir = self._space_dir(sid)
        if sdir is None:
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Upload files to project", str(sdir), "All Files (*)")
        if not files:
            return
        target = sdir / "files"
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            src = Path(str(f)).expanduser()
            if not src.is_file():
                continue
            dst = target / src.name
            if dst.exists():
                stem, suf = src.stem, src.suffix
                n = 2
                while dst.exists():
                    dst = target / f"{stem}_{n}{suf}"
                    n += 1
            try:
                shutil.copy2(str(src), str(dst))
            except Exception:
                continue
        self._refresh_project_files_list(sid)

    def _upload_project_paths(self, paths: list[str]) -> None:
        sid = str(self._selected_project_space_id() or (self._space_root.name if self._space_root else "")).strip()
        if not sid:
            return
        sdir = self._space_dir(sid)
        if sdir is None:
            return
        target = sdir / "files"
        target.mkdir(parents=True, exist_ok=True)
        for raw in paths:
            src = Path(str(raw or "")).expanduser()
            if not src.is_file():
                continue
            dst = target / src.name
            if dst.exists():
                stem, suf = src.stem, src.suffix
                n = 2
                while dst.exists():
                    dst = target / f"{stem}_{n}{suf}"
                    n += 1
            try:
                shutil.copy2(str(src), str(dst))
            except Exception:
                continue
        self._refresh_project_files_list(sid)

    def _new_chat_in_selected_project(self) -> None:
        sid = str(self._selected_project_space_id() or "").strip()
        if not sid or self._space_root is None:
            self._new_chat()
            return
        vault = self._space_root.parent.parent
        mgr = ChatManager(chats_dir=notes_chats_dir(vault, sid))
        cid = mgr.create_chat("New chat")
        if self._space_root.name == sid:
            self._current_chat_id = cid
            self._refresh_chat_list()
            self._select_chat_id(cid)
            self._show_current_chat()
            self.focus_chats_sidebar_mode()
            return
        self._refresh_project_chats_list(sid)

    def _open_project_chat_from_workspace(self, item: QListWidgetItem) -> None:
        sid, cid = self._decode_chat_ref(item.data(Qt.ItemDataRole.UserRole))
        if not cid:
            return
        if sid and self._space_root is not None and sid != self._space_root.name:
            self._pending_open_chat_id = cid
            self.projectSelected.emit(sid)
            return
        self._current_chat_id = cid
        self.focus_chats_sidebar_mode()
        self._set_main_view_mode(False)
        self._select_chat_id(cid)
        self._show_current_chat()

    def _open_selected_project_chat_from_workspace(self) -> None:
        it = self._project_chats_list.currentItem() if hasattr(self, "_project_chats_list") else None
        if it is None:
            return
        self._open_project_chat_from_workspace(it)

    def _selected_chat_id(self) -> Optional[str]:
        for listw in (self.chat_list_pinned, self.chat_list_recent):
            it = listw.currentItem()
            if it is None:
                continue
            raw = it.data(Qt.ItemDataRole.UserRole)
            text = str(raw or "")
            if "::" not in text:
                continue
            _sid, cid = self._decode_chat_ref(raw)
            return cid or None
        return None

    def _selected_project_space_id(self) -> Optional[str]:
        for listw in (self.project_list_pinned, self.project_list_recent):
            it = listw.currentItem()
            if it is not None:
                raw = it.data(Qt.ItemDataRole.UserRole)
                return str(raw) if raw else None
        return None

    def _spaces_root(self) -> Optional[Path]:
        if self._space_root is None:
            return None
        return self._space_root.parent

    def _space_dir(self, space_id: str) -> Optional[Path]:
        root = self._spaces_root()
        if root is None:
            return None
        return (root / space_id).resolve()

    def _events_scope_space_id(self) -> str:
        projects_mode = bool(getattr(self, "_btn_mode_projects", None) and self._btn_mode_projects.isChecked())
        if projects_mode:
            sid = str(self._selected_project_space_id() or "").strip()
            if sid:
                return sid
        return str(self._space_root.name if self._space_root is not None else "").strip()

    def _current_chat_dict(self) -> Optional[dict[str, Any]]:
        if self._chat_mgr is None or not self._current_chat_id:
            return None
        return self._chat_mgr.chats.get(self._current_chat_id) or self._chat_mgr.load_chat(self._current_chat_id)

    def _chat_metadata_records(self, key: str, chat: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        src = chat if chat is not None else self._current_chat_dict()
        if not isinstance(src, dict):
            return []
        meta = src.get("metadata") if isinstance(src.get("metadata"), dict) else {}
        return _coerce_record_list(meta.get(key))

    def _set_chat_metadata_records(self, key: str, records: list[dict[str, Any]], chat_id: str = "") -> None:
        if self._chat_mgr is None:
            return
        cid = str(chat_id or self._current_chat_id or "").strip()
        if not cid:
            return
        chat = self._chat_mgr.chats.get(cid) or self._chat_mgr.load_chat(cid)
        if not isinstance(chat, dict):
            return
        chat.setdefault("metadata", {})[key] = _coerce_record_list(records)
        self._chat_mgr.save_chat(cid)

    def _project_ledger_path(self, space_id: str = "") -> Optional[Path]:
        sid = str(space_id or self._events_scope_space_id()).strip()
        if not sid:
            return None
        sdir = self._space_dir(sid)
        if sdir is None:
            return None
        return sdir / _PROJECT_LEDGER_FILE

    def _read_project_ledger(self, space_id: str = "") -> dict[str, list[dict[str, Any]]]:
        path = self._project_ledger_path(space_id)
        if path is None or not path.is_file():
            return {_PROJECT_EVENTS_KEY: [], _PROJECT_ARTIFACTS_KEY: []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}
        return {
            _PROJECT_EVENTS_KEY: _coerce_record_list(data.get(_PROJECT_EVENTS_KEY)),
            _PROJECT_ARTIFACTS_KEY: _coerce_record_list(data.get(_PROJECT_ARTIFACTS_KEY)),
        }

    def _write_project_ledger(self, ledger: dict[str, list[dict[str, Any]]], space_id: str = "") -> None:
        path = self._project_ledger_path(space_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            _PROJECT_EVENTS_KEY: _coerce_record_list(ledger.get(_PROJECT_EVENTS_KEY)),
            _PROJECT_ARTIFACTS_KEY: _coerce_record_list(ledger.get(_PROJECT_ARTIFACTS_KEY)),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _cvops_http_json(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        win = self.window()
        http = getattr(win, "_http_json", None)
        if not callable(http):
            raise RuntimeError("CV Ops service bridge is not available")
        try:
            return dict(http(method, path, payload, timeout=30.0) or {})
        except TypeError:
            return dict(http(method, path, payload) or {})

    def _tacitus_mcp_context(self) -> dict[str, Any]:
        project_id = str(self._space_root.name if self._space_root is not None else "").strip()
        project_title = read_space_title(self._space_root, project_id) if self._space_root is not None else ""
        active_scenario = ""
        selected_dataset = ""
        win = self.window()
        catalog = getattr(win, "_catalog_panel", None)
        cur = getattr(catalog, "current_scenario", None)
        if callable(cur):
            try:
                active_scenario = str(cur() or "").strip()
            except Exception:
                active_scenario = ""
        if active_scenario:
            try:
                status = self._cvops_http_json("GET", f"/scenarios/{active_scenario}/status")
                selected_dataset = str(
                    status.get("dataset")
                    or status.get("dataset_name")
                    or status.get("dataset_slug")
                    or ""
                ).strip()
            except Exception:
                selected_dataset = ""
        ingested_memory = ""
        if self._space_root is not None:
            try:
                ingested_memory = ai_memory.ledger_summary_text(
                    _notes_vault_root_from_space(self._space_root)
                )
            except Exception:
                ingested_memory = ""
        return {
            "active_project": {"id": project_id, "title": project_title},
            "active_scenario": active_scenario,
            "selected_dataset": selected_dataset,
            "events_artifacts": self._read_project_ledger(project_id),
            "ingested_memory": ingested_memory,
        }

    def _tacitus_mcp_surface(self) -> TacitusMcpSurface:
        return TacitusMcpSurface(
            http_get=lambda path: self._cvops_http_json("GET", path),
            http_post=lambda path, body=None: self._cvops_http_json("POST", path, body),
            context_provider=self._tacitus_mcp_context,
            job_recorder=self.record_chat_job,
            artifact_recorder=lambda label, path, kind, metadata: self.record_chat_artifact(
                label,
                path,
                kind=kind,
                metadata=metadata,
            ),
            event_recorder=self.record_project_event,
        )

    def record_project_event(self, event: dict[str, Any], *, space_id: str = "") -> None:
        sid = str(space_id or self._events_scope_space_id()).strip()
        ledger = self._read_project_ledger(sid)
        clean = dict(event or {})
        clean.setdefault("created_at", _utc_now_iso())
        clean.setdefault("event_id", f"event-{len(ledger[_PROJECT_EVENTS_KEY]) + 1}")
        ledger[_PROJECT_EVENTS_KEY] = _upsert_record(
            ledger[_PROJECT_EVENTS_KEY],
            clean,
            key="event_id",
            limit=120,
        )
        self._write_project_ledger(ledger, sid)
        self._refresh_events_artifacts_panel()

    def record_chat_artifact(
        self,
        label: str,
        path: str = "",
        *,
        kind: str = "file",
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._chat_mgr is None or not self._current_chat_id:
            return
        artifact = dict(metadata or {})
        artifact.update(
            {
                "artifact_id": str(artifact.get("artifact_id") or path or label).strip(),
                "label": str(label or path or "Artifact").strip(),
                "kind": str(kind or "file").strip(),
                "path": str(path or "").strip(),
                "created_at": str(artifact.get("created_at") or _utc_now_iso()),
            }
        )
        records = _upsert_record(
            self._chat_metadata_records(_CHAT_ARTIFACTS_METADATA_KEY),
            artifact,
            key="artifact_id",
            limit=80,
        )
        self._set_chat_metadata_records(_CHAT_ARTIFACTS_METADATA_KEY, records)
        sid = str(self._space_root.name if self._space_root is not None else "").strip()
        if sid:
            ledger = self._read_project_ledger(sid)
            project_artifact = dict(artifact)
            project_artifact["chat_id"] = self._current_chat_id
            ledger[_PROJECT_ARTIFACTS_KEY] = _upsert_record(
                ledger[_PROJECT_ARTIFACTS_KEY],
                project_artifact,
                key="artifact_id",
                limit=120,
            )
            self._write_project_ledger(ledger, sid)
        self._refresh_events_artifacts_panel()

    def record_chat_job(self, job: dict[str, Any]) -> None:
        if self._chat_mgr is None or not self._current_chat_id:
            return
        job_id = str((job or {}).get("job_id") or "").strip()
        if not job_id:
            return
        record = {
            "job_id": job_id,
            "scenario": str(job.get("scenario") or "").strip(),
            "job_type": str(job.get("job_type") or job.get("type") or "").strip(),
            "state": str(job.get("state") or "queued").strip(),
            "source": str(job.get("source") or "tacitus").strip(),
            "created_at": str(job.get("created_at") or _utc_now_iso()),
            "updated_at": str(job.get("updated_at") or _utc_now_iso()),
        }
        records = _upsert_record(
            self._chat_metadata_records(_CHAT_JOBS_METADATA_KEY),
            record,
            key="job_id",
            limit=80,
        )
        self._set_chat_metadata_records(_CHAT_JOBS_METADATA_KEY, records)
        sid = str(self._space_root.name if self._space_root is not None else "").strip()
        if sid:
            self.record_project_event(
                {
                    "event_id": f"chat-job:{self._current_chat_id}:{job_id}",
                    "type": "chat_job",
                    "chat_id": self._current_chat_id,
                    "job_id": job_id,
                    "scenario": record.get("scenario", ""),
                    "state": record.get("state", ""),
                    "updated_at": record.get("updated_at", ""),
                },
                space_id=sid,
            )
        self._refresh_chat_list()
        self._refresh_events_artifacts_panel()

    def apply_cvops_event(self, payload: dict[str, Any]) -> None:
        if self._chat_mgr is None or not isinstance(payload, dict):
            return
        job_id = str(payload.get("job_id") or "").strip()
        if not job_id:
            return
        event_type = str(payload.get("type") or payload.get("event") or "").strip()
        patch = {
            "job_id": job_id,
            "state": str(payload.get("state") or payload.get("event") or "").strip(),
            "scenario": str(payload.get("scenario") or "").strip(),
            "job_type": str(payload.get("job_type") or "").strip(),
            "updated_at": _utc_now_iso(),
            "last_event": event_type,
        }
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        if result:
            patch["result_path"] = str(result.get("result_path") or result.get("output") or "").strip()
            patch["weights"] = str(result.get("weights") or result.get("weights_path") or "").strip()
        changed = False
        for cid, chat in list(self._chat_mgr.chats.items()):
            if not isinstance(chat, dict):
                continue
            jobs = self._chat_metadata_records(_CHAT_JOBS_METADATA_KEY, chat)
            if not any(str(item.get("job_id") or "").strip() == job_id for item in jobs):
                continue
            self._set_chat_metadata_records(
                _CHAT_JOBS_METADATA_KEY,
                _upsert_record(jobs, patch, key="job_id", limit=80),
                chat_id=cid,
            )
            if result:
                artifacts = self._chat_metadata_records(_CHAT_ARTIFACTS_METADATA_KEY, chat)
                for key, label in (("result_path", "Run directory"), ("weights", "Weights")):
                    path = str(patch.get(key) or "").strip()
                    if path:
                        artifacts = _upsert_record(
                            artifacts,
                            {
                                "artifact_id": f"{job_id}:{key}",
                                "label": f"{label} for {job_id}",
                                "kind": "path",
                                "path": path,
                                "job_id": job_id,
                                "created_at": _utc_now_iso(),
                            },
                            key="artifact_id",
                            limit=80,
                        )
                ci_cd = result.get("ci_cd") if isinstance(result.get("ci_cd"), dict) else {}
                report_path = str((ci_cd or {}).get("report_path") or "").strip()
                if report_path:
                    artifacts = _upsert_record(
                        artifacts,
                        {
                            "artifact_id": f"{job_id}:ci_cd_report",
                            "label": f"CI/CD gate report for {job_id}",
                            "kind": "ci_cd_report",
                            "path": report_path,
                            "job_id": job_id,
                            "scenario": patch.get("scenario", ""),
                            "created_at": _utc_now_iso(),
                        },
                        key="artifact_id",
                        limit=80,
                    )
                self._set_chat_metadata_records(_CHAT_ARTIFACTS_METADATA_KEY, artifacts, chat_id=cid)
            changed = True
        sid = str(self._space_root.name if self._space_root is not None else "").strip()
        if changed and sid:
            self.record_project_event(
                {
                    "event_id": f"job-event:{job_id}:{event_type or 'update'}",
                    "type": event_type or "job_update",
                    "job_id": job_id,
                    "scenario": patch.get("scenario", ""),
                    "state": patch.get("state", ""),
                    "updated_at": patch.get("updated_at", ""),
                },
                space_id=sid,
            )
        if changed:
            self._refresh_chat_list()
            self._refresh_events_artifacts_panel()

    def _add_empty_list_item(self, listw: QListWidget, text: str) -> None:
        item = QListWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        listw.addItem(item)

    def _add_events_artifacts_item(self, listw: QListWidget, text: str, payload: dict[str, Any]) -> None:
        data = dict(payload or {})
        base_text = str(text or "").strip()
        active = bool(data.get("activity_active"))
        if active:
            data["activity_base_text"] = base_text
            text = f"{self._activity_spinner_text()} {base_text}"
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, data)
        tip = str(payload.get("path") or payload.get("job_id") or payload.get("summary") or "").strip()
        if tip:
            item.setToolTip(tip)
        listw.addItem(item)

    def _refresh_events_artifacts_panel(self) -> None:
        if not hasattr(self, "_chat_jobs_list"):
            return
        for listw in (self._chat_jobs_list, self._chat_artifacts_list, self._project_events_list):
            listw.clear()

        sid = self._events_scope_space_id()
        project_title = read_space_title(self._space_dir(sid), sid) if self._space_dir(sid) is not None else sid
        chat = self._current_chat_dict()
        chat_title = str((chat or {}).get("title") or self._current_chat_id or "No chat").strip()
        if hasattr(self, "_events_artifacts_scope"):
            self._events_artifacts_scope.setText(f"Project: {project_title or 'None'}\nChat: {chat_title}")

        jobs = self._chat_metadata_records(_CHAT_JOBS_METADATA_KEY, chat)
        if jobs:
            measured_jobs = [self._measure_job_record(item) for item in jobs]
            if measured_jobs != jobs:
                self._set_chat_metadata_records(_CHAT_JOBS_METADATA_KEY, measured_jobs)
                jobs = measured_jobs
        if jobs:
            for item in reversed(jobs[-50:]):
                state = str(item.get("state") or "queued").strip()
                scen = str(item.get("scenario") or "").strip()
                job_id = str(item.get("job_id") or "").strip()
                label = f"{state.upper()}  {job_id}"
                if scen:
                    label += f"  {scen}"
                self._add_events_artifacts_item(
                    self._chat_jobs_list,
                    label,
                    {"kind": "job", "activity_active": _job_state_is_active(state), **item},
                )
        else:
            self._add_empty_list_item(self._chat_jobs_list, "No jobs attached to this chat.")

        artifacts = self._chat_metadata_records(_CHAT_ARTIFACTS_METADATA_KEY, chat)
        if artifacts:
            for item in reversed(artifacts[-50:]):
                label = str(item.get("label") or item.get("path") or "Artifact").strip()
                kind = str(item.get("kind") or "file").strip()
                self._add_events_artifacts_item(self._chat_artifacts_list, f"{kind}  {label}", {"kind": kind, **item})
        else:
            self._add_empty_list_item(self._chat_artifacts_list, "No artifacts attached to this chat.")

        ledger = self._read_project_ledger(sid)
        measured_events: list[dict[str, Any]] = []
        ledger_changed = False
        for item in ledger.get(_PROJECT_EVENTS_KEY, []):
            measured = self._measure_job_record(item) if isinstance(item, dict) and item.get("job_id") else dict(item)
            measured_events.append(measured)
            if measured != item:
                ledger_changed = True
        if ledger_changed:
            ledger[_PROJECT_EVENTS_KEY] = measured_events
            self._write_project_ledger(ledger, sid)
        project_rows: list[dict[str, Any]] = []
        for item in ledger.get(_PROJECT_ARTIFACTS_KEY, [])[-40:]:
            project_rows.append({"kind": "project_artifact", **item})
        for item in ledger.get(_PROJECT_EVENTS_KEY, [])[-80:]:
            project_rows.append({"kind": "project_event", **item})
        if project_rows:
            for item in reversed(project_rows[-80:]):
                kind = str(item.get("kind") or item.get("type") or "event").strip()
                label = str(
                    item.get("label")
                    or item.get("type")
                    or item.get("job_id")
                    or item.get("artifact_id")
                    or "event"
                ).strip()
                state = str(item.get("state") or "").strip()
                text = f"{kind}  {label}"
                if state:
                    text += f"  {state}"
                self._add_events_artifacts_item(
                    self._project_events_list,
                    text,
                    {"activity_active": _job_state_is_active(state), **item},
                )
        else:
            self._add_empty_list_item(self._project_events_list, "No project events or artifacts yet.")
        self._sync_activity_timer()

    def _open_events_artifacts_item(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        job_id = str(data.get("job_id") or "").strip()
        if job_id and str(data.get("kind") or "") in {"job", "project_event"}:
            nav = getattr(self.window(), "_on_eco_navigate", None)
            if callable(nav):
                nav("jobs", job_id, str(data.get("scenario") or ""))
            return
        raw_path = str(data.get("path") or data.get("result_path") or "").strip()
        if raw_path:
            p = Path(raw_path).expanduser()
            if p.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))
            return
        url = str(data.get("url") or "").strip()
        if url:
            QDesktopServices.openUrl(QUrl(url))

    def _menu_icon(self, which: str) -> QIcon:
        sty = self.style()
        if which == "pin":
            ic = QIcon.fromTheme("pinning-pin")
            if not ic.isNull():
                return ic
            return sty.standardIcon(QStyle.StandardPixmap.SP_ArrowUp)
        if which == "rename":
            ic = QIcon.fromTheme("document-edit")
            if not ic.isNull():
                return ic
            return sty.standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView)
        if which == "folder":
            ic = QIcon.fromTheme("folder")
            if not ic.isNull():
                return ic
            return sty.standardIcon(QStyle.StandardPixmap.SP_DirIcon)
        if which == "remove":
            ic = QIcon.fromTheme("folder-remove")
            if not ic.isNull():
                return ic
            return sty.standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton)
        if which == "delete":
            ic = QIcon.fromTheme("edit-delete")
            if not ic.isNull():
                return ic
            return sty.standardIcon(QStyle.StandardPixmap.SP_TrashIcon)
        return QIcon()

    def _fill_chat_context_menu(self, menu: QMenu, chat_id: str) -> None:
        menu.clear()
        sid, cid = self._decode_chat_ref(chat_id)
        if not cid:
            cid = str(chat_id).strip()
        if sid and self._space_root is not None and sid != self._space_root.name:
            open_act = menu.addAction(self._menu_icon("folder"), f"Open project {sid}")
            open_act.triggered.connect(lambda _checked=False, s=sid, c=cid: self._open_chat_from_other_project(s, c))
            return
        if self._chat_mgr is None:
            dis = menu.addAction("(No chat storage)")
            dis.setEnabled(False)
            return
        cid = str(cid).strip()
        if not cid:
            dis = menu.addAction("(Invalid chat)")
            dis.setEnabled(False)
            return
        pinned = self._chat_mgr.is_chat_pinned(cid)
        pin_act = menu.addAction(self._menu_icon("pin"), "Unpin" if pinned else "Pin")
        pin_act.triggered.connect(lambda _checked=False, c=cid, p=not pinned: self._apply_chat_pin(c, p))
        ren_act = menu.addAction(self._menu_icon("rename"), "Rename…")
        ren_act.triggered.connect(lambda _checked=False, c=cid: self._rename_chat_dialog(c))
        ch_act = menu.addAction(self._menu_icon("folder"), "Change project…")
        ch_act.triggered.connect(lambda _checked=False, c=cid: self._change_chat_project_dialog(c))
        rm_act = menu.addAction(self._menu_icon("remove"), "Remove from project")
        rm_act.triggered.connect(lambda _checked=False, c=cid: self._remove_chat_from_project(c))
        menu.addSeparator()
        del_act = menu.addAction(self._menu_icon("delete"), "Delete…")
        del_act.triggered.connect(lambda _checked=False, c=cid: self._delete_chat_confirmed(c))

    def _fill_project_context_menu(self, menu: QMenu, space_id: str) -> None:
        menu.clear()
        sid = str(space_id).strip()
        if not sid:
            dis = menu.addAction("(Invalid project)")
            dis.setEnabled(False)
            return
        sdir = self._space_dir(sid)
        if sdir is None or not sdir.is_dir():
            dis = menu.addAction("(Invalid project)")
            dis.setEnabled(False)
            return
        pinned = read_space_pinned(sdir)
        pin_act = menu.addAction(self._menu_icon("pin"), "Unpin project" if pinned else "Pin project")
        pin_act.triggered.connect(lambda _checked=False, s=sid, p=not pinned: self._apply_project_pin(s, p))
        ren_act = menu.addAction(self._menu_icon("rename"), "Rename…")
        ren_act.triggered.connect(lambda _checked=False, s=sid: self._rename_project_dialog(s))

    def _apply_chat_pin(self, chat_id: str, pinned: bool) -> None:
        if self._chat_mgr is None:
            return
        self._chat_mgr.set_chat_pinned(chat_id, pinned)
        self._refresh_chat_list()
        self._select_chat_id(chat_id)

    def _rename_chat_dialog(self, chat_id: str) -> None:
        if self._chat_mgr is None:
            return
        chat = self._chat_mgr.chats.get(chat_id) or self._chat_mgr.load_chat(chat_id)
        if not chat:
            return
        cur = str(chat.get("title") or "")
        text, ok = QInputDialog.getText(self, "Rename chat", "Chat title:", text=cur)
        if not ok:
            return
        title = text.strip()
        if not title:
            return
        self._chat_mgr.update_chat_metadata(chat_id, title=title)
        self._refresh_chat_list()
        self._select_chat_id(chat_id)

    def _change_chat_project_dialog(self, chat_id: str) -> None:
        if self._space_root is None:
            QMessageBox.warning(self, "Notes AI", "No active notes project.")
            return
        cur_sid = self._space_root.name
        choices = [(sid, title) for sid, title, _goals, _p in self._project_rows if sid != cur_sid]
        if not choices:
            QMessageBox.information(self, "Change project", "No other projects exist yet.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Change project")
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Move this chat to:"))
        combo = QComboBox()
        for sid, title in choices:
            combo.addItem(f"{title} ({sid})", sid)
        lay.addWidget(combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        target = str(combo.currentData() or "").strip()
        if not target:
            return
        ok, err = self._move_chat_json_to_space(chat_id, target)
        if not ok:
            QMessageBox.warning(self, "Change project", err or "Could not move chat.")
            return
        if self._current_chat_id == chat_id:
            self._current_chat_id = None
        self._refresh_chat_list()
        remaining = self._chat_mgr.list_chats()
        if remaining:
            self._current_chat_id = str(remaining[0]["id"])
            self._select_chat_id(self._current_chat_id)
            self._show_current_chat()
        else:
            self._current_chat_id = self._chat_mgr.create_chat("Notes chat")
            self._refresh_chat_list()
            self._select_chat_id(self._current_chat_id)
            self._show_current_chat()

    def _remove_chat_from_project(self, chat_id: str) -> None:
        if self._space_root is None:
            return
        if self._space_root.name == DEFAULT_SPACE_ID:
            QMessageBox.information(
                self,
                "Remove from project",
                "This chat is already in the main project. Use Change project to move it elsewhere.",
            )
            return
        ok, err = self._move_chat_json_to_space(chat_id, DEFAULT_SPACE_ID)
        if not ok:
            QMessageBox.warning(self, "Remove from project", err or "Could not move chat.")
            return
        if self._current_chat_id == chat_id:
            self._current_chat_id = None
        self._refresh_chat_list()
        remaining = self._chat_mgr.list_chats() if self._chat_mgr else []
        if remaining:
            self._current_chat_id = str(remaining[0]["id"])
            self._select_chat_id(self._current_chat_id)
            self._show_current_chat()
        else:
            if self._chat_mgr:
                self._current_chat_id = self._chat_mgr.create_chat("Notes chat")
                self._refresh_chat_list()
                self._select_chat_id(self._current_chat_id)
                self._show_current_chat()

    def _move_chat_json_to_space(self, chat_id: str, target_space_id: str) -> tuple[bool, str]:
        if self._space_root is None or self._chat_mgr is None:
            return False, "No active project."
        current_sid = self._space_root.name
        if target_space_id == current_sid:
            return False, "Target is the current project."
        spaces_root = self._spaces_root()
        if spaces_root is None:
            return False, "Could not resolve spaces folder."
        if not (spaces_root / target_space_id).is_dir():
            return False, f"Project {target_space_id!r} does not exist."
        vault = self._space_root.parent.parent
        src_dir = notes_chats_dir(vault, current_sid)
        dst_dir = notes_chats_dir(vault, target_space_id)
        src_file = src_dir / f"{chat_id}.json"
        if not src_file.is_file():
            return False, "Chat file is missing on disk."
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir / f"{chat_id}.json"
        if dst_file.exists():
            return False, "The target project already has a chat with this id."
        try:
            shutil.move(str(src_file), str(dst_file))
        except OSError as exc:
            return False, str(exc)
        self._chat_mgr.load_all_chats()
        return True, ""

    def _delete_chat_confirmed(self, chat_id: str) -> None:
        if self._chat_mgr is None:
            return
        ret = QMessageBox.question(
            self,
            "Delete chat",
            "Delete this chat and all of its messages? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        self._chat_mgr.delete_chat(chat_id)
        self._refresh_chat_list()
        remaining = self._chat_mgr.list_chats()
        if remaining:
            self._current_chat_id = str(remaining[0]["id"])
        else:
            self._current_chat_id = self._chat_mgr.create_chat("Notes chat")
            self._refresh_chat_list()
        self._select_chat_id(self._current_chat_id or "")
        self._show_current_chat()

    def _apply_project_pin(self, space_id: str, pinned: bool) -> None:
        sdir = self._space_dir(space_id)
        if sdir is None:
            return
        set_space_pinned(sdir, pinned)
        self.projectsMetadataChanged.emit()

    def _rename_project_dialog(self, space_id: str) -> None:
        sdir = self._space_dir(space_id)
        if sdir is None:
            return
        cur = read_space_title(sdir, space_id)
        text, ok = QInputDialog.getText(self, "Rename project", "Project title:", text=cur)
        if not ok:
            return
        title = text.strip()
        if not title:
            return
        if not update_space_meta_title(sdir, title):
            QMessageBox.warning(
                self,
                "Rename project",
                "Could not update meta.json (missing or unreadable).",
            )
            return
        self.projectsMetadataChanged.emit()

    def _new_chat(self) -> None:
        if self._chat_mgr is None:
            return
        self._current_chat_id = self._chat_mgr.create_chat("New chat")
        self._refresh_chat_list()
        self._select_chat_id(self._current_chat_id)
        self._show_current_chat()

    def _open_chat_from_other_project(self, space_id: str, chat_id: str) -> None:
        self._pending_open_chat_id = str(chat_id or "").strip()
        if self._pending_open_chat_id:
            self.projectSelected.emit(str(space_id or "").strip())

    def _select_chat_id(self, chat_id: str) -> None:
        current_sid = self._space_root.name if self._space_root is not None else ""
        target = self._encode_chat_ref(current_sid, chat_id)
        for listw in (self.chat_list_pinned, self.chat_list_recent):
            for i in range(listw.count()):
                it = listw.item(i)
                if it and str(it.data(Qt.ItemDataRole.UserRole) or "") == target:
                    listw.setCurrentItem(it)
                    return

    def _update_chat_header(self) -> None:
        """Sync the chat header (title + model + streaming status)."""
        if not hasattr(self, "_chat_header_title"):
            return
        title = "New chat"
        project_title = "Main"
        if self._space_root is not None:
            project_title = read_space_title(self._space_root, self._space_root.name)
        if self._chat_mgr is not None and self._current_chat_id:
            chat = self._chat_mgr.chats.get(self._current_chat_id) or self._chat_mgr.load_chat(self._current_chat_id)
            if chat:
                title = str(chat.get("title") or self._current_chat_id)
        self._chat_header_title.setText(f"Project[{project_title}] \\ Chat[{title}]")
        if hasattr(self, "chat_model"):
            label = self.chat_model.currentText().strip() or "—"
            self._chat_header_model.setText(label)
        if self._is_streaming:
            self._chat_header_status.setText("Streaming…")
            self._chat_header_status.setProperty("statusKind", "streaming")
        elif self._streaming_error:
            self._chat_header_status.setText("Error")
            self._chat_header_status.setProperty("statusKind", "error")
        else:
            self._chat_header_status.setText("Idle")
            self._chat_header_status.setProperty("statusKind", "idle")
        repolish(self._chat_header_status)

    def _toggle_events_artifacts_panel(self, visible: bool) -> None:
        """Show/hide the right-hand Events & Artifacts panel, preserving its width."""
        panel = getattr(self, "_events_artifacts_panel", None)
        if panel is None:
            return
        split = getattr(self, "_chat_splitter", None)
        if not visible and split is not None:
            sizes = split.sizes()
            if len(sizes) == 3 and sizes[2] > 0:
                self._events_panel_width = sizes[2]
        panel.setVisible(visible)
        if visible and split is not None:
            sizes = split.sizes()
            if len(sizes) == 3:
                sizes[2] = int(getattr(self, "_events_panel_width", 0) or 300)
                split.setSizes(sizes)

    def _toggle_chat_setup_bar(self, checked: bool) -> None:
        self._chat_setup_open = bool(checked)
        if hasattr(self, "_chat_setup_bar") and hasattr(self, "_main_stack"):
            self._chat_setup_bar.setVisible(self._chat_setup_open and self._main_stack.currentIndex() == 0)

    def _open_chat_header_menu(self) -> None:
        cid = self._current_chat_id
        menu = QMenu(self._btn_chat_header_more)
        setup_act = menu.addAction("AI connection")
        setup_act.setCheckable(True)
        setup_act.setChecked(bool(getattr(self, "_chat_setup_open", False)))
        setup_act.triggered.connect(lambda checked=False: self._btn_chat_header_setup.setChecked(bool(checked)))
        menu.addSeparator()
        if cid and self._chat_mgr is not None:
            export_act = menu.addAction("Export chat")
            export_act.triggered.connect(self._export_current_chat)
            self._fill_chat_context_menu(menu, cid)
            menu.addSeparator()
        new_act = menu.addAction("New chat")
        new_act.triggered.connect(self._new_chat)
        menu.exec(self._btn_chat_header_more.mapToGlobal(self._btn_chat_header_more.rect().bottomLeft()))

    def _export_current_chat(self) -> None:
        if self._chat_mgr is None or not self._current_chat_id:
            QMessageBox.information(self, "Export chat", "No chat is selected.")
            return
        chat = self._chat_mgr.chats.get(self._current_chat_id) or self._chat_mgr.load_chat(self._current_chat_id)
        if not chat:
            QMessageBox.information(self, "Export chat", "Could not load this chat.")
            return
        title = str(chat.get("title") or self._current_chat_id).strip() or self._current_chat_id
        safe_name = "".join(c if c.isalnum() or c in (" ", "_", "-") else "_" for c in title).strip() or "chat"
        default_path = str(Path.home() / f"{safe_name}.md")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export chat as Markdown", default_path, "Markdown (*.md);;All files (*)"
        )
        if not path:
            return
        lines = [f"# {title}", ""]
        assistant_name = self.assistant_name()
        for m in chat.get("messages", []) or []:
            role = str(m.get("role", "")).strip().lower() or "assistant"
            heading = {"user": "User", "assistant": assistant_name, "system": "System"}.get(role, role.title())
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(str(m.get("content", "")).rstrip())
            lines.append("")
        try:
            Path(path).write_text("\n".join(lines), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Export chat", f"Could not write file: {exc}")
            return
        self.errorRaised.emit(f"Exported chat to {path}")

    def _set_chat_status(self, message: str) -> None:
        """Surface a transient composer status message (dictation / TTS)."""
        msg = str(message or "").strip()
        if not msg:
            return
        if hasattr(self, "_chat_header_status"):
            self._chat_header_status.setText(msg)
            self._chat_header_status.setProperty("statusKind", "idle")
            repolish(self._chat_header_status)
        self.errorRaised.emit(msg)

    def _resync_composer_overlay(self) -> None:
        """Recompute the overlaid composer height so the TTS bar and the text
        input stay fully visible together (the bar adds a row to the dock)."""
        host = getattr(self, "_chat_overlay_host", None)
        if host is not None:
            QTimer.singleShot(0, host._sync_overlay_geometry)

    @staticmethod
    def _chat_timeline_label(index: int, message: dict[str, Any]) -> str:
        role = str(message.get("role") or "assistant").strip().lower()
        role_label = {
            "assistant": "AI",
            "user": "You",
            "system": "Sys",
        }.get(role, role[:3].title() or "Msg")
        stamp = NotesAiWorkspace._chat_timeline_stamp(index, message)
        return f"{role_label}\n{stamp}"

    @staticmethod
    def _chat_timeline_tooltip(index: int, message: dict[str, Any]) -> str:
        role = str(message.get("role") or "assistant").strip().title() or "Message"
        stamp = NotesAiWorkspace._chat_timeline_stamp(index, message)
        preview = " ".join(str(message.get("content") or "").split())
        if not preview:
            preview = "(empty)"
        return f"{stamp} {role}\n{preview}"

    @staticmethod
    def _chat_timeline_stamp(index: int, message: dict[str, Any]) -> str:
        raw = message.get("timestamp") or message.get("created_at")
        if raw:
            try:
                if isinstance(raw, (int, float)):
                    dt = datetime.fromtimestamp(float(raw)).astimezone()
                else:
                    text = str(raw).strip()
                    if text.endswith("Z"):
                        text = f"{text[:-1]}+00:00"
                    dt = datetime.fromisoformat(text)
                    if dt.tzinfo is not None:
                        dt = dt.astimezone()
                now = (
                    datetime.now(dt.tzinfo).date()
                    if dt.tzinfo is not None
                    else datetime.now().date()
                )
                return dt.strftime("%H:%M") if dt.date() == now else dt.strftime("%m/%d")
            except Exception:
                pass
        return f"{index + 1:02d}"

    def _sync_chat_timeline(
        self,
        messages: list[dict],
        *,
        include_streaming: bool = False,
    ) -> None:
        listw = getattr(self, "_chat_timeline_list", None)
        if listw is None:
            return
        listw.blockSignals(True)
        listw.clear()
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            item = QListWidgetItem(self._chat_timeline_label(idx, msg))
            item.setData(Qt.ItemDataRole.UserRole, f"msg-{idx}")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setSizeHint(QSize(48, 34))
            item.setToolTip(self._chat_timeline_tooltip(idx, msg))
            listw.addItem(item)
        if include_streaming:
            item = QListWidgetItem("AI\nlive")
            item.setData(Qt.ItemDataRole.UserRole, "msg-streaming")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setSizeHint(QSize(48, 34))
            item.setToolTip("AI response streaming")
            listw.addItem(item)
        if listw.count() == 0:
            item = QListWidgetItem("No\nchat")
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setSizeHint(QSize(48, 34))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            listw.addItem(item)
        else:
            listw.setCurrentRow(listw.count() - 1)
            listw.scrollToBottom()
        listw.blockSignals(False)

    def _on_chat_timeline_changed(
        self,
        current: Optional[QListWidgetItem],
        _previous: Optional[QListWidgetItem],
    ) -> None:
        if current is None:
            return
        anchor = str(current.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not anchor:
            return
        try:
            self.chat_view.scrollToAnchor(anchor)
        except Exception:
            pass

    def _on_dictate_clicked(self) -> None:
        self._dictation.toggle()

    def _on_dictation_state(self, state: str) -> None:
        btn = getattr(self, "_btn_compose_dictate", None)
        if btn is None:
            return
        if state == "recording":
            btn.setChecked(True)
            btn.setEnabled(True)
            btn.setText("■")  # stop square
            btn.setToolTip("Recording… click to stop and transcribe")
            if hasattr(self, "_chat_header_status"):
                self._chat_header_status.setText("Recording…")
                repolish(self._chat_header_status)
        elif state == "transcribing":
            btn.setChecked(False)
            btn.setEnabled(False)
            btn.setText("…")
            btn.setToolTip("Transcribing dictation…")
            if hasattr(self, "_chat_header_status"):
                self._chat_header_status.setText("Transcribing…")
                repolish(self._chat_header_status)
        else:  # idle
            btn.setChecked(False)
            btn.setEnabled(microphone_available())
            btn.setText("\U0001F3A4")
            btn.setToolTip(
                "Dictate: hold a short message, click again to stop and transcribe"
            )
            self._update_chat_header()

    def _on_dictation_text(self, text: str) -> None:
        clean = str(text or "").strip()
        if not clean or not hasattr(self, "chat_input"):
            return
        existing = self.chat_input.toPlainText()
        prefix = "" if (not existing or existing.endswith((" ", "\n"))) else " "
        self.chat_input.insertPlainText(prefix + clean)
        self.chat_input.setFocus()

    def _on_dictation_error(self, message: str) -> None:
        self.errorRaised.emit(f"Dictation: {message}")

    def _on_chat_anchor_clicked(self, url: QUrl) -> None:
        """Route ``cvops-action://`` links and open everything else externally."""
        scheme = url.scheme().lower()
        if scheme == "cvops-action":
            # QUrl parses cvops-action://copy/3 as host=copy, path=/3.
            action = url.host() or ""
            arg = url.path().lstrip("/")
            self._handle_chat_action(action, arg)
            return
        if scheme in ("http", "https", "mailto", "file"):
            QDesktopServices.openUrl(url)
            return
        # Unknown scheme: silently ignore so we never navigate inside the view.

    def _handle_chat_action(self, action: str, arg: str) -> None:
        if action == "use-prompt":
            try:
                idx = int(arg)
            except (TypeError, ValueError):
                return
            if 0 <= idx < len(_DEFAULT_PROMPT_SUGGESTIONS):
                _label, prompt = _DEFAULT_PROMPT_SUGGESTIONS[idx]
                self.chat_input.setPlainText(prompt)
                self.chat_input.setFocus()
            return
        try:
            mi = int(arg)
        except (TypeError, ValueError):
            return
        if self._chat_mgr is None or not self._current_chat_id:
            return
        msgs = list(self._chat_mgr.get_chat_messages(self._current_chat_id))
        if not (0 <= mi < len(msgs)):
            return
        target = msgs[mi]
        role = str(target.get("role", "")).strip().lower()
        content = str(target.get("content", ""))
        if action == "copy":
            QApplication.clipboard().setText(content)
            return
        if action == "speak" and role == "assistant":
            if hasattr(self, "_tts_bar"):
                self._tts_bar.speak(content)
            return
        if action == "edit" and role == "user":
            self._truncate_chat_to_index(mi, drop_target=True)
            self.chat_input.setPlainText(content)
            self.chat_input.setFocus()
            self._show_current_chat()
            return
        if action == "regen" and role == "assistant":
            # Drop this assistant turn (and anything after) and re-send the
            # most recent prior user message.
            prior_user_idx = -1
            for j in range(mi - 1, -1, -1):
                if str(msgs[j].get("role", "")).strip().lower() == "user":
                    prior_user_idx = j
                    break
            if prior_user_idx < 0:
                return
            user_text = str(msgs[prior_user_idx].get("content", ""))
            self._truncate_chat_to_index(prior_user_idx, drop_target=True)
            self.chat_input.setPlainText(user_text)
            self._show_current_chat()
            self._send_chat()
            return

    def _truncate_chat_to_index(self, index: int, *, drop_target: bool) -> None:
        """Rewrite the active chat's messages list to drop everything from ``index`` on."""
        if self._chat_mgr is None or not self._current_chat_id:
            return
        msgs = list(self._chat_mgr.get_chat_messages(self._current_chat_id))
        cut = index if drop_target else index + 1
        keep = msgs[:cut]
        chat = self._chat_mgr.chats.get(self._current_chat_id)
        if chat is None:
            chat = self._chat_mgr.load_chat(self._current_chat_id)
        if chat is None:
            return
        chat["messages"] = keep
        save_fn = getattr(self._chat_mgr, "save_chat", None)
        if callable(save_fn):
            try:
                save_fn(self._current_chat_id)
            except Exception:
                pass

    def _render_chat_html(
        self,
        messages: list[dict],
        *,
        streaming_text: str = "",
        streaming_error: str = "",
        streaming_model_label: str = "",
    ) -> str:
        """Build chat HTML: markdown bubbles, per-message actions, code styling.

        Themed via ``cvops_color`` so the chat tracks the active aurora palette
        and the global beacon-red accent. Per-message actions are emitted as
        ``cvops-action://`` anchor links and dispatched in
        ``_on_chat_anchor_clicked``.
        """
        bg_void = cvops_color("bg_void")
        line_med = cvops_color("line_med")
        text_bright = cvops_color("text_bright")
        text_signal = cvops_color("text_signal")
        text_iron = cvops_color("text_iron")
        accent_active = cvops_color("accent_active")
        accent_alert = cvops_color("accent_alert")
        bubble_line = cvops_rgba("line_light", 0.16)
        action_line = cvops_rgba("line_light", 0.12)
        column_fill = cvops_rgba("bg_panel", 0.20)
        column_edge = cvops_rgba("line_light", 0.10)
        assistant_name = self.assistant_name()

        role_palette = {
            "user": {
                "side": "right",
                "bubble_bg": "#1E2328",
                "bubble_fg": "#FFFFFF",
                "bubble_edge": cvops_rgba("line_light", 0.18),
            },
            "assistant": {
                "side": "left",
                "bubble_bg": cvops_color("bg_panel"),
                "bubble_fg": text_bright,
                "bubble_edge": cvops_rgba("line_light", 0.14),
            },
            "system": {
                "side": "left",
                "bubble_bg": line_med,
                "bubble_fg": text_iron,
                "bubble_edge": cvops_rgba("line_light", 0.12),
            },
        }

        def _action_link(label: str, url: str, color: str) -> str:
            return (
                f'<a href="{html.escape(url)}" '
                f'style="color:{color}; text-decoration:none; '
                f'font-size:11px; padding:0 2px;">{html.escape(label)}</a>'
            )

        def _action_join(links: list[str]) -> str:
            spacer = '<span style="display:inline-block; width:12px;">&nbsp;</span>'
            return (
                f'<span style="white-space:nowrap;">'
                f"{spacer.join(links)}"
                f"</span>"
            )

        def _bubble_width_hint(role: str, content: str) -> int:
            plain = " ".join(str(content or "").split())
            chars = len(plain)
            if role.lower() == "user":
                return max(220, min(440, 156 + chars * 5))
            return max(260, min(560, 188 + chars * 4))

        def _card(
            role: str,
            content: str,
            *,
            error: bool = False,
            actions: str = "",
            streaming: bool = False,
            model_label: str = "",
        ) -> str:
            pal = role_palette.get(role.lower(), role_palette["assistant"])
            side = pal["side"]
            bubble_bg = accent_alert if error else pal["bubble_bg"]
            bubble_fg = "#FFFFFF" if error else pal["bubble_fg"]
            bubble_edge = accent_alert if error else pal.get("bubble_edge", bubble_line)
            meta_block = ""
            if role.lower() == "assistant":
                meta_parts = [assistant_name]
                if model_label:
                    meta_parts.append(model_label)
                meta_block = (
                    f'<div style="margin:0 0 6px 0; color:{text_iron}; font-size:10px; '
                    f'font-weight:600; letter-spacing:0.06em; text-transform:uppercase;">'
                    f"{html.escape(' · '.join(meta_parts))}</div>"
                )
            if error:
                rendered = html.escape(str(content or "")).replace("\n", "<br>")
            else:
                rendered = (
                    '<div class="cvops-md">'
                    + _render_markdown_html(str(content or ""))
                    + "</div>"
                )
            if streaming:
                # Soft pulsing block-cursor at the very end of the streaming text.
                rendered += (
                    f'<span style="color:{text_iron};">&nbsp;&#9612;</span>'
                )
            if not rendered.strip():
                rendered = f'<i style="color:{text_iron};">…</i>'
            actions_block = ""
            if actions:
                actions_block = (
                    f'<div style="margin-top:8px; padding-top:5px; border-top:1px solid {action_line}; '
                    f'color:{text_iron}; font-size:11px;">{actions}</div>'
                )
            bubble_w = _bubble_width_hint(role, content)
            bubble_cell = (
                f'<td width="86%" align="{side}" '
                f'style="padding: 2px 0 2px 0;">'
                f'<table class="cvops-layout" cellpadding="0" cellspacing="0" border="0" '
                f'style="background: {bubble_bg}; border: 1px solid {bubble_edge}; '
                f'border-radius: 12px; min-width:{bubble_w}px;">'
                f'<tr><td style="padding: 9px 13px; color: {bubble_fg}; '
                f'border-radius: 12px;">{meta_block}{rendered}{actions_block}</td></tr>'
                f"</table>"
                f"</td>"
            )
            spacer_cell = '<td width="12%">&nbsp;</td>'
            row = (spacer_cell + bubble_cell) if side == "right" else (bubble_cell + spacer_cell)
            bubble_table = (
                f'<table class="cvops-layout" cellpadding="0" cellspacing="0" border="0" '
                f'width="100%" style="margin: 0;">'
                f"<tr>{row}</tr>"
                f"</table>"
            )
            return bubble_table + (
                f'<div style="height:2px; line-height:2px;">&nbsp;</div>'
            )

        parts: list[str] = []
        for idx, m in enumerate(messages):
            role = str(m.get("role", "")).strip().lower()
            display_content = str(m.get("content", ""))
            metadata = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
            model_label = str(
                metadata.get("model_label")
                or metadata.get("model")
                or ""
            ).strip()
            if role == "user":
                actions = _action_join([
                    _action_link("Copy", f"cvops-action://copy/{idx}", text_iron)
                    , _action_link("Edit", f"cvops-action://edit/{idx}", text_iron)
                ])
            elif role == "assistant":
                assistant_links = [
                    _action_link("Copy", f"cvops-action://copy/{idx}", text_iron),
                    _action_link("Regenerate", f"cvops-action://regen/{idx}", text_iron),
                ]
                if self._tts_enabled:
                    assistant_links.append(
                        _action_link(
                            "▶ Play", f"cvops-action://speak/{idx}", accent_active
                        )
                    )
                actions = _action_join(assistant_links)
            else:
                actions = ""
            parts.append(f'<a name="msg-{idx}"></a>')
            parts.append(_card(role, display_content, actions=actions, model_label=model_label))

        if streaming_text or self._is_streaming:
            stream_display = str(streaming_text or "")
            parts.append('<a name="msg-streaming"></a>')
            parts.append(
                _card("assistant", stream_display, streaming=True, model_label=streaming_model_label)
            )
        if streaming_error:
            parts.append(_card("error", streaming_error, error=True))

        if not parts:
            # Empty-state: smaller intro + starter cards. In the narrow assistant
            # overlay the cards go full width (stacked) and everything scales down
            # so nothing overflows the card.
            compact = bool(getattr(self, "_compact_overlay", False))
            card_width = "100%" if compact else "47%"
            card_min_width = 0 if compact else 260
            card_max_width = 320 if compact else 360
            card_padding = "7px 10px" if compact else "10px 14px"
            card_margin = "3px 0 4px 0" if compact else "4px 10px 6px 0"
            desc_fs = 10 if compact else 11
            desc_chars = 64 if compact else 90
            intro_pad = "10px 0 6px 0" if compact else "22px 0 10px 0"
            intro_title_fs = 13 if compact else 16
            intro_sub_fs = 11 if compact else 12
            intro_mt = 10 if compact else 14
            cards: list[str] = []
            for sidx, (label, prompt) in enumerate(_DEFAULT_PROMPT_SUGGESTIONS):
                href = f"cvops-action://use-prompt/{sidx}"
                cards.append(
                    f'<a href="{html.escape(href)}" '
                    f'style="display:inline-block; width:{card_width}; min-width:{card_min_width}px; '
                    f'max-width:{card_max_width}px; vertical-align:top; text-decoration:none;">'
                    f'<table class="cvops-layout" cellpadding="0" cellspacing="0" border="0" '
                    f'style="background:{bg_void}; border:1px solid {line_med}; '
                    f'margin: {card_margin};">'
                    f'<tr><td style="padding:{card_padding}; color:{text_bright};">'
                    f'<b>{html.escape(label)}</b><br>'
                    f'<span style="color:{text_iron}; font-size:{desc_fs}px;">'
                    f'{html.escape(prompt[:desc_chars] + ("…" if len(prompt) > desc_chars else ""))}'
                    f'</span></td></tr></table></a>'
                )
            parts.append(
                f'<div style="padding: {intro_pad}; text-align:left;">'
                f'<div style="color:{text_bright}; font-size:{intro_title_fs}px; font-weight:600;">{html.escape(assistant_name)} workspace</div>'
                f'<div style="color:{text_iron}; font-size:{intro_sub_fs}px; margin-top:4px;">'
                f"Pick a model, start a prompt, or use one of these starters."
                f"</div>"
                f'<div style="margin-top:{intro_mt}px;">{"".join(cards)}</div>'
                f"</div>"
            )

        font_family = "'Inter', 'Söhne', -apple-system, 'Segoe UI', system-ui, sans-serif"
        # Code styling: terminal-feel block tied to the cvops palette.
        css = (
            f"pre {{ background: {bg_void}; border: 1px solid {line_med}; "
            f"padding: 10px 12px; margin: 6px 0; color: {text_bright}; "
            f"font-family: 'JetBrains Mono', 'Menlo', 'Consolas', monospace; "
            f"font-size: 12px; white-space: pre-wrap; }} "
            f"code {{ font-family: 'JetBrains Mono', 'Menlo', 'Consolas', monospace; "
            f"font-size: 12px; background: {bg_void}; padding: 1px 4px; "
            f"border: 1px solid {line_med}; }} "
            f"pre code {{ background: transparent; border: none; padding: 0; }} "
            f"table.cvops-layout, table.cvops-layout tr, table.cvops-layout td, "
            f"table.cvops-layout th {{ border: none !important; background: transparent; }}"
            f"blockquote {{ border-left: 3px solid {accent_active}; "
            f"margin: 6px 0; padding: 2px 10px; color: {text_signal}; }} "
            f".cvops-md table {{ border-collapse: collapse; }} "
            f".cvops-md th, .cvops-md td {{ border: 1px solid {line_med}; padding: 4px 8px; }} "
            f"a {{ color: {accent_active}; }}"
        )
        wrapper_open = (
            f'<style>{css}</style>'
            f'<table class="cvops-layout" cellpadding="0" cellspacing="0" border="0" width="100%">'
            f'<tr><td align="center">'
            f'<div style="font-family: {font_family}; font-size: 13px; '
            f'color: {text_signal}; line-height: 1.5; width: 100%; max-width: 980px; '
            f'background: {column_fill}; border-left: 1px solid {column_edge}; '
            f'border-right: 1px solid {column_edge}; padding: 20px 28px 170px 28px;">'
            f'<div style="width: 100%; max-width: 812px; margin: 0 auto;">'
        )
        return wrapper_open + "".join(parts) + "</div></div></td></tr></table>"

    def _show_current_chat(self) -> None:
        self._update_chat_header()
        if not self._current_chat_id or self._chat_mgr is None:
            self._sync_chat_timeline([])
            self.chat_view.setHtml(self._render_chat_html([]))
            self._refresh_events_artifacts_panel()
            return
        messages = list(self._chat_mgr.get_chat_messages(self._current_chat_id))
        self._sync_chat_timeline(
            messages,
            include_streaming=bool(self._streaming_assistant or self._is_streaming),
        )
        self.chat_view.setHtml(
            self._render_chat_html(
                messages,
                streaming_text=self._streaming_assistant,
                streaming_error=self._streaming_error,
                streaming_model_label=self._streaming_model_label,
            )
        )
        # Auto-scroll to the most recent message.
        bar = self.chat_view.verticalScrollBar()
        if bar is not None:
            bar.setValue(bar.maximum())
        self._refresh_events_artifacts_panel()

    def _should_offer_tacitus_mcp_tools(self, text: str) -> bool:
        body = str(text or "").lower()
        if any(Path(p).expanduser().is_dir() for p in list(self._composer_attachments or [])):
            return True
        if "@scenario" in body or "@dataset" in body or "@job" in body:
            return True
        return bool(
            re.search(
                r"\b(cv\s*ops|cvops|scenario|dataset|pipeline|ci/cd|cicd|gate|job|artifact|promot|train|update|run)\b",
                body,
            )
        )

    def _maybe_dispatch_tacitus_mcp_model_response(self, text: str) -> Optional[dict[str, Any]]:
        call = _extract_structured_mcp_tool_call(text)
        if call is None:
            return None
        surface = self._tacitus_mcp_surface()
        result = surface.dispatch_provider_tool_call(call)
        tool_name = str(result.get("tool") or "")
        parsed = TacitusMcpSurface.validate_provider_tool_call(call)
        if parsed.get("mcp_tool"):
            tool_name = str(parsed.get("mcp_tool") or tool_name)
        event = {
            "event_id": f"mcp:{self._current_chat_id or 'chat'}:{int(time.time() * 1000)}",
            "type": "mcp_tool_call",
            "chat_id": self._current_chat_id or "",
            "tool": tool_name,
            "ok": bool(result.get("ok")),
            "summary": str(result.get("summary") or ""),
            "error": str(result.get("error") or ""),
            "model_label": self._streaming_model_label,
        }
        self.record_project_event(event)
        return {"call": call, "result": result}

    def _format_tacitus_mcp_tool_reply(self, dispatch: dict[str, Any]) -> str:
        result = dispatch.get("result") if isinstance(dispatch.get("result"), dict) else {}
        tool = str(result.get("tool") or "provider.tool_call").strip()
        if not result.get("ok"):
            err = str(result.get("error") or "tool call failed").strip()
            return f"Tacitus MCP rejected `{tool}`: {err}"

        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        lines = [str(result.get("summary") or f"Tacitus MCP completed `{tool}`.").strip()]
        if tool == "pipeline.get":
            scenario = str(data.get("scenario") or "").strip()
            ci_cd = data.get("ci_cd") if isinstance(data.get("ci_cd"), dict) else {}
            candidate = data.get("candidate") if isinstance(data.get("candidate"), dict) else {}
            prod = data.get("prod") if isinstance(data.get("prod"), dict) else {}
            latest_gate = data.get("latest_gate") if isinstance(data.get("latest_gate"), dict) else {}
            if scenario:
                lines.append(f"Scenario: `{scenario}`")
            if ci_cd:
                lines.append(
                    f"CI/CD: enabled=`{ci_cd.get('enabled')}` promotion=`{ci_cd.get('promotion') or 'manual'}`"
                )
            if candidate:
                lines.append(f"Candidate: `{candidate.get('version_id') or candidate.get('run_version') or 'available'}`")
            if prod:
                lines.append(f"Production: `{prod.get('version_id') or prod.get('run_version') or 'set'}`")
            if latest_gate:
                lines.append(f"Latest gate: `{latest_gate.get('gate_status') or latest_gate.get('status') or 'available'}`")
        elif tool == "run.launch":
            job_id = str(data.get("job_id") or "").strip()
            scenario = str(data.get("scenario") or "").strip()
            if scenario:
                lines.append(f"Scenario: `{scenario}`")
            if job_id:
                lines.append(f"Job: `{job_id}`")
            lines.append("Promotion remains manual. Job progress and artifacts will appear in Events & artifacts.")
        elif tool == "job.status":
            job = data.get("job") if isinstance(data.get("job"), dict) else {}
            result_data = data.get("result") if isinstance(data.get("result"), dict) else {}
            if job.get("job_id"):
                lines.append(f"Job: `{job.get('job_id')}` state=`{job.get('state') or 'unknown'}`")
            if result_data.get("run_version"):
                lines.append(f"Run version: `{result_data.get('run_version')}`")
        elif tool == "gate.get":
            if data.get("gate_status"):
                lines.append(f"Gate: `{data.get('gate_status')}`")
            if data.get("report_path"):
                lines.append(f"Report: `{data.get('report_path')}`")
        elif tool == "promotion.request":
            state = str(data.get("state") or "").strip()
            if state == "confirmation_required":
                lines.append("Promotion was not executed. Explicit confirmation is required.")
        return "\n\n".join(line for line in lines if line)

    def _maybe_handle_tacitus_controlled_run(self, text: str) -> bool:
        req = parse_controlled_run_request(text, self._composer_attachments)
        if req is None:
            return False
        if self._chat_mgr is None:
            return False
        if not self._current_chat_id:
            self._current_chat_id = self._chat_mgr.create_chat("Tacitus controlled run")
            self._refresh_chat_list()
        self._chat_mgr.add_message(
            self._current_chat_id,
            "user",
            text,
            metadata={"mcp_intent": "tacitus.controlled_run"},
        )
        self.chat_input.clear()
        self._composer_attachments = []
        self._update_tools_strip()
        try:
            result = self._tacitus_mcp_surface().controlled_run(**req)
        except Exception as exc:
            result = {
                "ok": False,
                "summary": "",
                "error": str(exc),
                "data": {},
            }
        reply = self._format_tacitus_controlled_run_reply(result)
        self._chat_mgr.add_message(
            self._current_chat_id,
            "assistant",
            reply,
            metadata={"model_label": "Tacitus MCP", "mcp_result": result},
        )
        self._show_current_chat()
        return True

    def _format_tacitus_controlled_run_reply(self, result: dict[str, Any]) -> str:
        if not result.get("ok"):
            err = str(result.get("error") or "controlled run failed").strip()
            return f"Tacitus MCP could not launch the run: {err}"
        data = result.get("data") if isinstance(result.get("data"), dict) else {}
        scenario = str(data.get("scenario") or "").strip()
        job_id = str(data.get("job_id") or "").strip()
        mode = str(data.get("mode") or "train").strip()
        lines = [str(result.get("summary") or "Tacitus MCP queued the controlled run.").strip()]
        if scenario:
            lines.append(f"Scenario: `{scenario}`")
        if job_id:
            lines.append(f"Job: `{job_id}`")
        lines.append(f"Mode: `{mode}`")
        lines.append("Promotion remains manual. Gate reports and run artifacts will appear in Events & artifacts as CV Ops emits job updates.")
        return "\n\n".join(lines)

    def _send_chat(self) -> None:
        if self._chat_mgr is None:
            return
        if not self._current_chat_id:
            self._current_chat_id = self._chat_mgr.create_chat("Notes chat")
            self._refresh_chat_list()
        text = self.chat_input.toPlainText().strip()
        if not text:
            return
        if self._chat_worker is not None and self._chat_worker.isRunning():
            QMessageBox.information(self, "Chat", "Wait for the current reply to finish.")
            return

        if self._maybe_handle_tacitus_controlled_run(text):
            return

        provider, model, model_label = self._current_chat_route_parts()

        settings = load_ai_settings()
        if provider == "openai" and not settings.get(KEY_OPENAI, "").strip():
            QMessageBox.warning(self, "Chat", "Add an OpenAI API key in AI settings.")
            return
        if provider == "anthropic" and not settings.get(KEY_ANTHROPIC, "").strip():
            QMessageBox.warning(self, "Chat", "Add an Anthropic API key in AI settings.")
            return
        if provider == "grok" and not settings.get(KEY_GROK, "").strip():
            QMessageBox.warning(self, "Chat", "Add a Grok (xAI) API key in AI settings.")
            return
        if provider == "gemini" and not settings.get(KEY_GEMINI, "").strip():
            QMessageBox.warning(self, "Chat", "Add a Gemini API key in AI settings.")
            return
        if provider == "ollama" and not model.strip():
            QMessageBox.warning(self, "Chat", "Choose an Ollama model (or type one).")
            return

        tool_lines: list[str] = []
        mentioned_labels, mentioned_context = self._mentioned_file_context(text)
        if bool(self._composer_tool_state.get("web", False)):
            tool_lines.append("Web tool: enabled (UI-only in cvops local workspace).")
        if bool(self._composer_tool_state.get("rag", False)):
            tool_lines.append("RAG preference: use project RAG context if available.")
        if self._composer_attachments:
            names = [Path(p).name for p in self._composer_attachments[:12]]
            more = "" if len(self._composer_attachments) <= 12 else f" (+{len(self._composer_attachments) - 12} more)"
            tool_lines.append(f"Attached files: {', '.join(names)}{more}")
        if mentioned_labels:
            names = mentioned_labels[:12]
            more = "" if len(mentioned_labels) <= 12 else f" (+{len(mentioned_labels) - 12} more)"
            tool_lines.append(f"Resolved @files: {', '.join(names)}{more}")
        outbound_text = text
        context_parts: list[str] = []
        if tool_lines:
            context_parts.append("[workspace context]\n" + "\n".join(f"- {ln}" for ln in tool_lines))
        if mentioned_context:
            context_parts.append("[file context]\n" + mentioned_context)
        if bool(self._composer_tool_state.get("rag", False)):
            memory_block = self._ingested_memory_context_block(text)
            if memory_block:
                context_parts.append(memory_block)
        if context_parts:
            outbound_text = f"{text}\n\n" + "\n\n".join(context_parts)

        self._chat_mgr.add_message(self._current_chat_id, "user", outbound_text)
        self.chat_input.clear()
        self._streaming_assistant = ""
        self._streaming_error = ""
        self._streaming_model_label = model_label
        self._streaming_provider = provider
        self._streaming_mcp_enabled = provider == "ollama" and self._should_offer_tacitus_mcp_tools(text)
        self._show_current_chat()

        msgs = self._chat_mgr.get_chat_messages(self._current_chat_id)

        # Global system prompt from AI settings, applied to every provider. Cloud
        # providers receive it as a leading system message; Ollama takes it via its
        # dedicated system_prompt field.
        sys_prompt = system_prompt(settings)
        cloud_msgs = ([{"role": "system", "content": sys_prompt}] + msgs) if sys_prompt else msgs

        worker: Optional[QThread] = None
        if provider == "ollama":
            mcp_context = self._tacitus_mcp_context() if self._streaming_mcp_enabled else None
            mcp_catalog = _compact_tacitus_mcp_catalog() if self._streaming_mcp_enabled else None
            prompt = _build_ollama_prompt(
                msgs,
                assistant_name=self.assistant_name(),
                mcp_context=mcp_context,
                mcp_catalog=mcp_catalog,
            )
            base = self.chat_ollama_url.text().strip().rstrip("/")
            worker = OllamaWorker(
                base_url=base,
                model=model.strip(),
                prompt=prompt,
                system_prompt=sys_prompt or None,
            )
        elif provider == "openai":
            omsgs = chat_messages_to_openai(cloud_msgs)
            if not omsgs:
                self._streaming_error = "No messages to send."
                self._show_current_chat()
                return
            worker = OpenAICompatChatWorker(
                base_url="https://api.openai.com/v1",
                api_key=settings[KEY_OPENAI],
                model=model.strip(),
                messages=omsgs,
            )
        elif provider == "grok":
            omsgs = chat_messages_to_openai(cloud_msgs)
            if not omsgs:
                self._streaming_error = "No messages to send."
                self._show_current_chat()
                return
            worker = OpenAICompatChatWorker(
                base_url="https://api.x.ai/v1",
                api_key=settings[KEY_GROK],
                model=model.strip(),
                messages=omsgs,
            )
        elif provider == "anthropic":
            system, amsgs = chat_messages_to_anthropic(cloud_msgs)
            if not amsgs:
                self._streaming_error = "No messages to send."
                self._show_current_chat()
                return
            worker = AnthropicChatWorker(
                api_key=settings[KEY_ANTHROPIC],
                model=model.strip(),
                system=system,
                messages=amsgs,
            )
        elif provider == "gemini":
            contents = chat_messages_to_gemini(cloud_msgs)
            if not contents:
                self._streaming_error = "No messages to send."
                self._show_current_chat()
                return
            worker = GeminiChatWorker(
                api_key=settings[KEY_GEMINI],
                model=model.strip(),
                contents=contents,
            )
        else:
            self._streaming_error = f"Unknown provider: {provider}"
            self._streaming_provider = ""
            self._streaming_mcp_enabled = False
            self._show_current_chat()
            return

        # Show an empty assistant bubble; tokens stream into it via
        # _on_chat_token, which re-renders the whole chat HTML each time.
        self._is_streaming = True
        self._sync_chat_busy_ui()
        self._show_current_chat()
        self._chat_worker = worker
        worker.token_received.connect(self._on_chat_token)
        worker.response_received.connect(self._on_chat_done)
        worker.error_occurred.connect(self._on_chat_err)
        worker.start()

    def _on_chat_token(self, tok: str) -> None:
        self._streaming_assistant += str(tok or "")
        self._show_current_chat()

    def _on_chat_done(self, payload: dict) -> None:
        full = str(payload.get("full_response", "")).strip()
        if self._current_chat_id and full and self._chat_mgr is not None:
            metadata: dict[str, Any] = {"model_label": self._streaming_model_label}
            saved_text = full
            if self._streaming_provider == "ollama" and self._streaming_mcp_enabled:
                dispatch = self._maybe_dispatch_tacitus_mcp_model_response(full)
                if dispatch is not None:
                    saved_text = self._format_tacitus_mcp_tool_reply(dispatch)
                    metadata.update(
                        {
                            "model_label": "Tacitus MCP",
                            "mcp_model_label": self._streaming_model_label,
                            "mcp_tool_call": dispatch.get("call") if isinstance(dispatch.get("call"), dict) else {},
                            "mcp_result": dispatch.get("result") if isinstance(dispatch.get("result"), dict) else {},
                        }
                    )
            self._chat_mgr.add_message(
                self._current_chat_id,
                "assistant",
                saved_text,
                metadata=metadata,
            )
        self._is_streaming = False
        self._streaming_assistant = ""
        self._streaming_error = ""
        self._streaming_model_label = ""
        self._streaming_provider = ""
        self._streaming_mcp_enabled = False
        self._chat_worker = None
        self._sync_chat_busy_ui()
        self._show_current_chat()

    def _on_chat_err(self, msg: str) -> None:
        self._streaming_error = str(msg or "")
        self._streaming_assistant = ""
        self._streaming_model_label = ""
        self._streaming_provider = ""
        self._streaming_mcp_enabled = False
        self._is_streaming = False
        self._chat_worker = None
        self._sync_chat_busy_ui()
        self._show_current_chat()
        self.errorRaised.emit(f"Notes chat: {msg}")

    def _sync_chat_busy_ui(self) -> None:
        busy = bool(self._is_streaming and self._chat_worker is not None)
        if hasattr(self, "_btn_compose_send"):
            self._btn_compose_send.setEnabled(not busy)
            self._btn_compose_send.setText("Sending…" if busy else "Send")
        if hasattr(self, "_btn_compose_stop"):
            self._btn_compose_stop.setEnabled(busy)
        if hasattr(self, "chat_input"):
            self.chat_input.setReadOnly(busy)
        self._refresh_chat_list()

    def _stop_chat_generation(self) -> None:
        worker = self._chat_worker
        if worker is None:
            return
        # Try the worker's own cancel hook first (cooperative); fall back to a
        # disconnect + quit so any in-flight tokens can't land in this view.
        for fn_name in ("cancel", "stop", "requestInterruption"):
            fn = getattr(worker, fn_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        try:
            worker.token_received.disconnect(self._on_chat_token)
        except Exception:
            pass
        try:
            worker.response_received.disconnect(self._on_chat_done)
        except Exception:
            pass
        try:
            worker.error_occurred.disconnect(self._on_chat_err)
        except Exception:
            pass
        # If the user already saw partial tokens, persist them so the turn
        # isn't lost on stop. Empty assistant turn is dropped.
        partial = self._streaming_assistant
        if partial.strip() and self._current_chat_id and self._chat_mgr is not None:
            self._chat_mgr.add_message(
                self._current_chat_id,
                "assistant",
                partial.rstrip() + "\n\n_(stopped)_",
                metadata={"model_label": self._streaming_model_label},
            )
        try:
            worker.quit()
        except Exception:
            pass
        self._chat_worker = None
        self._is_streaming = False
        self._streaming_assistant = ""
        self._streaming_error = ""
        self._streaming_model_label = ""
        self._streaming_provider = ""
        self._streaming_mcp_enabled = False
        self._sync_chat_busy_ui()
        self._show_current_chat()

    def _on_sidebar_search_changed(self, text: str) -> None:
        """Filter the active sidebar lists by case-insensitive title substring."""
        needle = (text or "").strip().lower()
        projects_mode = bool(getattr(self, "_btn_mode_projects", None) and self._btn_mode_projects.isChecked())
        lists = (
            (getattr(self, "project_list_pinned", None), getattr(self, "_project_pinned_header", None)),
            (getattr(self, "project_list_recent", None), getattr(self, "_project_recent_header", None)),
        ) if projects_mode else (
            (getattr(self, "chat_list_pinned", None), getattr(self, "_chat_pinned_header", None)),
            (getattr(self, "chat_list_recent", None), getattr(self, "_chat_recent_header", None)),
        )
        for listw, hdr in lists:
            if listw is None:
                continue
            any_visible = False
            for i in range(listw.count()):
                item = listw.item(i)
                if item is None:
                    continue
                host = listw.itemWidget(item)
                title_lab = getattr(host, "_title", None) if host is not None else None
                title = ""
                if title_lab is not None:
                    title = str(getattr(title_lab, "_raw_text", "") or "").lower()
                if not title:
                    raw = item.data(Qt.ItemDataRole.UserRole)
                    title = str(raw or "").lower()
                visible = (not needle) or (needle in title)
                item.setHidden(not visible)
                any_visible = any_visible or visible
            if hdr is not None:
                hdr.setVisible(any_visible and listw.count() > 0)
        pinned_list = getattr(self, "project_list_pinned", None) if projects_mode else getattr(self, "chat_list_pinned", None)
        self._sync_pinned_list_height(pinned_list, max_rows=3)
