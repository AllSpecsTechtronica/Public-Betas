"""Import (ingest) exported chat histories from other AIs into a notes chats dir.

Supports the two export shapes users most commonly have on disk:

* ChatGPT ("Export data" -> ``conversations.json`` inside a ``.zip``). Each
  conversation is a node ``mapping`` tree; messages carry epoch ``create_time``.
* Claude ("Export data" -> ``conversations.json`` inside a ``.zip``). Each
  conversation has a flat ``chat_messages`` list with ISO ``created_at``.

The parsers are intentionally Qt-free so they can be unit tested and reused.
Every imported chat is written in the exact on-disk format that
``ChatManager`` (``solo_rag_chat.chat_manager``) loads, and crucially the
original timestamps are preserved ("back-dating") on the chat envelope
(``created_at`` / ``updated_at``) and on each message so the conversation lands
in the timeline where it actually happened rather than at import time.
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Batching defaults: a giant export (ChatGPT histories can hold thousands of
# conversations) is written in small batches with a short pause between them so
# the disk/CPU are never saturated in one burst — the import progresses "over
# time" instead of freezing the machine. ``ProgressFn`` reports cumulative
# (done, total) conversations so callers can render a progress bar + ETA.
ProgressFn = Callable[[int, int], None]
DEFAULT_BATCH_SIZE = 50
DEFAULT_BATCH_PAUSE = 0.0  # seconds between batches; UI passes a small value

# Source tags stored on the chat/message metadata so a chat can always be traced
# back to the export it came from.
SOURCE_CHATGPT = "chatgpt"
SOURCE_CLAUDE = "claude"

_CONVERSATIONS_MEMBER = "conversations.json"
_PROJECTS_MEMBER = "projects.json"


@dataclass
class ImportResult:
    """Outcome of an ingest run, suitable for a short user-facing summary."""

    source: str = ""
    chats_written: int = 0
    messages_written: int = 0
    skipped_empty: int = 0
    skipped_duplicate: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.chats_written > 0 or not self.errors

    def summary_line(self) -> str:
        label = {SOURCE_CHATGPT: "ChatGPT", SOURCE_CLAUDE: "Claude"}.get(
            self.source, self.source or "AI export"
        )
        bits = [
            f"{self.chats_written} conversation(s)",
            f"{self.messages_written} message(s)",
        ]
        if self.skipped_duplicate:
            bits.append(f"{self.skipped_duplicate} already imported")
        if self.skipped_empty:
            bits.append(f"{self.skipped_empty} empty skipped")
        return f"{label}: imported {', '.join(bits)}."


# --------------------------------------------------------------------------- #
# Timestamp helpers (back-dating)
# --------------------------------------------------------------------------- #

def _iso_local(dt: datetime) -> str:
    """Render ``dt`` as a naive-local ISO string matching ChatManager's style."""
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt.isoformat()


def _epoch_to_iso(value: Any) -> Optional[str]:
    try:
        ts = float(value)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    # Treat as UTC epoch, convert to local naive for consistency with the app.
    return _iso_local(datetime.fromtimestamp(ts, tz=timezone.utc))


def _iso_normalize(value: Any) -> Optional[str]:
    """Normalize an ISO-8601 string (possibly ``...Z``) to naive-local ISO."""
    text = str(value or "").strip()
    if not text:
        return None
    candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return _iso_local(dt)


def _any_timestamp(value: Any) -> Optional[str]:
    """Accept either an epoch number or an ISO string."""
    if isinstance(value, (int, float)):
        return _epoch_to_iso(value)
    return _iso_normalize(value)


# --------------------------------------------------------------------------- #
# Raw export loading (file / zip / dir)
# --------------------------------------------------------------------------- #

def _read_conversations_json_from_zip(path: Path) -> Optional[Any]:
    try:
        with zipfile.ZipFile(path) as zf:
            member = None
            for name in zf.namelist():
                if name.rsplit("/", 1)[-1] == _CONVERSATIONS_MEMBER:
                    member = name
                    break
            if member is None:
                return None
            with zf.open(member) as fh:
                return json.loads(fh.read().decode("utf-8"))
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError, KeyError):
        return None


def resolve_export_source(path: Path) -> Optional[Path]:
    """Return the actual file to read (the ``.zip`` or ``conversations.json``).

    Folders are resolved to the first ``conversations.json`` found within. Used
    by both the loader and the size estimator so they always agree on what file
    is about to be ingested.
    """
    path = Path(path).expanduser()
    if path.is_dir():
        candidate = path / _CONVERSATIONS_MEMBER
        if candidate.is_file():
            return candidate
        for child in sorted(path.rglob(_CONVERSATIONS_MEMBER)):
            return child
        return None
    if not path.exists():
        return None
    return path


def load_export_payload(path: Path) -> Optional[Any]:
    """Load the parsed JSON conversations array from a file, zip, or folder.

    Returns the decoded JSON (typically a list of conversations) or ``None`` if
    nothing usable was found.
    """
    src = resolve_export_source(path)
    if src is None:
        return None
    if src.suffix.lower() == ".zip":
        return _read_conversations_json_from_zip(src)
    try:
        return json.loads(src.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Size estimation (pre-ingest guard against zip bombs / out-of-space)
# --------------------------------------------------------------------------- #

@dataclass
class ExportSizeInfo:
    """Measured footprint of an export, for a pre-ingest confirmation dialog."""

    resolved_path: Optional[Path]
    source_bytes: int  # on-disk size of the selected file (compressed, for zips)
    payload_bytes: int  # UNCOMPRESSED conversations.json bytes -> what we parse/write
    is_archive: bool

    @property
    def compression_ratio(self) -> float:
        if self.source_bytes <= 0:
            return 1.0
        return self.payload_bytes / self.source_bytes


def human_bytes(num: float) -> str:
    """Human-readable byte size (B/KB/MB/GB/TB/PB)."""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} PB"


def measure_export(path: Path, member_names: Optional[list[str]] = None) -> ExportSizeInfo:
    """Measure an export without extracting it.

    For ``.zip`` exports the *uncompressed* size of the relevant member(s) is
    read from the central directory — this is the figure that matters for a zip
    bomb, since a few-KB archive can expand to gigabytes. ``member_names``
    defaults to just ``conversations.json``; the Claude-projects flow also counts
    ``projects.json`` (knowledge docs).
    """
    names = set(member_names or [_CONVERSATIONS_MEMBER])
    path = Path(path).expanduser()
    if path.is_dir():
        source_bytes = 0
        for name in names:
            for child in sorted(path.rglob(name)):
                try:
                    source_bytes += child.stat().st_size
                except OSError:
                    pass
        return ExportSizeInfo(path, source_bytes, source_bytes, False)
    src = resolve_export_source(path)
    if src is None:
        return ExportSizeInfo(None, 0, 0, False)
    try:
        source_bytes = src.stat().st_size
    except OSError:
        source_bytes = 0
    if src.suffix.lower() == ".zip":
        payload = 0
        try:
            with zipfile.ZipFile(src) as zf:
                for info in zf.infolist():
                    if info.filename.rsplit("/", 1)[-1] in names:
                        payload += int(info.file_size)
        except (zipfile.BadZipFile, OSError):
            payload = 0
        return ExportSizeInfo(src, source_bytes, payload, True)
    return ExportSizeInfo(src, source_bytes, source_bytes, False)


def free_space_bytes(target_dir: Path) -> int:
    """Free bytes on the volume holding ``target_dir`` (walks up to an existing parent)."""
    p = Path(target_dir)
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    try:
        return int(shutil.disk_usage(p).free)
    except OSError:
        return 0


def volume_label(path: Path) -> str:
    """Human name of the volume holding ``path`` (e.g. ``ExternalSSD``).

    Finds the mount point by walking up until the device id changes, then uses
    its directory name (``/Volumes/ExternalSSD`` -> ``ExternalSSD``). Falls back
    to ``Macintosh HD`` for the system root and the raw path otherwise.
    """
    p = Path(path).expanduser()
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    try:
        dev = p.stat().st_dev
    except OSError:
        return ""
    mount = p
    while mount.parent != mount:
        try:
            if mount.parent.stat().st_dev != dev:
                break
        except OSError:
            break
        mount = mount.parent
    name = mount.name
    if name:
        return name
    return "Macintosh HD" if str(mount) == "/" else str(mount)


def detect_source(payload: Any) -> Optional[str]:
    """Best-effort detection of which AI produced ``payload``."""
    convs = _as_conversation_list(payload)
    if not convs:
        return None
    sample = convs[0]
    if not isinstance(sample, dict):
        return None
    if "mapping" in sample:
        return SOURCE_CHATGPT
    if "chat_messages" in sample:
        return SOURCE_CLAUDE
    # ChatGPT uses create_time floats; Claude uses ISO created_at + uuid/name.
    if "uuid" in sample and ("name" in sample or "summary" in sample):
        return SOURCE_CLAUDE
    if "create_time" in sample:
        return SOURCE_CHATGPT
    return None


def _as_conversation_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("conversations", "data", "items"):
            inner = payload.get(key)
            if isinstance(inner, list):
                return inner
    return []


# --------------------------------------------------------------------------- #
# ChatGPT parsing
# --------------------------------------------------------------------------- #

def _chatgpt_part_text(content: Any) -> str:
    if not isinstance(content, dict):
        return ""
    ctype = content.get("content_type")
    if ctype in (None, "text"):
        parts = content.get("parts") or []
        out = [str(p) for p in parts if isinstance(p, str) and p.strip()]
        return "\n".join(out).strip()
    # Skip non-text parts (code interpreter outputs, images, tool json, etc.).
    return ""


def _chatgpt_ordered_nodes(mapping: dict[str, Any], current_node: Any) -> list[dict[str, Any]]:
    """Linearize the mapping tree to the current leaf via parent links.

    Falls back to create_time ordering of all message nodes when the parent
    chain is unusable.
    """
    if isinstance(current_node, str) and current_node in mapping:
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        node_id: Optional[str] = current_node
        while node_id and node_id in mapping and node_id not in seen:
            seen.add(node_id)
            node = mapping[node_id]
            if isinstance(node, dict):
                chain.append(node)
                node_id = node.get("parent")
            else:
                break
        if chain:
            chain.reverse()
            return chain
    nodes = [n for n in mapping.values() if isinstance(n, dict) and n.get("message")]
    nodes.sort(key=lambda n: float((n.get("message") or {}).get("create_time") or 0.0))
    return nodes


def parse_chatgpt(payload: Any) -> list[dict[str, Any]]:
    """Convert a ChatGPT export into normalized chat dicts (ChatManager format)."""
    chats: list[dict[str, Any]] = []
    for conv in _as_conversation_list(payload):
        if not isinstance(conv, dict):
            continue
        mapping = conv.get("mapping")
        if not isinstance(mapping, dict):
            continue
        title = str(conv.get("title") or "").strip() or "ChatGPT conversation"
        created = _epoch_to_iso(conv.get("create_time"))
        updated = _epoch_to_iso(conv.get("update_time")) or created
        messages: list[dict[str, Any]] = []
        for node in _chatgpt_ordered_nodes(mapping, conv.get("current_node")):
            msg = node.get("message")
            if not isinstance(msg, dict):
                continue
            author = msg.get("author") or {}
            role = str(author.get("role") or "").strip().lower()
            if role not in ("user", "assistant"):
                continue
            text = _chatgpt_part_text(msg.get("content"))
            if not text:
                continue
            ts = _epoch_to_iso(msg.get("create_time")) or created
            messages.append(_make_message(role, text, ts))
        if not messages:
            continue
        if not created:
            created = messages[0].get("timestamp")
        if not updated:
            updated = messages[-1].get("timestamp")
        chats.append(
            _make_chat(
                title=title,
                created_at=created,
                updated_at=updated,
                messages=messages,
                source=SOURCE_CHATGPT,
                original_id=str(conv.get("id") or conv.get("conversation_id") or ""),
            )
        )
    return chats


# --------------------------------------------------------------------------- #
# Claude parsing
# --------------------------------------------------------------------------- #

def _claude_message_text(msg: dict[str, Any]) -> str:
    # Newer exports use a content blocks list; older use a flat "text" field.
    blocks = msg.get("content")
    if isinstance(blocks, list):
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                t = str(block.get("text") or "").strip()
                if t:
                    parts.append(t)
        if parts:
            return "\n".join(parts).strip()
    return str(msg.get("text") or "").strip()


def parse_claude(payload: Any) -> list[dict[str, Any]]:
    """Convert a Claude export into normalized chat dicts (ChatManager format)."""
    chats: list[dict[str, Any]] = []
    for conv in _as_conversation_list(payload):
        if not isinstance(conv, dict):
            continue
        raw_messages = conv.get("chat_messages")
        if not isinstance(raw_messages, list):
            continue
        title = str(conv.get("name") or "").strip() or "Claude conversation"
        created = _iso_normalize(conv.get("created_at"))
        updated = _iso_normalize(conv.get("updated_at")) or created
        messages: list[dict[str, Any]] = []
        for msg in raw_messages:
            if not isinstance(msg, dict):
                continue
            sender = str(msg.get("sender") or "").strip().lower()
            role = {"human": "user", "assistant": "assistant"}.get(sender)
            if role is None:
                continue
            text = _claude_message_text(msg)
            if not text:
                continue
            ts = _iso_normalize(msg.get("created_at")) or created
            messages.append(_make_message(role, text, ts))
        if not messages:
            continue
        if not created:
            created = messages[0].get("timestamp")
        if not updated:
            updated = messages[-1].get("timestamp")
        chats.append(
            _make_chat(
                title=title,
                created_at=created,
                updated_at=updated,
                messages=messages,
                source=SOURCE_CLAUDE,
                original_id=str(conv.get("uuid") or ""),
            )
        )
    return chats


def parse_export(payload: Any, source: Optional[str] = None) -> tuple[str, list[dict[str, Any]]]:
    """Parse ``payload`` into ``(source, chats)``; auto-detects when ``source`` is None."""
    src = source or detect_source(payload)
    if src == SOURCE_CHATGPT:
        return SOURCE_CHATGPT, parse_chatgpt(payload)
    if src == SOURCE_CLAUDE:
        return SOURCE_CLAUDE, parse_claude(payload)
    return "", []


# --------------------------------------------------------------------------- #
# Normalized chat construction + writing
# --------------------------------------------------------------------------- #

def _make_message(role: str, content: str, timestamp: Optional[str]) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "timestamp": timestamp or _iso_local(datetime.now()),
        "metadata": {},
    }


def _make_chat(
    *,
    title: str,
    created_at: Optional[str],
    updated_at: Optional[str],
    messages: list[dict[str, Any]],
    source: str,
    original_id: str,
) -> dict[str, Any]:
    now = _iso_local(datetime.now())
    return {
        "title": title[:120],
        "description": f"Imported from {source} export.",
        "created_at": created_at or now,
        "updated_at": updated_at or created_at or now,
        "messages": messages,
        "metadata": {
            "imported": True,
            "import_source": source,
            "original_id": original_id,
            "imported_at": now,
        },
    }


def _chat_id_for(chat: dict[str, Any], used: set[str]) -> str:
    """A unique ``chat_<ms>`` id derived from the back-dated created_at.

    Basing the id on the original time keeps files roughly time-ordered on disk;
    a counter suffix guarantees uniqueness within a single import batch.
    """
    base_iso = str(chat.get("created_at") or "")
    stamp = None
    norm = _iso_normalize(base_iso)
    if norm:
        try:
            stamp = int(datetime.fromisoformat(norm).timestamp() * 1000)
        except ValueError:
            stamp = None
    if stamp is None:
        stamp = int(datetime.now().timestamp() * 1000)
    candidate = f"chat_{stamp}"
    suffix = 0
    while candidate in used:
        suffix += 1
        candidate = f"chat_{stamp}_{suffix}"
    used.add(candidate)
    return candidate


def _existing_import_keys(chats_dir: Path) -> set[tuple[str, str]]:
    """``(import_source, original_id)`` for chats already imported into ``chats_dir``.

    Used to make re-imports idempotent: a conversation that was ingested before
    (matched on its export-native id) is skipped instead of duplicated.
    """
    keys: set[tuple[str, str]] = set()
    for path in chats_dir.glob("chat_*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        meta = data.get("metadata") if isinstance(data, dict) else None
        if not isinstance(meta, dict):
            continue
        orig = str(meta.get("original_id") or "").strip()
        if orig:
            keys.add((str(meta.get("import_source") or ""), orig))
    return keys


def _import_key(chat: dict[str, Any]) -> Optional[tuple[str, str]]:
    meta = chat.get("metadata") or {}
    orig = str(meta.get("original_id") or "").strip()
    if not orig:
        return None
    return (str(meta.get("import_source") or ""), orig)


def write_chats_to_dir(
    chats: Iterable[dict[str, Any]],
    chats_dir: Path,
    *,
    source: str = "",
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_pause: float = DEFAULT_BATCH_PAUSE,
    on_chat: Optional[Callable[[], None]] = None,
) -> ImportResult:
    """Write normalized chats into ``chats_dir`` as ``chat_*.json`` files.

    Conversations whose ``original_id`` already exists in ``chats_dir`` are
    skipped so importing the same export twice does not create duplicates.

    Writing happens in batches of ``batch_size`` with a ``batch_pause`` second
    sleep between batches so a huge export never saturates the disk in one burst
    (which is what makes the machine lag). ``on_chat`` is invoked once per
    conversation processed (written, skipped, or errored) so a caller can drive a
    progress bar / ETA.
    """
    result = ImportResult(source=source)
    chats_dir = Path(chats_dir)
    chats_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = {p.stem for p in chats_dir.glob("chat_*.json")}
    seen_keys = _existing_import_keys(chats_dir)
    processed = 0
    for chat in chats:
        processed += 1
        try:
            msgs = chat.get("messages") or []
            if not msgs:
                result.skipped_empty += 1
                continue
            key = _import_key(chat)
            if key is not None and key in seen_keys:
                result.skipped_duplicate += 1
                continue
            if key is not None:
                seen_keys.add(key)
            chat_id = _chat_id_for(chat, used)
            record = dict(chat)
            record["id"] = chat_id
            try:
                target = chats_dir / f"{chat_id}.json"
                target.write_text(
                    json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except OSError as exc:
                result.errors.append(f"{chat.get('title', chat_id)}: {exc}")
                continue
            result.chats_written += 1
            result.messages_written += len(msgs)
        finally:
            if on_chat is not None:
                on_chat()
            # Pace the work: yield the disk/CPU between batches.
            if batch_pause > 0 and batch_size > 0 and processed % batch_size == 0:
                time.sleep(batch_pause)
    return result


def ingest_export_file(
    path: Path,
    chats_dir: Path,
    source: Optional[str] = None,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_pause: float = DEFAULT_BATCH_PAUSE,
    on_progress: Optional[ProgressFn] = None,
) -> ImportResult:
    """End-to-end: load an export file/zip/folder and write its chats to ``chats_dir``.

    ``on_progress(done, total)`` is called as conversations are written so the UI
    can show batched progress + an estimated time remaining for a large export.
    """
    payload = load_export_payload(Path(path))
    if payload is None:
        return ImportResult(
            source=source or "",
            errors=["Could not read a conversations.json from the selected file."],
        )
    detected, chats = parse_export(payload, source=source)
    if not chats:
        return ImportResult(
            source=detected or source or "",
            errors=[
                "No conversations were found. Expected a ChatGPT or Claude data export."
            ],
        )
    chats = list(chats)
    total = len(chats)
    if on_progress is not None:
        on_progress(0, total)
    done = 0

    def _tick() -> None:
        nonlocal done
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    return write_chats_to_dir(
        chats,
        chats_dir,
        source=detected,
        batch_size=batch_size,
        batch_pause=batch_pause,
        on_chat=_tick,
    )


# --------------------------------------------------------------------------- #
# Claude Projects: respawn a project (instructions + knowledge + its chats)
# --------------------------------------------------------------------------- #

def load_export_members(path: Path, names: list[str]) -> dict[str, Any]:
    """Load several named JSON members (e.g. conversations.json + projects.json).

    Works against a ``.zip`` export, an unpacked folder, or a single JSON file
    (in which case sibling files in the same folder are tried for the others).
    """
    path = Path(path).expanduser()
    out: dict[str, Any] = {}

    def _safe(p: Path) -> Optional[Any]:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None

    if path.is_dir():
        for name in names:
            cand = path / name
            if not cand.is_file():
                for child in sorted(path.rglob(name)):
                    cand = child
                    break
            if cand.is_file():
                out[name] = _safe(cand)
        return out
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as zf:
                by_base = {n.rsplit("/", 1)[-1]: n for n in zf.namelist()}
                for name in names:
                    if name in by_base:
                        try:
                            with zf.open(by_base[name]) as fh:
                                out[name] = json.loads(fh.read().decode("utf-8"))
                        except (KeyError, json.JSONDecodeError, OSError):
                            continue
        except (zipfile.BadZipFile, OSError):
            pass
        return out
    if path.is_file():
        data = _safe(path)
        if path.name in names:
            out[path.name] = data
        elif names:
            out[names[0]] = data
        for name in names:
            if name in out:
                continue
            sib = path.parent / name
            if sib.is_file():
                out[name] = _safe(sib)
    return out


@dataclass
class ProjectsImportResult:
    """Outcome of a Claude Projects respawn."""

    projects_created: int = 0
    projects_reused: int = 0
    docs_written: int = 0
    chats_written: int = 0
    messages_written: int = 0
    skipped_duplicate: int = 0
    unmapped_chats: int = 0
    errors: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        bits = [
            f"{self.projects_created} project(s) created",
        ]
        if self.projects_reused:
            bits.append(f"{self.projects_reused} updated")
        bits.append(f"{self.docs_written} knowledge file(s)")
        bits.append(f"{self.chats_written} conversation(s)")
        if self.unmapped_chats:
            bits.append(f"{self.unmapped_chats} unlinked chat(s) -> current project")
        if self.skipped_duplicate:
            bits.append(f"{self.skipped_duplicate} already imported")
        return "Claude projects: " + ", ".join(bits) + "."


def _first_nonempty(*values: Any) -> str:
    for v in values:
        s = str(v or "").strip()
        if s:
            return s
    return ""


def parse_claude_projects(payload: Any) -> list[dict[str, Any]]:
    """Normalize Claude ``projects.json`` into respawn-ready project dicts."""
    out: list[dict[str, Any]] = []
    for proj in _as_conversation_list(payload):
        if not isinstance(proj, dict):
            continue
        name = _first_nonempty(proj.get("name"), proj.get("title")) or "Claude project"
        instructions = _first_nonempty(
            proj.get("prompt_template"),
            proj.get("custom_instructions"),
            proj.get("instructions"),
            proj.get("system_prompt"),
        )
        description = _first_nonempty(proj.get("description"), proj.get("summary"))
        created = _iso_normalize(proj.get("created_at"))
        updated = _iso_normalize(proj.get("updated_at")) or created
        docs: list[dict[str, Any]] = []
        for d in proj.get("docs") or proj.get("documents") or []:
            if not isinstance(d, dict):
                continue
            content = d.get("content")
            if not isinstance(content, str):
                content = d.get("text")
            if not isinstance(content, str) or not content.strip():
                continue
            filename = _first_nonempty(
                d.get("filename"), d.get("file_name"), d.get("name"), d.get("uuid")
            ) or "document"
            docs.append(
                {
                    "filename": filename,
                    "content": content,
                    "created_at": _iso_normalize(d.get("created_at")),
                }
            )
        out.append(
            {
                "name": name,
                "instructions": instructions,
                "description": description,
                "created_at": created,
                "updated_at": updated,
                "original_id": str(proj.get("uuid") or proj.get("id") or ""),
                "docs": docs,
            }
        )
    return out


def _conversation_project_map(conversations_payload: Any) -> dict[str, str]:
    """conversation uuid -> project uuid, from whatever linkage the export carries."""
    mapping: dict[str, str] = {}
    for conv in _as_conversation_list(conversations_payload):
        if not isinstance(conv, dict):
            continue
        cuid = str(conv.get("uuid") or "").strip()
        if not cuid:
            continue
        pid = conv.get("project_uuid") or conv.get("project_id")
        if not pid:
            proj = conv.get("project")
            if isinstance(proj, dict):
                pid = proj.get("uuid") or proj.get("id")
        pid = str(pid or "").strip()
        if pid:
            mapping[cuid] = pid
    return mapping


def _safe_doc_filename(raw: str) -> str:
    name = str(raw or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(c for c in name if c.isprintable()).strip()
    if not name:
        return ""
    if "." not in name:
        name += ".md"
    return name[:120]


def _unique_path(folder: Path, filename: str) -> Path:
    dst = folder / filename
    if not dst.exists():
        return dst
    stem, dot, suffix = filename.partition(".")
    n = 2
    while True:
        cand = folder / (f"{stem}_{n}.{suffix}" if dot else f"{stem}_{n}")
        if not cand.exists():
            return cand
        n += 1


def _write_project_docs(space_root: Path, docs: list[dict[str, Any]]) -> int:
    """Write knowledge docs into ``<space>/files/``; idempotent on identical content."""
    if not docs:
        return 0
    files_dir = space_root / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for d in docs:
        filename = _safe_doc_filename(d.get("filename", ""))
        if not filename:
            continue
        content = str(d.get("content") or "")
        dst = files_dir / filename
        if dst.exists():
            try:
                if dst.read_text(encoding="utf-8", errors="ignore") == content:
                    continue  # already imported, unchanged
            except OSError:
                pass
            dst = _unique_path(files_dir, filename)
        try:
            dst.write_text(content, encoding="utf-8")
            written += 1
        except OSError:
            continue
    return written


def _write_imported_space_meta(space_root: Path, space_id: str, proj: dict[str, Any]) -> None:
    """Persist project name + instructions ("physics") + import provenance to meta.json."""
    meta = space_root / "meta.json"
    try:
        raw = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        if not isinstance(raw, dict):
            raw = {}
    except (OSError, json.JSONDecodeError):
        raw = {}
    goals_parts = [p for p in (proj.get("instructions"), proj.get("description")) if p]
    raw["id"] = space_id
    raw["title"] = str(proj.get("name") or space_id)[:120]
    raw["goals"] = "\n\n".join(goals_parts).strip()
    raw["created_at"] = proj.get("created_at") or raw.get("created_at") or _iso_local(datetime.now())
    raw["imported"] = True
    raw["import_source"] = SOURCE_CLAUDE
    raw["import_kind"] = "project"
    raw["original_id"] = str(proj.get("original_id") or "")
    raw["imported_at"] = _iso_local(datetime.now())
    try:
        meta.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _existing_imported_projects(spaces_root: Path) -> dict[str, str]:
    """Map original Claude project uuid -> existing space id (for idempotent respawn)."""
    out: dict[str, str] = {}
    if not spaces_root.is_dir():
        return out
    for child in sorted(spaces_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        meta = child / "meta.json"
        try:
            raw = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        if raw.get("import_source") == SOURCE_CLAUDE and raw.get("import_kind") == "project":
            orig = str(raw.get("original_id") or "").strip()
            if orig:
                out[orig] = child.name
    return out


def ingest_claude_projects(
    path: Path,
    spaces_root: Path,
    vault_root: Path,
    *,
    fallback_chats_dir: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    batch_pause: float = DEFAULT_BATCH_PAUSE,
    on_progress: Optional[ProgressFn] = None,
) -> ProjectsImportResult:
    """Respawn Claude projects from an export: instructions, knowledge, and chats.

    Each Claude project becomes a notes space with the same name and custom
    instructions (its "physics"/baseline), its knowledge docs land in the
    space's ``files/`` folder, and the project's conversations are back-dated
    into that space's chat history. Re-running is idempotent: existing imported
    projects are updated in place and already-imported chats are skipped.
    Conversations with no project link go to ``fallback_chats_dir`` (the project
    that was open at import time).
    """
    from .notes_spaces import create_space, notes_chats_dir

    result = ProjectsImportResult()
    spaces_root = Path(spaces_root)
    vault_root = Path(vault_root)

    members = load_export_members(path, [_CONVERSATIONS_MEMBER, _PROJECTS_MEMBER])
    projects = parse_claude_projects(members.get(_PROJECTS_MEMBER))
    conv_payload = members.get(_CONVERSATIONS_MEMBER)
    if not projects:
        result.errors.append(
            "No projects.json with Claude projects was found in the selected export."
        )
        return result

    _, all_chats = SOURCE_CLAUDE, parse_claude(conv_payload)
    chats_by_uuid: dict[str, dict[str, Any]] = {}
    for chat in all_chats:
        cu = str((chat.get("metadata") or {}).get("original_id") or "")
        if cu:
            chats_by_uuid[cu] = chat
    conv_map = _conversation_project_map(conv_payload)

    # Cumulative progress across every per-project write_chats_to_dir call plus
    # the leftover pass, so the UI sees one smooth (done, total) ramp.
    total_chats = len(chats_by_uuid)
    done_chats = 0
    if on_progress is not None:
        on_progress(0, total_chats)

    def _tick() -> None:
        nonlocal done_chats
        done_chats += 1
        if on_progress is not None:
            on_progress(done_chats, total_chats)

    existing = _existing_imported_projects(spaces_root)
    mapped_uuids: set[str] = set()
    for proj in projects:
        orig = str(proj.get("original_id") or "")
        sid = existing.get(orig) if orig else None
        if sid is None or not (spaces_root / sid).is_dir():
            try:
                sid = create_space(spaces_root, proj["name"])
            except Exception as exc:  # pragma: no cover - fs guard
                result.errors.append(f"{proj['name']}: could not create project ({exc})")
                continue
            result.projects_created += 1
        else:
            result.projects_reused += 1
        space_root = spaces_root / sid
        _write_imported_space_meta(space_root, sid, proj)
        result.docs_written += _write_project_docs(space_root, proj["docs"])

        proj_chats = []
        for cuid, pid in conv_map.items():
            if pid == orig and cuid in chats_by_uuid:
                proj_chats.append(chats_by_uuid[cuid])
                mapped_uuids.add(cuid)
        if proj_chats:
            wr = write_chats_to_dir(
                proj_chats,
                notes_chats_dir(vault_root, sid),
                source=SOURCE_CLAUDE,
                batch_size=batch_size,
                batch_pause=batch_pause,
                on_chat=_tick,
            )
            result.chats_written += wr.chats_written
            result.messages_written += wr.messages_written
            result.skipped_duplicate += wr.skipped_duplicate
            result.errors.extend(wr.errors)

    # Conversations with no project linkage -> fallback (current) project.
    if fallback_chats_dir is not None:
        leftovers = [
            chat
            for cu, chat in chats_by_uuid.items()
            if cu not in mapped_uuids
        ]
        if leftovers:
            wr = write_chats_to_dir(
                leftovers,
                Path(fallback_chats_dir),
                source=SOURCE_CLAUDE,
                batch_size=batch_size,
                batch_pause=batch_pause,
                on_chat=_tick,
            )
            result.unmapped_chats += wr.chats_written
            result.messages_written += wr.messages_written
            result.skipped_duplicate += wr.skipped_duplicate
            result.errors.extend(wr.errors)

    return result


# --------------------------------------------------------------------------- #
# Inventory + cleanse: identify and remove ingested data (keep baseline only)
# --------------------------------------------------------------------------- #

@dataclass
class ImportedInventory:
    """Counts of ingested items across a notes vault, by source."""

    chats_by_source: dict[str, int] = field(default_factory=dict)
    projects_by_source: dict[str, int] = field(default_factory=dict)

    @property
    def total_chats(self) -> int:
        return sum(self.chats_by_source.values())

    @property
    def total_projects(self) -> int:
        return sum(self.projects_by_source.values())

    def is_empty(self) -> bool:
        return self.total_chats == 0 and self.total_projects == 0


def _chat_is_imported(path: Path) -> tuple[bool, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ""
    meta = data.get("metadata") if isinstance(data, dict) else None
    if isinstance(meta, dict) and bool(meta.get("imported")):
        return True, str(meta.get("import_source") or "")
    return False, ""


def _iter_space_chats_dirs(spaces_root: Path, vault_root: Path):
    """Yield ``(space_id, chats_dir)`` for every space in the vault."""
    from .notes_spaces import list_space_ids, notes_chats_dir

    for sid in list_space_ids(Path(spaces_root)):
        yield sid, notes_chats_dir(Path(vault_root), sid)


def scan_imported(spaces_root: Path, vault_root: Path) -> ImportedInventory:
    """Tally ingested chats and respawned projects across the whole vault."""
    inv = ImportedInventory()
    for _sid, chats_dir in _iter_space_chats_dirs(spaces_root, vault_root):
        if not chats_dir.is_dir():
            continue
        for chat_file in chats_dir.glob("chat_*.json"):
            imported, src = _chat_is_imported(chat_file)
            if imported:
                key = src or "unknown"
                inv.chats_by_source[key] = inv.chats_by_source.get(key, 0) + 1
    for orig_id, sid in _existing_imported_projects(Path(spaces_root)).items():
        meta = Path(spaces_root) / sid / "meta.json"
        src = SOURCE_CLAUDE
        try:
            raw = json.loads(meta.read_text(encoding="utf-8"))
            src = str(raw.get("import_source") or SOURCE_CLAUDE)
        except (OSError, json.JSONDecodeError):
            pass
        inv.projects_by_source[src] = inv.projects_by_source.get(src, 0) + 1
    return inv


def delete_imported_chats(
    spaces_root: Path, vault_root: Path, *, source: Optional[str] = None
) -> int:
    """Delete ingested chat files (optionally only one ``source``). Returns count.

    Native ("baseline") chats created on this system are never touched. When
    ``source`` is None, every ingested chat regardless of origin is removed.
    """
    want = (source or "").strip().lower()
    removed = 0
    for _sid, chats_dir in _iter_space_chats_dirs(spaces_root, vault_root):
        if not chats_dir.is_dir():
            continue
        for chat_file in chats_dir.glob("chat_*.json"):
            imported, src = _chat_is_imported(chat_file)
            if not imported:
                continue
            if want and src.strip().lower() != want:
                continue
            try:
                chat_file.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def delete_imported_projects(
    spaces_root: Path, vault_root: Path, *, source: Optional[str] = None
) -> int:
    """Delete respawned project spaces (dir + chats), keeping baseline projects.

    Destructive: removes the whole space directory (instructions, knowledge
    files) and its chat history. Returns the number of projects removed.
    """
    from .notes_spaces import notes_chats_dir

    want = (source or "").strip().lower()
    spaces_root = Path(spaces_root)
    removed = 0
    for orig_id, sid in _existing_imported_projects(spaces_root).items():
        space_dir = spaces_root / sid
        if want:
            try:
                raw = json.loads((space_dir / "meta.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                raw = {}
            if str(raw.get("import_source") or "").strip().lower() != want:
                continue
        chats_dir = notes_chats_dir(Path(vault_root), sid)
        try:
            shutil.rmtree(space_dir, ignore_errors=True)
            shutil.rmtree(chats_dir, ignore_errors=True)
            removed += 1
        except OSError:
            continue
    return removed
