"""Turn ingested AI-export chats into durable RAG memory for Tacitus.

Importing chats from another AI (see :mod:`notes_ai_import`) only writes
``chat_*.json`` files. This module takes that influx of knowledge and organizes
it into a memory corpus the in-app assistant can actually retrieve:

* **Detected memory files** -- instructions / custom-instructions / project
  knowledge pulled straight out of the export.
* **Full transcripts** -- one markdown doc per ingested conversation.
* **Condensed summaries** -- a best-effort short summary per conversation
  (skipped silently when no local model is reachable).

All docs land under ``<vault>/ingested_memory/<source>/`` and a provenance
``ledger.json`` records every ingest batch. The dedicated RAG namespace
``ingested_memory`` is built *from that directory*, so deleting a subset of the
memory is just "remove the docs, then rebuild".

This module is intentionally Qt-free so it can be unit tested and reused; the
actual FAISS (re)build is driven by ``RAGWorker`` from the workspace.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .notes_ai_import import (
    SOURCE_CHATGPT,
    SOURCE_CLAUDE,
    load_export_members,
    parse_claude_projects,
)

# Registry namespace for the global ingested-memory RAG. Any string works with
# ``set_rag_config``/``get_rag_system``; this one is dedicated and cross-project.
INGESTED_MEMORY_NAMESPACE = "ingested_memory"

# Directory names under the notes vault.
INGESTED_MEMORY_DIRNAME = "ingested_memory"
INGESTED_MEMORY_INDEX_DIRNAME = "ingested_memory_rag_index"
LEDGER_FILENAME = "ledger.json"

_SOURCE_LABELS = {SOURCE_CHATGPT: "ChatGPT", SOURCE_CLAUDE: "Claude"}


def _source_label(source: object) -> str:
    s = str(source or "").strip().lower()
    return _SOURCE_LABELS.get(s, s or "another AI")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def ingested_memory_root(vault_root: Path) -> Path:
    """The managed memory corpus dir: ``<vault>/ingested_memory``."""
    return Path(vault_root) / INGESTED_MEMORY_DIRNAME


def ingested_memory_index_dir(vault_root: Path) -> Path:
    """FAISS index dir for the ``ingested_memory`` namespace."""
    return Path(vault_root) / INGESTED_MEMORY_INDEX_DIRNAME


def _source_dir(vault_root: Path, source: str) -> Path:
    return ingested_memory_root(vault_root) / (str(source or "unknown").strip().lower() or "unknown")


def _ledger_path(vault_root: Path) -> Path:
    return ingested_memory_root(vault_root) / LEDGER_FILENAME


def memory_doc_paths(vault_root: Path) -> list[str]:
    """Absolute paths of every memory doc remaining on disk (for rebuilds)."""
    root = ingested_memory_root(vault_root)
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.rglob("*.md") if p.is_file())


def memory_doc_count(vault_root: Path, *, source: Optional[str] = None) -> int:
    """Number of memory docs on disk, optionally scoped to one ``source``."""
    if source:
        sdir = _source_dir(vault_root, source)
        if not sdir.is_dir():
            return 0
        return sum(1 for p in sdir.rglob("*.md") if p.is_file())
    return len(memory_doc_paths(vault_root))


# --------------------------------------------------------------------------- #
# Reading the imported chats back off disk
# --------------------------------------------------------------------------- #

def iter_imported_chats(
    spaces_root: Path, vault_root: Path, *, source: Optional[str] = None
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(chat_id, chat_dict)`` for ingested chats, optionally one source."""
    from .notes_ai_import import _chat_is_imported, _iter_space_chats_dirs

    want = (source or "").strip().lower()
    for _sid, chats_dir in _iter_space_chats_dirs(Path(spaces_root), Path(vault_root)):
        if not Path(chats_dir).is_dir():
            continue
        for chat_file in sorted(Path(chats_dir).glob("chat_*.json")):
            imported, src = _chat_is_imported(chat_file)
            if not imported:
                continue
            if want and src.strip().lower() != want:
                continue
            try:
                data = json.loads(chat_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict):
                yield (str(data.get("id") or chat_file.stem), data)


# --------------------------------------------------------------------------- #
# Memory-file detection (the "check for memory files" step)
# --------------------------------------------------------------------------- #

def _chatgpt_memory_text(user_payload: Any) -> str:
    """Pull custom-instructions / memory text out of a ChatGPT ``user.json``."""
    if not isinstance(user_payload, dict):
        return ""
    keys = (
        "custom_instructions",
        "about_user_message",
        "about_model_message",
        "memory",
        "memories",
        "system_message",
    )
    parts: list[str] = []
    for key in keys:
        val = user_payload.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"## {key}\n{val.strip()}")
        elif isinstance(val, (list, dict)) and val:
            parts.append(f"## {key}\n{json.dumps(val, indent=2, ensure_ascii=False)}")
    return "\n\n".join(parts).strip()


def detect_memory_files(path: Path) -> list[dict[str, str]]:
    """Best-effort extraction of memory/instruction docs from an export.

    Returns ``[{"name": ..., "content": ...}]``. Never raises -- a missing or
    unrecognized export simply yields an empty list.
    """
    out: list[dict[str, str]] = []
    try:
        members = load_export_members(Path(path), ["user.json", "projects.json"])
    except Exception:
        members = {}

    text = _chatgpt_memory_text(members.get("user.json"))
    if text:
        out.append({"name": "memory_chatgpt_instructions.md", "content": text})

    projects = members.get("projects.json")
    if projects is not None:
        try:
            parsed = parse_claude_projects(projects)
        except Exception:
            parsed = []
        for idx, proj in enumerate(parsed):
            title = str(proj.get("title") or proj.get("name") or f"project_{idx}").strip()
            instructions = str(proj.get("instructions") or proj.get("goals") or "").strip()
            docs = proj.get("docs") or proj.get("knowledge") or []
            chunks: list[str] = [f"# Project: {title}"]
            if instructions:
                chunks.append(f"## Instructions\n{instructions}")
            if isinstance(docs, list):
                for d in docs:
                    if not isinstance(d, dict):
                        continue
                    dname = str(d.get("filename") or d.get("name") or d.get("title") or "knowledge").strip()
                    dcontent = str(d.get("content") or d.get("text") or "").strip()
                    if dcontent:
                        chunks.append(f"## Knowledge: {dname}\n{dcontent}")
            body = "\n\n".join(chunks).strip()
            if body and (instructions or len(chunks) > 1):
                safe = "".join(c if c.isalnum() else "_" for c in title.lower())[:40] or f"project_{idx}"
                out.append({"name": f"memory_claude_project_{safe}.md", "content": body})
    return out


# --------------------------------------------------------------------------- #
# Document rendering
# --------------------------------------------------------------------------- #

def render_transcript_doc(chat: dict[str, Any]) -> str:
    """Render a full chat transcript as a markdown memory doc."""
    title = str(chat.get("title") or "Untitled conversation").strip()
    source = _source_label((chat.get("metadata") or {}).get("import_source"))
    created = str(chat.get("created_at") or "").strip()
    lines = [f"# {title}", "", f"_Imported from {source}{(' — ' + created) if created else ''}._", ""]
    for msg in chat.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        speaker = "User" if role in ("user", "human") else ("Assistant" if role in ("assistant", "ai", "model") else role or "?")
        lines.append(f"**{speaker}:** {content}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _plain_transcript_text(chat: dict[str, Any]) -> str:
    bits: list[str] = []
    for msg in chat.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if content:
            speaker = "User" if role in ("user", "human") else "Assistant"
            bits.append(f"{speaker}: {content}")
    return "\n".join(bits)


def summarize_text(
    text: str, *, model: Optional[str], base_url: Optional[str], timeout: float = 60.0
) -> str:
    """Best-effort condensed summary via a local Ollama model. "" on any failure."""
    body = (text or "").strip()
    if not body or not model or not base_url:
        return ""
    try:
        import httpx

        prompt = (
            "Summarize the following conversation into durable, factual memory "
            "notes a future assistant could reuse. Keep names, decisions, and "
            "preferences. Be concise (under 200 words).\n\n" + body[:12000]
        )
        url = base_url.rstrip("/") + "/api/generate"
        resp = httpx.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return str(data.get("response") or "").strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Ledger (provenance)
# --------------------------------------------------------------------------- #

def read_ledger(vault_root: Path) -> dict[str, Any]:
    path = _ledger_path(vault_root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("batches"), list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"batches": []}


def _write_ledger(vault_root: Path, ledger: dict[str, Any]) -> None:
    ingested_memory_root(vault_root).mkdir(parents=True, exist_ok=True)
    _ledger_path(vault_root).write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def append_batch(vault_root: Path, batch: dict[str, Any]) -> None:
    ledger = read_ledger(vault_root)
    ledger["batches"].append(batch)
    _write_ledger(vault_root, ledger)


def _ledgered_chat_ids(vault_root: Path) -> set[str]:
    """Every chat id already recorded in a ledger batch.

    Used so a reindex (which regenerates docs for chats that were ingested long
    ago) does not append duplicate provenance batches — only genuinely new chats
    create a batch.
    """
    out: set[str] = set()
    for batch in read_ledger(vault_root).get("batches", []):
        if isinstance(batch, dict):
            for cid in batch.get("chat_ids") or []:
                out.add(str(cid))
    return out


def mark_chats_deleted(vault_root: Path, *, source: Optional[str] = None) -> int:
    """Flag batches whose source chats were deleted (memory kept). Returns count."""
    want = (source or "").strip().lower()
    ledger = read_ledger(vault_root)
    n = 0
    for batch in ledger.get("batches", []):
        if not isinstance(batch, dict):
            continue
        if want and str(batch.get("source") or "").strip().lower() != want:
            continue
        if not batch.get("chats_deleted"):
            batch["chats_deleted"] = True
            n += 1
    if n:
        _write_ledger(vault_root, ledger)
    return n


def ledger_summary_text(vault_root: Path) -> str:
    """Short human/Tacitus-facing description of what knowledge was ingested."""
    ledger = read_ledger(vault_root)
    batches = [b for b in ledger.get("batches", []) if isinstance(b, dict)]
    if not batches:
        return ""
    chats_by_source: dict[str, int] = {}
    any_deleted = False
    for b in batches:
        src = str(b.get("source") or "unknown").strip().lower() or "unknown"
        counts = b.get("counts") if isinstance(b.get("counts"), dict) else {}
        chats_by_source[src] = chats_by_source.get(src, 0) + int(counts.get("chats") or 0)
        if b.get("chats_deleted"):
            any_deleted = True
    parts = [f"{n} {_source_label(src)} conversation(s)" for src, n in sorted(chats_by_source.items()) if n]
    if not parts:
        return ""
    tail = f" ({len(batches)} import batch(es)"
    tail += "; some source chats deleted but their memory is retained)" if any_deleted else ")"
    return "Transferred knowledge in memory: " + ", ".join(parts) + tail


# --------------------------------------------------------------------------- #
# Build / delete
# --------------------------------------------------------------------------- #

def build_memory_docs(
    *,
    vault_root: Path,
    spaces_root: Path,
    source: str,
    export_path: Optional[Path] = None,
    summarize: bool = False,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    force: bool = False,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, Any]:
    """Generate memory docs for ingested chats and record ledger batches.

    ``source`` selects one origin (``"chatgpt"``/``"claude"``); ``""``, ``None``
    or ``"all"`` reindex every ingested chat regardless of origin, writing each
    into its own source directory.

    Idempotent per chat by default: a chat that already has a transcript doc is
    skipped, so re-running after a partial import only adds the new conversations.
    Pass ``force=True`` to *reindex* — regenerate transcripts/summaries for chats
    that are already on disk (e.g. they were ingested before this corpus existed,
    or you changed the summarizer). ``on_progress(done, total)`` reports cumulative
    conversation progress so the UI can show a batched ETA.

    Returns a summary dict including ``doc_paths`` (newly written, absolute).
    """
    want = str(source or "").strip().lower()
    all_sources = want in ("", "all")

    # Materialize the chat list up front so ``total`` is known for progress/ETA.
    chats = list(
        iter_imported_chats(
            spaces_root, vault_root, source=None if all_sources else want
        )
    )
    total = len(chats)
    if on_progress is not None:
        on_progress(0, total)

    already_ledgered = _ledgered_chat_ids(vault_root)
    new_docs: list[str] = []
    n_transcripts = n_summaries = 0
    # Per effective source: chats seen, brand-new chats (for the ledger), docs.
    per_source: dict[str, dict[str, list[str]]] = {}

    done = 0
    for chat_id, chat in chats:
        csrc = str((chat.get("metadata") or {}).get("import_source") or "").strip().lower()
        if not csrc:
            csrc = "unknown" if all_sources else (want or "unknown")
        sdir = _source_dir(vault_root, csrc)
        sdir.mkdir(parents=True, exist_ok=True)
        acc = per_source.setdefault(csrc, {"chat_ids": [], "new_chat_ids": [], "docs": []})
        acc["chat_ids"].append(chat_id)
        if chat_id not in already_ledgered:
            acc["new_chat_ids"].append(chat_id)

        transcript_path = sdir / f"transcript_{chat_id}.md"
        if force or not transcript_path.exists():
            try:
                transcript_path.write_text(render_transcript_doc(chat), encoding="utf-8")
            except OSError:
                done += 1
                if on_progress is not None:
                    on_progress(done, total)
                continue
            new_docs.append(str(transcript_path))
            acc["docs"].append(str(transcript_path))
            n_transcripts += 1

            if summarize:
                summary = summarize_text(
                    _plain_transcript_text(chat), model=model, base_url=base_url
                )
                if summary:
                    summary_path = sdir / f"summary_{chat_id}.md"
                    title = str(chat.get("title") or "Conversation").strip()
                    try:
                        summary_path.write_text(
                            f"# Summary: {title}\n\n{summary}\n", encoding="utf-8"
                        )
                        new_docs.append(str(summary_path))
                        acc["docs"].append(str(summary_path))
                        n_summaries += 1
                    except OSError:
                        pass
        done += 1
        if on_progress is not None:
            on_progress(done, total)

    # Detected memory / instruction files from the export itself (only available
    # on a fresh import of a single source, not a reindex from disk).
    n_memory_files = 0
    if export_path is not None and not all_sources:
        msrc = want or "unknown"
        sdir = _source_dir(vault_root, msrc)
        sdir.mkdir(parents=True, exist_ok=True)
        acc = per_source.setdefault(msrc, {"chat_ids": [], "new_chat_ids": [], "docs": []})
        for mem in detect_memory_files(Path(export_path)):
            mem_path = sdir / mem["name"]
            try:
                mem_path.write_text(mem["content"], encoding="utf-8")
            except OSError:
                continue
            if str(mem_path) not in new_docs:
                new_docs.append(str(mem_path))
                acc["docs"].append(str(mem_path))
            n_memory_files += 1

    # One provenance batch per source — but only for genuinely new chats so a
    # reindex of already-recorded chats doesn't bloat the ledger.
    now_iso = datetime.now().isoformat(timespec="seconds")
    for csrc, acc in per_source.items():
        if not acc["new_chat_ids"] and not acc["docs"]:
            continue
        if not acc["new_chat_ids"]:
            # Reindex only (no new chats): docs were rewritten but provenance
            # already exists, so skip appending a duplicate batch.
            continue
        append_batch(
            vault_root,
            {
                "id": f"batch_{int(datetime.now().timestamp() * 1000)}_{csrc}",
                "source": csrc,
                "created_at": now_iso,
                "chat_ids": acc["new_chat_ids"],
                "doc_paths": [
                    str(Path(p).relative_to(Path(vault_root))) for p in acc["docs"]
                ],
                "counts": {"chats": len(acc["new_chat_ids"])},
                "chats_deleted": False,
            },
        )

    return {
        "source": "all" if all_sources else (want or "unknown"),
        "doc_paths": new_docs,
        "transcripts": n_transcripts,
        "summaries": n_summaries,
        "memory_files": n_memory_files,
        "chats": total,
    }


def delete_memory_docs(vault_root: Path, *, source: Optional[str] = None) -> int:
    """Remove memory docs (one source or all) and their ledger batches.

    Returns the number of doc files removed. The FAISS namespace must be rebuilt
    afterwards from :func:`memory_doc_paths` (or cleared when none remain).
    """
    want = (source or "").strip().lower()
    root = ingested_memory_root(vault_root)
    removed = 0

    if want:
        sdir = _source_dir(vault_root, want)
        if sdir.is_dir():
            removed = sum(1 for _ in sdir.rglob("*.md"))
            shutil.rmtree(sdir, ignore_errors=True)
    else:
        if root.is_dir():
            for sub in root.iterdir():
                if sub.is_dir():
                    removed += sum(1 for _ in sub.rglob("*.md"))
                    shutil.rmtree(sub, ignore_errors=True)

    # Prune ledger batches for the affected source(s).
    ledger = read_ledger(vault_root)
    if want:
        ledger["batches"] = [
            b for b in ledger.get("batches", [])
            if isinstance(b, dict) and str(b.get("source") or "").strip().lower() != want
        ]
    else:
        ledger["batches"] = []
    _write_ledger(vault_root, ledger)
    return removed
