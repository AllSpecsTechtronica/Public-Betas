"""Notes vault layout: ``notes/spaces/<id>/`` for project content; ``notes/chats/<id>/`` for chat JSON (siblings)."""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NOTES_SPACES_MARKER = ".spaces_layout_v1"
DEFAULT_SPACE_ID = "main"
_LEGACY_VAULT_SUBDIRS = ("files", "recordings", "sessions", "captures")
_SPACE_EXTRA_SUBDIRS = ("rag_index",)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_space_tree(space_root: Path) -> None:
    for name in (*_LEGACY_VAULT_SUBDIRS, *_SPACE_EXTRA_SUBDIRS):
        (space_root / name).mkdir(parents=True, exist_ok=True)


def notes_chats_dir(notes_vault: Path, space_id: str) -> Path:
    """Per-project chat storage: ``notes/chats/<space_id>/`` (sibling of ``notes/spaces/``)."""
    return (notes_vault.resolve() / "chats" / space_id)


def _migrate_inline_chats_to_sibling(space_root: Path, target_chats: Path) -> None:
    """Move legacy ``spaces/<id>/chats/`` into ``notes/chats/<id>/`` when present."""
    inline = space_root / "chats"
    if not inline.is_dir():
        return
    target_chats.mkdir(parents=True, exist_ok=True)
    for item in list(inline.iterdir()):
        dest = target_chats / item.name
        try:
            if dest.exists():
                continue
            shutil.move(str(item), str(dest))
        except Exception:
            continue
    try:
        if inline.is_dir() and not any(inline.iterdir()):
            inline.rmdir()
    except Exception:
        pass


def _ensure_notes_chats_layout(notes_vault: Path, spaces_root: Path) -> None:
    chats_root = notes_vault / "chats"
    chats_root.mkdir(parents=True, exist_ok=True)
    for sid in list_space_ids(spaces_root):
        dest = chats_root / sid
        dest.mkdir(parents=True, exist_ok=True)
        _migrate_inline_chats_to_sibling(spaces_root / sid, dest)


def _write_meta(space_root: Path, space_id: str, title: str, goals: str = "") -> None:
    meta = space_root / "meta.json"
    if meta.exists():
        return
    payload: dict[str, Any] = {
        "id": space_id,
        "title": title,
        "goals": (goals or "").strip(),
        "created_at": _utc_now_iso(),
    }
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_space_ids(spaces_root: Path) -> list[str]:
    spaces_root = spaces_root.resolve()
    if not spaces_root.is_dir():
        return []
    out: list[str] = []
    for child in sorted(spaces_root.iterdir()):
        if child.is_dir() and not child.name.startswith("."):
            out.append(child.name)
    return out


def slugify_space_id(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", (raw or "").strip()).strip("_").lower()
    s = s[:48] if s else ""
    if not s:
        s = f"space_{int(datetime.now(timezone.utc).timestamp())}"
    return s


def ensure_unique_space_id(spaces_root: Path, base_id: str) -> str:
    candidate = base_id
    n = 2
    while (spaces_root / candidate).exists():
        candidate = f"{base_id}_{n}"
        n += 1
    return candidate


def read_space_title(space_root: Path, fallback_id: str) -> str:
    meta = space_root / "meta.json"
    try:
        if meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            t = str(data.get("title") or "").strip()
            if t:
                return t
    except Exception:
        pass
    return fallback_id


def read_space_goals(space_root: Path) -> str:
    meta = space_root / "meta.json"
    try:
        if meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            return str(data.get("goals") or "").strip()
    except Exception:
        pass
    return ""


def read_space_import_source(space_root: Path) -> str:
    """``import_source`` (e.g. ``claude``) for a respawned/imported project, else ""."""
    meta = space_root / "meta.json"
    try:
        if meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            if bool(data.get("imported")):
                return str(data.get("import_source") or "").strip()
    except Exception:
        pass
    return ""


def read_space_pinned(space_root: Path) -> bool:
    meta = space_root / "meta.json"
    try:
        if meta.is_file():
            data = json.loads(meta.read_text(encoding="utf-8"))
            return bool(data.get("pinned", False))
    except Exception:
        pass
    return False


def _read_space_meta_dict(space_root: Path) -> dict[str, Any]:
    meta = space_root / "meta.json"
    if not meta.is_file():
        return {}
    try:
        raw = json.loads(meta.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def set_space_pinned(space_root: Path, pinned: bool) -> None:
    meta = space_root / "meta.json"
    data = _read_space_meta_dict(space_root)
    if not data:
        data = {
            "id": space_root.name,
            "title": space_root.name,
            "goals": "",
            "created_at": _utc_now_iso(),
        }
    data["pinned"] = bool(pinned)
    meta.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_space_meta_title(space_root: Path, new_title: str) -> bool:
    meta = space_root / "meta.json"
    data = _read_space_meta_dict(space_root)
    if not data:
        return False
    t = (new_title or "").strip()
    if not t:
        return False
    data["title"] = t
    meta.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True


def create_space(spaces_root: Path, title: str, goals: str = "") -> str:
    """Create a new space directory. Returns the space id (directory name)."""
    spaces_root = spaces_root.resolve()
    spaces_root.mkdir(parents=True, exist_ok=True)
    base = slugify_space_id(title)
    space_id = ensure_unique_space_id(spaces_root, base)
    root = spaces_root / space_id
    root.mkdir(parents=True, exist_ok=False)
    _ensure_space_tree(root)
    _write_meta(root, space_id, title.strip() or space_id, goals)
    notes_vault = spaces_root.parent
    (notes_vault / "chats" / space_id).mkdir(parents=True, exist_ok=True)
    return space_id


def ensure_notes_spaces_layout(notes_vault: Path) -> Path:
    """Create ``notes/spaces`` and ``notes/chats``, migrate legacy flat dirs once, return ``spaces`` path."""
    notes_vault = notes_vault.resolve()
    notes_vault.mkdir(parents=True, exist_ok=True)
    spaces = notes_vault / "spaces"
    marker = notes_vault / NOTES_SPACES_MARKER

    if not marker.exists():
        spaces.mkdir(parents=True, exist_ok=True)
        main = spaces / DEFAULT_SPACE_ID
        main.mkdir(parents=True, exist_ok=True)
        for name in _LEGACY_VAULT_SUBDIRS:
            src = notes_vault / name
            dst = main / name
            if not src.is_dir():
                continue
            try:
                if src.resolve() == dst.resolve():
                    continue
            except Exception:
                pass
            if dst.exists():
                try:
                    if any(dst.iterdir()):
                        continue
                    shutil.rmtree(dst)
                except Exception:
                    continue
            try:
                shutil.move(str(src), str(dst))
            except Exception:
                continue
        _ensure_space_tree(main)
        _write_meta(main, DEFAULT_SPACE_ID, "Main", "")
        try:
            marker.write_text("1\n", encoding="utf-8")
        except Exception:
            pass
    spaces.mkdir(parents=True, exist_ok=True)

    ids = list_space_ids(spaces)
    if not ids:
        main = spaces / DEFAULT_SPACE_ID
        main.mkdir(parents=True, exist_ok=True)
        _ensure_space_tree(main)
        _write_meta(main, DEFAULT_SPACE_ID, "Main", "")

    for sid in list_space_ids(spaces):
        _ensure_space_tree(spaces / sid)

    _ensure_notes_chats_layout(notes_vault, spaces)

    return spaces
