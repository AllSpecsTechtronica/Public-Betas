"""Discover Ollama tag names (HTTP ``/api/tags``, ``ollama list`` CLI) and local .gguf paths for CV Ops."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

_SKIP_DIR_NAMES = frozenset(
    {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".tox",
        "site-packages",
        "blobs",
    }
)


def ollama_api_base_url() -> str:
    """HTTP origin for the Ollama API (e.g. http://127.0.0.1:11434), suitable for GET /api/tags."""
    host = os.environ.get("OLLAMA_HOST", "").strip()
    if host:
        if "://" in host:
            return host.rstrip("/")
        return f"http://{host}".rstrip("/")
    try:
        from insight_local.config import INSIGHT_OLLAMA_URL

        parsed = urlparse(INSIGHT_OLLAMA_URL)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:11434"


def fetch_ollama_model_names(*, base_url: str | None = None, timeout: float = 2.0) -> list[str]:
    base = (base_url or ollama_api_base_url()).rstrip("/")
    url = f"{base}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return []
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return []
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []
    names: list[str] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if name:
            names.append(name)
    return sorted(set(names), key=str.lower)


def ollama_host_env_value_from_api_origin(base_url: str) -> str | None:
    """Value for ``OLLAMA_HOST`` when shelling out to the ``ollama`` CLI (``host:port``)."""
    raw = str(base_url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port
    if port is not None:
        return f"{host}:{port}"
    return host


def parse_ollama_list_stdout(text: str) -> list[str]:
    """Parse ``ollama list`` table output; first column is the model tag."""
    names: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.upper().startswith("NAME"):
            continue
        parts = [p for p in re.split(r"\s{2,}", line) if p]
        if not parts:
            continue
        name = parts[0].strip()
        if not name or name.upper() == "NAME":
            continue
        names.append(name)
    return sorted(set(names), key=str.lower)


def fetch_ollama_model_names_via_cli(
    *,
    base_url: str | None = None,
    timeout: float = 25.0,
) -> list[str]:
    """Run ``ollama list`` locally (respects ``OLLAMA_HOST`` / URL-derived host for remote daemons)."""
    exe = shutil.which("ollama")
    if not exe:
        return []
    env = os.environ.copy()
    host_val = ollama_host_env_value_from_api_origin(str(base_url or "").strip())
    if host_val:
        env["OLLAMA_HOST"] = host_val
    try:
        proc = subprocess.run(
            [exe, "list"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return parse_ollama_list_stdout(proc.stdout or "")


def discover_ollama_model_tags(
    *,
    base_url: str | None = None,
    timeout_http: float = 3.0,
    timeout_cli: float = 25.0,
) -> list[str]:
    """Union of tags from HTTP ``GET /api/tags`` and the ``ollama list`` CLI (installed models)."""
    http_tags = fetch_ollama_model_names(base_url=base_url, timeout=timeout_http)
    cli_tags = fetch_ollama_model_names_via_cli(base_url=base_url, timeout=timeout_cli)
    merged = {t.strip() for t in (*http_tags, *cli_tags) if str(t).strip()}
    return sorted(merged, key=str.lower)


_EMBEDDING_NAME_HINTS = (
    "embed",
    "embedding",
    "nomic",
    "mxbai",
    "bge",
    "e5",
    "gte",
    "jina",
    "minilm",
    "snowflake",
    "arctic",
)

_EMBEDDING_MODEL_PRIORITY = (
    "nomic-embed-text",
    "mxbai-embed-large",
    "bge-m3",
    "bge-large",
    "snowflake-arctic-embed",
    "all-minilm",
    "embeddinggemma",
)


def _ollama_tag_key(tag: str) -> str:
    value = str(tag or "").strip().lower()
    return value[:-7] if value.endswith(":latest") else value


def looks_like_ollama_embedding_model(tag: str) -> bool:
    """Best-effort name check for locally installed Ollama embedding models."""
    value = _ollama_tag_key(tag)
    return bool(value and any(hint in value for hint in _EMBEDDING_NAME_HINTS))


def choose_ollama_embedding_model(
    tags: list[str] | tuple[str, ...],
    *,
    current: str = "",
    fallback: str = "nomic-embed-text",
) -> str:
    """Pick an installed Ollama tag for embeddings, preferring known embedding families."""
    installed = [str(t).strip() for t in tags if str(t).strip()]
    if not installed:
        return str(current or fallback).strip()

    current_key = _ollama_tag_key(current)
    if current_key:
        for tag in installed:
            if _ollama_tag_key(tag) == current_key:
                return tag

    keyed = [(_ollama_tag_key(tag), tag) for tag in installed]
    for preferred in _EMBEDDING_MODEL_PRIORITY:
        pref = _ollama_tag_key(preferred)
        for key, tag in keyed:
            if key == pref or key.startswith(f"{pref}:") or pref in key:
                return tag

    for tag in installed:
        if looks_like_ollama_embedding_model(tag):
            return tag
    return installed[0]


def _dir_blocked(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    if "site-packages" in parts:
        return True
    return False


def _walk_ggufs(root: Path, *, max_depth: int, budget: list[int], found: dict[str, None]) -> None:
    if budget[0] <= 0:
        return
    try:
        root = root.expanduser().resolve()
    except (OSError, ValueError):
        return
    if not root.is_dir() or _dir_blocked(root):
        return
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and budget[0] > 0:
        path, depth = stack.pop()
        if _dir_blocked(path):
            continue
        try:
            children = list(path.iterdir())
        except (OSError, PermissionError):
            continue
        for child in children:
            if budget[0] <= 0:
                return
            name = child.name
            if child.is_dir():
                if name in _SKIP_DIR_NAMES or name.startswith("."):
                    continue
                if depth < max_depth:
                    stack.append((child, depth + 1))
                continue
            if child.is_file() and child.suffix.lower() == ".gguf":
                try:
                    key = str(child.resolve())
                except (OSError, ValueError):
                    key = str(child)
                if key not in found:
                    found[key] = None
                    budget[0] -= 1


def _volume_scan_roots() -> list[Path]:
    if os.environ.get("INSIGHT_GGUF_SCAN_VOLUMES", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return []
    vol = Path("/Volumes")
    if not vol.is_dir():
        return []
    out: list[Path] = []
    try:
        for child in vol.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                out.append(child)
    except (OSError, PermissionError):
        return []
    return out


def discover_local_gguf_files(
    *,
    repo_root: Path | None = None,
    max_files: int = 500,
    per_root_depth: int = 8,
) -> list[str]:
    """Find .gguf files under common locations and optional INSIGHT_GGUF_SCAN_ROOTS."""
    roots: list[Path] = []
    for raw in os.environ.get("INSIGHT_GGUF_SCAN_ROOTS", "").split(os.pathsep):
        part = raw.strip()
        if part:
            roots.append(Path(part))
    ollama_models = os.environ.get("OLLAMA_MODELS", "").strip()
    if ollama_models:
        roots.append(Path(ollama_models))
    home = Path.home()
    roots.extend(
        [
            home / "models",
            home / "Downloads",
        ]
    )
    if os.environ.get("INSIGHT_GGUF_SCAN_DOT_OLLAMA", "").strip().lower() in {"1", "true", "yes", "on"}:
        roots.append(home / ".ollama")
    roots.extend(_volume_scan_roots())
    if repo_root is not None:
        roots.insert(0, Path(repo_root))
    seen: set[str] = set()
    uniq: list[Path] = []
    for r in roots:
        try:
            rp = r.expanduser().resolve()
        except (OSError, ValueError):
            continue
        key = str(rp)
        if key in seen or not rp.is_dir():
            continue
        seen.add(key)
        uniq.append(rp)
    found: dict[str, None] = {}
    budget = [max(1, int(max_files))]
    for r in uniq:
        _walk_ggufs(r, max_depth=max(1, int(per_root_depth)), budget=budget, found=found)
        if budget[0] <= 0:
            break
    return sorted(found.keys(), key=str.lower)


def list_finetune_base_candidates(
    *,
    repo_root: Path | None = None,
    ollama_timeout: float = 2.0,
    gguf_max_files: int = 500,
) -> tuple[list[str], list[str]]:
    """Return (ollama_model_names, absolute_gguf_paths)."""
    ollama_names = fetch_ollama_model_names(timeout=ollama_timeout)
    ggufs = discover_local_gguf_files(repo_root=repo_root, max_files=gguf_max_files)
    return ollama_names, ggufs
