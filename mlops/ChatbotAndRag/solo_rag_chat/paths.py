"""
Isolated on-disk layout for solo RAG + chat. Never uses techtronica prime paths.
"""

from pathlib import Path


def get_solo_data_root() -> Path:
    """Return the root directory for solo app data (under this package)."""
    return Path(__file__).resolve().parent / "_solo_data"


def ensure_solo_data_layout() -> Path:
    """Create solo data directories if missing."""
    root = get_solo_data_root()
    (root / "chats").mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_rag_index_dir() -> Path:
    """FAISS index directory for solo RAG."""
    return get_solo_data_root() / "rag_index"
