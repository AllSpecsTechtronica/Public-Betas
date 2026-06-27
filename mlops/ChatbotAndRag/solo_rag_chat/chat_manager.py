"""
Solo chat management: sessions and Ollama embedding recall.
Storage lives under solo_rag_chat/_solo_data only (not techtronica prime).
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .paths import get_solo_data_root, ensure_solo_data_layout

try:
    import httpx

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

_embedding_cache: Dict[str, List[float]] = {}
_response_cache: Dict[str, Dict[str, Any]] = {}

_SOLO_ROOT = ensure_solo_data_layout()
_DEFAULT_CHATS_DIR = _SOLO_ROOT / "chats"
_DEFAULT_CHATS_DIR.mkdir(parents=True, exist_ok=True)
# Backwards compatibility for imports of CHATS_DIR (solo default).
CHATS_DIR = _DEFAULT_CHATS_DIR

EMBEDDING_CACHE_FILE = _SOLO_ROOT / "embedding_cache.json"
RESPONSE_CACHE_FILE = _SOLO_ROOT / "response_cache.json"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "ibm/granite-embedding:278m-multilingual-q4_K_M"

_resolved_embedding_model: Optional[str] = None


def load_embedding_cache() -> None:
    global _embedding_cache
    try:
        if EMBEDDING_CACHE_FILE.exists():
            with open(EMBEDDING_CACHE_FILE, "r", encoding="utf-8") as f:
                _embedding_cache = json.load(f)
            print(f"[SOLO][CACHE] Loaded {len(_embedding_cache)} cached embeddings")
    except Exception as e:
        print(f"[SOLO][CACHE] Error loading embedding cache: {e}")
        _embedding_cache = {}


def save_embedding_cache() -> None:
    try:
        with open(EMBEDDING_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_embedding_cache, f, indent=2)
    except Exception as e:
        print(f"[SOLO][CACHE] Error saving embedding cache: {e}")


def load_response_cache() -> None:
    global _response_cache
    try:
        if RESPONSE_CACHE_FILE.exists():
            with open(RESPONSE_CACHE_FILE, "r", encoding="utf-8") as f:
                _response_cache = json.load(f)
            print(f"[SOLO][CACHE] Loaded {len(_response_cache)} cached responses")
    except Exception as e:
        print(f"[SOLO][CACHE] Error loading response cache: {e}")
        _response_cache = {}


def save_response_cache() -> None:
    try:
        with open(RESPONSE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_response_cache, f, indent=2)
    except Exception as e:
        print(f"[SOLO][CACHE] Error saving response cache: {e}")


def get_cache_key(prompt: str, model: str, system_prompt: str = "") -> str:
    content = f"{model}:{system_prompt}:{prompt}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


async def _discover_embedding_model(client: "httpx.AsyncClient") -> Optional[str]:
    try:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if response.status_code != 200:
            return None
        models = [m.get("name", "") for m in response.json().get("models", [])]
        if EMBEDDING_MODEL in models:
            return EMBEDDING_MODEL
        embed_keywords = ("embed", "nomic", "mxbai", "bge", "e5")
        for name in models:
            if any(kw in name.lower() for kw in embed_keywords):
                return name
        return models[0] if models else None
    except Exception:
        return None


async def get_embedding(text: str, use_cache: bool = True) -> Optional[List[float]]:
    global _resolved_embedding_model

    cache_key = hashlib.md5(text.encode("utf-8")).hexdigest()
    if use_cache and cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    if not EMBEDDING_AVAILABLE:
        return None

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            if _resolved_embedding_model is None:
                _resolved_embedding_model = await _discover_embedding_model(client)
            if _resolved_embedding_model is None:
                return None

            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": _resolved_embedding_model, "prompt": text},
            )
            if response.status_code == 200:
                data = response.json()
                embedding = data.get("embedding", [])
                if embedding:
                    _embedding_cache[cache_key] = embedding
                    return embedding
            elif response.status_code in (404, 400):
                print(
                    f"[SOLO][EMBEDDING] Model '{_resolved_embedding_model}' not found, will re-discover"
                )
                _resolved_embedding_model = None
    except Exception as e:
        print(f"[SOLO][EMBEDDING] Error getting embedding: {e}")

    return None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


class ChatManager:
    """Manages chat sessions with embedding-based recall.

    When ``chats_dir`` is omitted, storage uses ``solo_rag_chat/_solo_data/chats``
    (standalone app). When set (e.g. CV Ops notes space), JSON files live only
    under that directory.
    """

    def __init__(self, chats_dir: Optional[Path] = None) -> None:
        if chats_dir is None:
            ensure_solo_data_layout()
            self._chats_dir = _DEFAULT_CHATS_DIR
        else:
            self._chats_dir = Path(chats_dir).expanduser().resolve()
            self._chats_dir.mkdir(parents=True, exist_ok=True)
        self.chats: Dict[str, Dict[str, Any]] = {}
        self.load_all_chats()
        load_embedding_cache()
        load_response_cache()

    def get_chat_file(self, chat_id: str) -> Path:
        return self._chats_dir / f"{chat_id}.json"

    def create_chat(
        self, title: Optional[str] = None, description: Optional[str] = None
    ) -> str:
        chat_id = f"chat_{int(time.time() * 1000)}"
        chat = {
            "id": chat_id,
            "title": title or "New Chat",
            "description": description or "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "metadata": {},
        }
        self.chats[chat_id] = chat
        self.save_chat(chat_id)
        return chat_id

    def update_chat_metadata(
        self,
        chat_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        save_to_system_memory: Optional[bool] = None,
    ) -> bool:
        if chat_id not in self.chats:
            return False
        if title is not None:
            self.chats[chat_id]["title"] = title
        if description is not None:
            self.chats[chat_id]["description"] = description
        if messages is not None:
            self.chats[chat_id]["messages"] = messages
        if save_to_system_memory is not None:
            self.chats[chat_id].setdefault("metadata", {})[
                "save_to_system_memory"
            ] = save_to_system_memory
        self.chats[chat_id]["updated_at"] = datetime.now().isoformat()
        self.save_chat(chat_id)
        return True

    def set_chat_pinned(self, chat_id: str, pinned: bool) -> bool:
        if chat_id not in self.chats:
            if not self.load_chat(chat_id):
                return False
        self.chats[chat_id].setdefault("metadata", {})["pinned"] = bool(pinned)
        self.chats[chat_id]["updated_at"] = datetime.now().isoformat()
        self.save_chat(chat_id)
        return True

    def is_chat_pinned(self, chat_id: str) -> bool:
        if chat_id not in self.chats:
            if not self.load_chat(chat_id):
                return False
        return bool(self.chats[chat_id].get("metadata", {}).get("pinned", False))

    def save_chat(self, chat_id: str) -> None:
        if chat_id not in self.chats:
            return
        try:
            chat_file = self.get_chat_file(chat_id)
            self.chats[chat_id]["updated_at"] = datetime.now().isoformat()
            with open(chat_file, "w", encoding="utf-8") as f:
                json.dump(self.chats[chat_id], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[SOLO][CHAT] Error saving chat {chat_id}: {e}")

    def load_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        try:
            chat_file = self.get_chat_file(chat_id)
            if chat_file.exists():
                with open(chat_file, "r", encoding="utf-8") as f:
                    chat = json.load(f)
                    self.chats[chat_id] = chat
                    return chat
        except Exception as e:
            print(f"[SOLO][CHAT] Error loading chat {chat_id}: {e}")
        return None

    def load_all_chats(self) -> None:
        self.chats = {}
        try:
            for chat_file in self._chats_dir.glob("chat_*.json"):
                try:
                    with open(chat_file, "r", encoding="utf-8") as f:
                        chat = json.load(f)
                        self.chats[chat.get("id", chat_file.stem)] = chat
                except Exception as e:
                    print(f"[SOLO][CHAT] Error loading {chat_file}: {e}")
            print(f"[SOLO][CHAT] Loaded {len(self.chats)} chats")
        except Exception as e:
            print(f"[SOLO][CHAT] Error loading chats: {e}")

    def list_chats(self) -> List[Dict[str, Any]]:
        chats_list = []
        for chat_id, chat in self.chats.items():
            chats_list.append(
                {
                    "id": chat_id,
                    "title": chat.get("title", "Untitled"),
                    "description": chat.get("description", ""),
                    "created_at": chat.get("created_at", ""),
                    "updated_at": chat.get("updated_at", ""),
                    "message_count": len(chat.get("messages", [])),
                    "save_to_system_memory": chat.get("metadata", {}).get(
                        "save_to_system_memory", False
                    ),
                    "pinned": bool(chat.get("metadata", {}).get("pinned", False)),
                    "imported": bool(chat.get("metadata", {}).get("imported", False)),
                    "import_source": str(
                        chat.get("metadata", {}).get("import_source", "")
                    ),
                }
            )
        pinned = [c for c in chats_list if c.get("pinned")]
        unpinned = [c for c in chats_list if not c.get("pinned")]
        pinned.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        unpinned.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return pinned + unpinned

    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if chat_id not in self.chats:
            if not self.load_chat(chat_id):
                chat_id = self.create_chat()
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self.chats[chat_id]["messages"].append(message)
        self.chats[chat_id]["updated_at"] = datetime.now().isoformat()
        self.save_chat(chat_id)

    def get_chat_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        if chat_id in self.chats:
            return self.chats[chat_id].get("messages", [])
        return []

    def delete_chat(self, chat_id: str) -> bool:
        try:
            if chat_id in self.chats:
                del self.chats[chat_id]
            chat_file = self.get_chat_file(chat_id)
            if chat_file.exists():
                chat_file.unlink()
            return True
        except Exception as e:
            print(f"[SOLO][CHAT] Error deleting chat {chat_id}: {e}")
            return False

    async def search_chats(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_embedding = await get_embedding(query)
        if not query_embedding:
            return self._text_search_chats(query, limit)
        results = []
        for chat_id, chat in self.chats.items():
            best_score = 0.0
            best_message = None
            for message in chat.get("messages", []):
                content = message.get("content", "")
                if not content:
                    continue
                msg_embedding = await get_embedding(content)
                if msg_embedding:
                    score = cosine_similarity(query_embedding, msg_embedding)
                    if score > best_score:
                        best_score = score
                        best_message = message
            if best_score > 0.3:
                results.append(
                    {
                        "chat_id": chat_id,
                        "title": chat.get("title", "Untitled"),
                        "score": best_score,
                        "message": best_message,
                        "updated_at": chat.get("updated_at", ""),
                    }
                )
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def _text_search_chats(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        results = []
        for chat_id, chat in self.chats.items():
            for message in chat.get("messages", []):
                content = message.get("content", "").lower()
                if query_lower in content:
                    results.append(
                        {
                            "chat_id": chat_id,
                            "title": chat.get("title", "Untitled"),
                            "score": 0.5,
                            "message": message,
                            "updated_at": chat.get("updated_at", ""),
                        }
                    )
                    break
        return results[:limit]

    def get_cached_response(
        self, prompt: str, model: str, system_prompt: str = ""
    ) -> Optional[Dict[str, Any]]:
        cache_key = get_cache_key(prompt, model, system_prompt)
        return _response_cache.get(cache_key)

    def cache_response(
        self, prompt: str, model: str, response: str, system_prompt: str = ""
    ) -> None:
        cache_key = get_cache_key(prompt, model, system_prompt)
        _response_cache[cache_key] = {
            "response": response,
            "timestamp": datetime.now().isoformat(),
            "model": model,
        }
        if len(_response_cache) > 1000:
            sorted_items = sorted(
                _response_cache.items(), key=lambda x: x[1].get("timestamp", "")
            )
            for key, _ in sorted_items[:-1000]:
                del _response_cache[key]
