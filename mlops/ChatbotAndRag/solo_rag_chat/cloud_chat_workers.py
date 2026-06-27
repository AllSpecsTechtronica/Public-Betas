"""
QThread workers for cloud chat APIs (OpenAI-compatible, Anthropic, Gemini).
Uses httpx async streaming where supported; keys stay in the application layer.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import httpx
from PyQt6.QtCore import QThread, pyqtSignal


def chat_messages_to_openai(msgs: list[dict]) -> list[dict[str, Any]]:
    """Turn stored chat rows into OpenAI ``chat.completions`` message objects."""
    out: list[dict[str, Any]] = []
    for m in msgs[-40:]:
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role in ("user", "human"):
            out.append({"role": "user", "content": content})
        elif role in ("assistant", "ai", "model"):
            out.append({"role": "assistant", "content": content})
        elif role == "system":
            out.append({"role": "system", "content": content})
    return out


def merge_adjacent_same_role(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anthropic requires strict user/assistant alternation; merge consecutive same roles."""
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = str(m.get("content", ""))
        if not content.strip():
            continue
        if not out:
            out.append({"role": role, "content": content})
            continue
        if out[-1].get("role") == role:
            out[-1]["content"] = str(out[-1].get("content", "")).strip() + "\n\n" + content
        else:
            out.append({"role": role, "content": content})
    return out


def chat_messages_to_anthropic(msgs: list[dict]) -> tuple[Optional[str], list[dict[str, Any]]]:
    """Return ``(system_text | None, anthropic messages)``."""
    system_chunks: list[str] = []
    messages: list[dict[str, Any]] = []
    for m in msgs[-40:]:
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            system_chunks.append(content)
            continue
        if role in ("user", "human"):
            messages.append({"role": "user", "content": content})
        elif role in ("assistant", "ai", "model"):
            messages.append({"role": "assistant", "content": content})
    system = "\n\n".join(system_chunks).strip() or None
    return system, merge_adjacent_same_role(messages)


def chat_messages_to_gemini(msgs: list[dict]) -> list[dict[str, Any]]:
    """Gemini ``contents`` entries use roles ``user`` and ``model``."""
    contents: list[dict[str, Any]] = []
    for m in msgs[-40:]:
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role in ("user", "human"):
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role in ("assistant", "ai", "model"):
            contents.append({"role": "model", "parts": [{"text": content}]})
        elif role == "system":
            contents.append({"role": "user", "parts": [{"text": f"[system]\n{content}"}]})
    return contents


class OpenAICompatChatWorker(QThread):
    """Streaming ``/v1/chat/completions`` (OpenAI, xAI Grok, and other compatible hosts)."""

    token_received = pyqtSignal(str)
    response_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[dict[str, Any]],
        timeout: float = 180.0,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model
        self.messages = messages
        self.timeout = timeout
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            asyncio.run(self._execute())
        except Exception as e:
            self.error_occurred.emit(str(e))

    async def _execute(self) -> None:
        if not self.api_key:
            self.error_occurred.emit("Missing API key.")
            return
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": self.messages,
            "stream": True,
        }
        full = ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    if response.status_code != 200:
                        detail = (await response.aread()).decode("utf-8", errors="ignore")[:800]
                        self.error_occurred.emit(
                            f"HTTP {response.status_code}: {detail or response.reason_phrase}"
                        )
                        return
                    async for line in response.aiter_lines():
                        if self._cancelled:
                            self.error_occurred.emit("Cancelled")
                            return
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        if line == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        try:
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            delta = (choices[0] or {}).get("delta") or {}
                            piece = delta.get("content") or ""
                            if isinstance(piece, str) and piece:
                                full += piece
                                self.token_received.emit(piece)
                        except Exception:
                            continue
        except httpx.HTTPError as e:
            self.error_occurred.emit(str(e))
            return
        self.response_received.emit({"full_response": full})


class AnthropicChatWorker(QThread):
    """Streaming Anthropic Messages API."""

    token_received = pyqtSignal(str)
    response_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        system: Optional[str],
        messages: list[dict[str, Any]],
        timeout: float = 180.0,
    ) -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.model = model
        self.system = system
        self.messages = messages
        self.timeout = timeout
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            asyncio.run(self._execute())
        except Exception as e:
            self.error_occurred.emit(str(e))

    async def _execute(self) -> None:
        if not self.api_key:
            self.error_occurred.emit("Missing Anthropic API key.")
            return
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 8192,
            "messages": self.messages,
            "stream": True,
        }
        if self.system:
            body["system"] = self.system
        full = ""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, headers=headers, json=body) as response:
                    if response.status_code != 200:
                        detail = (await response.aread()).decode("utf-8", errors="ignore")[:800]
                        msg = f"Anthropic HTTP {response.status_code}: {detail or response.reason_phrase}"
                        if response.status_code == 404 and "not_found_error" in detail:
                            msg += (
                                " Tip: older snapshot ids (e.g. claude-3-opus-20240229) are often removed; "
                                "use a current id such as claude-opus-4-7 or claude-sonnet-4-6."
                            )
                        self.error_occurred.emit(msg)
                        return
                    async for line in response.aiter_lines():
                        if self._cancelled:
                            self.error_occurred.emit("Cancelled")
                            return
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            evt = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        etype = str(evt.get("type", ""))
                        if etype == "content_block_delta":
                            delta = evt.get("delta") or {}
                            if delta.get("type") == "text_delta":
                                piece = str(delta.get("text") or "")
                                if piece:
                                    full += piece
                                    self.token_received.emit(piece)
                        elif etype == "message_stop":
                            break
        except httpx.HTTPError as e:
            self.error_occurred.emit(str(e))
            return
        self.response_received.emit({"full_response": full})


class GeminiChatWorker(QThread):
    """Non-streaming Gemini ``generateContent`` (single response)."""

    token_received = pyqtSignal(str)
    response_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        contents: list[dict[str, Any]],
        timeout: float = 180.0,
    ) -> None:
        super().__init__()
        self.api_key = api_key.strip()
        self.model = model
        self.contents = contents
        self.timeout = timeout
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            asyncio.run(self._execute())
        except Exception as e:
            self.error_occurred.emit(str(e))

    async def _execute(self) -> None:
        if not self.api_key:
            self.error_occurred.emit("Missing Gemini API key.")
            return
        mid = self.model.strip()
        if mid.startswith("models/"):
            mid = mid[len("models/") :]
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{mid}:generateContent"
        )
        params = {"key": self.api_key}
        body = {"contents": self.contents}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(url, params=params, json=body)
        except httpx.HTTPError as e:
            self.error_occurred.emit(str(e))
            return
        if self._cancelled:
            self.error_occurred.emit("Cancelled")
            return
        if r.status_code != 200:
            self.error_occurred.emit(
                f"Gemini HTTP {r.status_code}: {r.text[:800]!r}"
            )
            return
        try:
            data = r.json()
        except Exception:
            self.error_occurred.emit("Gemini returned invalid JSON.")
            return
        text = ""
        try:
            cands = data.get("candidates") or []
            if cands:
                parts = (cands[0].get("content") or {}).get("parts") or []
                for p in parts:
                    t = str(p.get("text") or "")
                    if t:
                        text += t
        except Exception:
            text = ""
        if not text.strip():
            self.error_occurred.emit("Gemini returned an empty reply.")
            return
        self.token_received.emit(text)
        self.response_received.emit({"full_response": text})
