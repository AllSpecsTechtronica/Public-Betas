"""
QThread workers for Ollama API and RAG (solo_rag_chat package only).
"""

import asyncio
import json
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
import httpx


class OllamaWorker(QThread):
    """Worker thread for async Ollama requests. Uses its own event loop and HTTP client to avoid 'Event loop is closed'."""

    DEFAULT_REASONING_FIELDS = ("thinking", "reasoning", "thought", "think", "reason")
    DEFAULT_STREAM_IDLE_TIMEOUT_SEC = 45.0
    MIN_STREAM_IDLE_TIMEOUT_SEC = 1.0
    MAX_STREAM_IDLE_TIMEOUT_SEC = 600.0

    response_received = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    token_received = pyqtSignal(str)

    def __init__(
        self,
        base_url: str,
        model: str,
        prompt: str,
        system_prompt=None,
        images=None,
        parameters=None,
        timeout: float = 600.0,
        stream_idle_timeout_sec: float = DEFAULT_STREAM_IDLE_TIMEOUT_SEC,
    ):
        super().__init__()
        self.base_url = base_url
        self.timeout = timeout
        self.model = model
        self.prompt = prompt
        self.system_prompt = system_prompt
        self.images = [
            str(item or "").strip() for item in (images or []) if str(item or "").strip()
        ]
        self.parameters = parameters or {}
        try:
            parsed_idle = float(stream_idle_timeout_sec)
        except Exception:
            parsed_idle = self.DEFAULT_STREAM_IDLE_TIMEOUT_SEC
        if not (parsed_idle == parsed_idle and abs(parsed_idle) != float("inf")):
            parsed_idle = self.DEFAULT_STREAM_IDLE_TIMEOUT_SEC
        self.stream_idle_timeout_sec = max(
            self.MIN_STREAM_IDLE_TIMEOUT_SEC,
            min(parsed_idle, self.MAX_STREAM_IDLE_TIMEOUT_SEC),
        )
        self._cancelled = False

    def cancel(self):
        """Cancel the request."""
        self._cancelled = True

    def _get_reasoning_fields(self) -> list[str]:
        """Return configured reasoning field names plus defaults."""
        configured = []
        raw = self.parameters.get("thinking_fields")
        if isinstance(raw, list):
            configured = [str(x).strip().lower() for x in raw if str(x).strip()]
        out = []
        seen = set()
        for key in configured + list(self.DEFAULT_REASONING_FIELDS):
            lower = str(key or "").strip().lower()
            if not lower or lower in seen:
                continue
            seen.add(lower)
            out.append(lower)
        return out

    def _extract_reasoning_token(self, payload: dict, fields: list[str]) -> str:
        """Extract reasoning text from known keys with heuristic fallback."""
        for key in fields:
            maybe = payload.get(key)
            if isinstance(maybe, str) and maybe:
                return maybe
        for raw_key, raw_val in payload.items():
            key = str(raw_key or "").strip().lower()
            if (
                ("think" in key or "reason" in key)
                and isinstance(raw_val, str)
                and raw_val
            ):
                return raw_val
        return ""

    def run(self):
        """Run async Ollama request."""
        try:
            asyncio.run(self._execute())
        except Exception as e:
            self.error_occurred.emit(str(e))

    async def _execute(self):
        """Execute the Ollama request. Omit num_predict when 0 or missing to stream until model ends."""
        options = {
            "temperature": self.parameters.get("temperature", 0.7),
            "top_p": self.parameters.get("top_p", 0.9),
            "top_k": self.parameters.get("top_k", 40),
        }
        num_predict = self.parameters.get("num_predict")
        if num_predict is not None and num_predict > 0:
            options["num_predict"] = num_predict
        request_data = {
            "model": self.model,
            "prompt": self.prompt,
            "stream": True,
            "options": options,
        }

        if self.system_prompt:
            request_data["system"] = self.system_prompt
        if self.images:
            request_data["images"] = list(self.images)

        try:
            async with httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout
            ) as client:
                async with client.stream(
                    "POST", "/api/generate", json=request_data
                ) as response:
                    if response.status_code == 200:
                        full_response = ""
                        in_reasoning_block = False
                        reasoning_fields = self._get_reasoning_fields()
                        saw_done = False
                        line_iter = response.aiter_lines()
                        while True:
                            if self._cancelled:
                                self.error_occurred.emit("Cancelled")
                                return
                            try:
                                line = await asyncio.wait_for(
                                    line_iter.__anext__(),
                                    timeout=self.stream_idle_timeout_sec,
                                )
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                if in_reasoning_block:
                                    full_response += "</think>"
                                    self.token_received.emit("</think>")
                                msg = (
                                    "Stream stalled (no tokens for "
                                    f"{int(self.stream_idle_timeout_sec)}s)."
                                )
                                if full_response.strip():
                                    msg += " Partial output preserved."
                                self.error_occurred.emit(msg)
                                return

                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    response_token = data.get("response")
                                    if response_token is None:
                                        response_token = ""
                                    response_token = str(response_token)

                                    reasoning_token = self._extract_reasoning_token(
                                        data,
                                        reasoning_fields,
                                    )

                                    if reasoning_token and not response_token:
                                        if not in_reasoning_block:
                                            full_response += "<think>"
                                            self.token_received.emit("<think>")
                                            in_reasoning_block = True
                                        full_response += reasoning_token
                                        self.token_received.emit(reasoning_token)
                                        continue

                                    if response_token:
                                        lower_token = response_token.lower()
                                        if (
                                            "<think>" in lower_token
                                            or "</think>" in lower_token
                                        ):
                                            # model emits think tags inline in response stream
                                            pass
                                        if in_reasoning_block:
                                            full_response += "</think>"
                                            self.token_received.emit("</think>")
                                            in_reasoning_block = False
                                        full_response += response_token
                                        self.token_received.emit(response_token)

                                    if data.get("done", False):
                                        saw_done = True
                                        if in_reasoning_block:
                                            full_response += "</think>"
                                            self.token_received.emit("</think>")
                                            in_reasoning_block = False
                                        self.response_received.emit(
                                            {
                                                "full_response": full_response,
                                                "stats": data,
                                            }
                                        )
                                        break
                                except json.JSONDecodeError:
                                    continue
                        if not saw_done:
                            if in_reasoning_block:
                                full_response += "</think>"
                                self.token_received.emit("</think>")
                            if full_response.strip():
                                self.response_received.emit(
                                    {
                                        "full_response": full_response,
                                        "stats": {"stream_done": False},
                                        "warnings": [
                                            "Model stream ended without done=true; returned preserved partial output."
                                        ],
                                    }
                                )
                            else:
                                msg = "Stream interrupted (connection closed before done=true)."
                                self.error_occurred.emit(msg)
                    else:
                        body_text = ""
                        try:
                            raw_body = await response.aread()
                            if raw_body:
                                body_text = raw_body.decode(
                                    "utf-8", errors="ignore"
                                ).strip()
                        except Exception:
                            body_text = ""

                        body_error = ""
                        if body_text:
                            try:
                                parsed = json.loads(body_text)
                                if isinstance(parsed, dict):
                                    body_error = str(
                                        parsed.get("error")
                                        or parsed.get("message")
                                        or ""
                                    ).strip()
                            except Exception:
                                body_error = ""
                        retry_after = str(
                            response.headers.get("retry-after", "") or ""
                        ).strip()
                        queue_pos = str(
                            response.headers.get("x-queue-position")
                            or response.headers.get("x-queue-pos")
                            or ""
                        ).strip()

                        if response.status_code == 429:
                            msg = "Ollama rate limited (HTTP 429)"
                            if retry_after:
                                msg += f" | retry_after={retry_after}"
                            if queue_pos:
                                msg += f" | queue={queue_pos}"
                            if body_error:
                                msg += f" | detail={body_error}"
                            elif body_text:
                                compact = " ".join(body_text.split())
                                msg += f" | detail={compact[:220]}"
                        else:
                            msg = f"Ollama error: HTTP {response.status_code}"
                            if retry_after:
                                msg += f" | retry_after={retry_after}"
                            if queue_pos:
                                msg += f" | queue={queue_pos}"
                            detail = body_error or (
                                " ".join(body_text.split()) if body_text else ""
                            )
                            if detail:
                                msg += f" | detail={detail[:220]}"
                        self.error_occurred.emit(msg)
        except Exception as e:
            self.error_occurred.emit(f"Request error: {str(e)}")


class RAGWorker(QThread):
    """Worker thread for RAG operations."""

    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, action, namespace=None, **kwargs):
        super().__init__()
        self.action = action
        self.namespace = namespace
        self.kwargs = kwargs

    def _emit_progress(self, message: str) -> None:
        """Thread-safe progress line (queued across to the GUI thread by Qt)."""
        try:
            self.progress.emit(str(message))
        except Exception:
            pass

    def run(self):
        try:
            asyncio.run(self._execute())
        except Exception as e:
            self.error.emit(str(e))

    async def _execute(self):
        from .rag_system import (
            DEFAULT_NAMESPACE,
            RAG_DEPENDENCIES_AVAILABLE,
            get_rag_system,
        )

        if not RAG_DEPENDENCIES_AVAILABLE:
            raise Exception(
                "RAG dependencies not installed. Install with: pip install sentence-transformers faiss-cpu pymupdf openai"
            )

        namespace = self.namespace or DEFAULT_NAMESPACE
        self._emit_progress(f"Connecting to RAG engine (namespace={namespace!r})…")
        rag = await get_rag_system(namespace)
        if not rag:
            raise Exception(
                "Could not initialize RAG system. Check RAG dependencies and configuration."
            )

        result = {}
        if self.action in ("build", "add"):
            files = self.kwargs.get("files", [])
            extra_context_files = self.kwargs.get("extra_context_files", [])
            self._emit_progress(
                f"Engine ready (embeddings={rag.embedding_backend}:{rag.embedding_model}). "
                f"Reading {len(files)} file(s)…"
            )
            context_files = []
            file_context_count = 0
            extra_context_count = 0
            for f in files:
                path = Path(f)
                if not path.exists():
                    continue

                content = ""
                if path.suffix.lower() == ".pdf":
                    content = "PDF_FILE"
                else:
                    try:
                        content = path.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue

                context_files.append(
                    {"path": str(path), "name": path.name, "content": content}
                )
                file_context_count += 1

            if isinstance(extra_context_files, list):
                for row in extra_context_files:
                    if not isinstance(row, dict):
                        continue
                    path = str(row.get("path") or "").strip()
                    name = str(row.get("name") or "").strip()
                    content = str(row.get("content") or "").strip()
                    if not path or not content:
                        continue
                    context_files.append(
                        {
                            "path": path,
                            "name": name or Path(path).name,
                            "content": content,
                        }
                    )
                    extra_context_count += 1

            if self.action == "add":
                # Incremental: append to the existing index instead of rebuilding.
                count = await rag.add_context_files_to_index(
                    context_files, progress_cb=self._emit_progress
                )
                verb = "Added"
            else:
                count = await rag.build_index_from_context_files(
                    context_files, progress_cb=self._emit_progress
                )
                verb = "Indexed"
            source_parts = [f"{file_context_count} file docs"]
            if extra_context_count:
                source_parts.append(f"{extra_context_count} gallery docs")
            result = {
                "count": count,
                "namespace": namespace,
                "message": (
                    f"{verb} {count} chunks from {len(context_files)} context docs "
                    f"({', '.join(source_parts)})"
                ),
            }

        elif self.action == "clear":
            await rag.clear_index()
            result = {"namespace": namespace, "message": "Index cleared"}

        elif self.action == "status":
            info = rag.get_index_info()
            result = {"namespace": namespace, "info": info}

        elif self.action == "query":
            question = str(self.kwargs.get("question", "") or "")
            result = await rag.query_rag(
                question,
                k=int(self.kwargs.get("k", 4)),
                return_sources=bool(self.kwargs.get("return_sources", True)),
            )
            result["namespace"] = namespace
        else:
            raise ValueError(f"Unknown RAG worker action: {self.action!r}")

        self.finished.emit(result)
