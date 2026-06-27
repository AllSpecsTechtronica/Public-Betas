"""
Run solo RAG + chat desktop: python -m solo_rag_chat
(from repository root). Uses only solo_rag_chat/_solo_data for storage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .chat_manager import ChatManager
from .paths import ensure_solo_data_layout, get_rag_index_dir, get_solo_data_root
from .rag_system import RAG_DEPENDENCIES_AVAILABLE, reset_rag_system, set_rag_config
from .workers import OllamaWorker, RAGWorker


def _build_ollama_prompt(messages: List[dict]) -> str:
    """Format recent turns for Ollama /api/generate completion."""
    parts: List[str] = []
    for m in messages[-40:]:
        role = str(m.get("role", "")).strip().lower()
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        if role in ("user", "human"):
            parts.append(f"User: {content}")
        elif role in ("assistant", "ai", "model"):
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"{role}: {content}")
    if not parts:
        return "Assistant:"
    return "\n\n".join(parts) + "\n\nAssistant:"


class SoloWindow(QMainWindow):
    """Minimal combined chat composer and RAG console."""

    def __init__(self) -> None:
        super().__init__()
        ensure_solo_data_layout()
        self.setWindowTitle("Solo RAG + Chat (isolated from Techtronica prime)")
        self.resize(960, 720)

        self.chat_manager = ChatManager()
        self.current_chat_id: Optional[str] = None
        self._ollama_worker: Optional[OllamaWorker] = None

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        info = QLabel(
            f"Data root (solo only): {get_solo_data_root()}\n"
            "No paths from techtronica prime are read or written."
        )
        info.setWordWrap(True)
        outer.addWidget(info)

        tabs = QTabWidget()
        outer.addWidget(tabs)

        tabs.addTab(self._build_chat_tab(), "AI Chat")
        tabs.addTab(self._build_rag_tab(), "RAG")

    def _build_chat_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        row = QHBoxLayout()
        row.addWidget(QLabel("Ollama base URL"))
        self.chat_ollama_url = QLineEdit("http://localhost:11434")
        row.addWidget(self.chat_ollama_url)
        row.addWidget(QLabel("Model"))
        self.chat_model = QLineEdit("gemma3:4b")
        row.addWidget(self.chat_model)
        layout.addLayout(row)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.chat_list = QListWidget()
        self.chat_list.currentItemChanged.connect(self._on_chat_selected)
        split.addWidget(self.chat_list)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.chat_view = QTextBrowser()
        self.chat_view.setOpenExternalLinks(True)
        rv.addWidget(self.chat_view)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Message...")
        self.chat_input.returnPressed.connect(self._send_chat)
        rv.addWidget(self.chat_input)
        btn_row = QHBoxLayout()
        new_btn = QPushButton("New chat")
        new_btn.clicked.connect(self._new_chat)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send_chat)
        btn_row.addWidget(new_btn)
        btn_row.addWidget(send_btn)
        rv.addLayout(btn_row)
        split.addWidget(right)
        split.setSizes([220, 700])
        layout.addWidget(split)

        self._refresh_chat_list()
        existing = self.chat_manager.list_chats()
        if existing:
            self.current_chat_id = existing[0]["id"]
        else:
            self.current_chat_id = self.chat_manager.create_chat("Solo chat")
            self._refresh_chat_list()
        self._select_chat_id(self.current_chat_id or "")
        self._show_current_chat()
        return w

    def _build_rag_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        if not RAG_DEPENDENCIES_AVAILABLE:
            layout.addWidget(
                QLabel(
                    "RAG Python dependencies are not installed. "
                    "Install with: pip install -r solo_rag_chat/requirements-solo.txt"
                )
            )
            return w

        form = QFormLayout()
        self.rag_ollama_url = QLineEdit("http://localhost:11434")
        form.addRow("Ollama URL", self.rag_ollama_url)
        self.rag_chat_model = QLineEdit("gemma3:4b")
        form.addRow("Answer model", self.rag_chat_model)
        self.rag_embed_backend = QComboBox()
        self.rag_embed_backend.addItems(["ollama", "huggingface"])
        form.addRow("Embedding backend", self.rag_embed_backend)
        self.rag_embed_model = QLineEdit("nomic-embed-text")
        form.addRow("Embedding model", self.rag_embed_model)
        layout.addLayout(form)

        self.rag_files_label = QLabel("No files selected")
        self.rag_files_label.setWordWrap(True)
        layout.addWidget(self.rag_files_label)
        self._rag_file_paths: List[str] = []

        pick = QPushButton("Choose files (txt, md, pdf)")
        pick.clicked.connect(self._pick_rag_files)
        layout.addWidget(pick)

        row = QHBoxLayout()
        build_btn = QPushButton("Build index")
        build_btn.clicked.connect(self._rag_build)
        clear_btn = QPushButton("Clear index")
        clear_btn.clicked.connect(self._rag_clear)
        status_btn = QPushButton("Status")
        status_btn.clicked.connect(self._rag_status)
        row.addWidget(build_btn)
        row.addWidget(clear_btn)
        row.addWidget(status_btn)
        layout.addLayout(row)

        layout.addWidget(QLabel("RAG question"))
        self.rag_question = QLineEdit()
        layout.addWidget(self.rag_question)
        q_btn = QPushButton("Query RAG")
        q_btn.clicked.connect(self._rag_query)
        layout.addWidget(q_btn)

        self.rag_output = QTextBrowser()
        layout.addWidget(self.rag_output)
        return w

    def _apply_rag_config(self) -> None:
        reset_rag_system()
        backend = self.rag_embed_backend.currentText().strip().lower()
        set_rag_config(
            model_id=self.rag_chat_model.text().strip(),
            embedding_backend=backend,
            embedding_model=self.rag_embed_model.text().strip(),
            ollama_base_url=self.rag_ollama_url.text().strip(),
            rag_index_path=str(get_rag_index_dir()),
        )

    def _pick_rag_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select documents",
            str(Path.home()),
            "Documents (*.txt *.md *.pdf);;All (*.*)",
        )
        if paths:
            self._rag_file_paths = list(paths)
            self.rag_files_label.setText("\n".join(self._rag_file_paths))

    def _rag_build(self) -> None:
        if not self._rag_file_paths:
            QMessageBox.warning(self, "RAG", "Select at least one file.")
            return
        self._apply_rag_config()
        self.rag_output.append("[RAG] Building index...")
        worker = RAGWorker("build", files=self._rag_file_paths)
        worker.finished.connect(self._on_rag_finished)
        worker.error.connect(self._on_rag_error)
        worker.start()

    def _rag_clear(self) -> None:
        self._apply_rag_config()
        worker = RAGWorker("clear")
        worker.finished.connect(self._on_rag_finished)
        worker.error.connect(self._on_rag_error)
        worker.start()

    def _rag_status(self) -> None:
        self._apply_rag_config()
        worker = RAGWorker("status")
        worker.finished.connect(self._on_rag_finished)
        worker.error.connect(self._on_rag_error)
        worker.start()

    def _rag_query(self) -> None:
        q = self.rag_question.text().strip()
        if not q:
            return
        self._apply_rag_config()
        worker = RAGWorker("query", question=q, k=4, return_sources=True)
        worker.finished.connect(self._on_rag_query_finished)
        worker.error.connect(self._on_rag_error)
        worker.start()

    def _on_rag_finished(self, payload: dict) -> None:
        self.rag_output.append(str(payload))

    def _on_rag_query_finished(self, payload: dict) -> None:
        ans = payload.get("answer", "")
        self.rag_output.append(f"Answer:\n{ans}\n")
        if payload.get("sources"):
            self.rag_output.append(f"Sources: {payload.get('sources')}\n")

    def _on_rag_error(self, msg: str) -> None:
        self.rag_output.append(f"[ERROR] {msg}")
        QMessageBox.warning(self, "RAG", msg)

    def _refresh_chat_list(self) -> None:
        self.chat_list.clear()
        for meta in self.chat_manager.list_chats():
            item = QListWidgetItem(meta.get("title", meta["id"]))
            item.setData(Qt.ItemDataRole.UserRole, meta["id"])
            self.chat_list.addItem(item)

    def _on_chat_selected(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        if not current:
            return
        cid = current.data(Qt.ItemDataRole.UserRole)
        if cid:
            self.current_chat_id = str(cid)
            self._show_current_chat()

    def _new_chat(self) -> None:
        self.current_chat_id = self.chat_manager.create_chat("New chat")
        self._refresh_chat_list()
        self._select_chat_id(self.current_chat_id)
        self._show_current_chat()

    def _select_chat_id(self, chat_id: str) -> None:
        for i in range(self.chat_list.count()):
            it = self.chat_list.item(i)
            if it and it.data(Qt.ItemDataRole.UserRole) == chat_id:
                self.chat_list.setCurrentItem(it)
                break

    def _show_current_chat(self) -> None:
        if not self.current_chat_id:
            return
        lines = []
        for m in self.chat_manager.get_chat_messages(self.current_chat_id):
            role = m.get("role", "")
            lines.append(f"<b>{role}</b>: {m.get('content', '')}")
        self.chat_view.setHtml("<br><br>".join(lines) if lines else "<i>Empty</i>")

    def _send_chat(self) -> None:
        if not self.current_chat_id:
            self.current_chat_id = self.chat_manager.create_chat("Solo chat")
            self._refresh_chat_list()
        text = self.chat_input.text().strip()
        if not text:
            return
        if self._ollama_worker and self._ollama_worker.isRunning():
            QMessageBox.information(self, "Chat", "Wait for the current reply to finish.")
            return

        self.chat_manager.add_message(self.current_chat_id, "user", text)
        self.chat_input.clear()
        self._show_current_chat()

        msgs = self.chat_manager.get_chat_messages(self.current_chat_id)
        prompt = _build_ollama_prompt(msgs)

        base = self.chat_ollama_url.text().strip().rstrip("/")
        model = self.chat_model.text().strip()
        worker = OllamaWorker(
            base_url=base,
            model=model,
            prompt=prompt,
            system_prompt=None,
        )
        self._ollama_worker = worker
        worker.token_received.connect(self._on_chat_token)
        worker.response_received.connect(self._on_chat_done)
        worker.error_occurred.connect(self._on_chat_err)
        self.chat_view.append("<br><b>assistant</b>: ")
        worker.start()

    def _on_chat_token(self, tok: str) -> None:
        self.chat_view.moveCursor(self.chat_view.textCursor().End)
        self.chat_view.insertPlainText(tok)

    def _on_chat_done(self, payload: dict) -> None:
        full = str(payload.get("full_response", "")).strip()
        if self.current_chat_id and full:
            self.chat_manager.add_message(self.current_chat_id, "assistant", full)
        self._ollama_worker = None

    def _on_chat_err(self, msg: str) -> None:
        self.chat_view.append(f"<br><span style='color:red'>{msg}</span>")
        self._ollama_worker = None


def main() -> None:
    app = QApplication(sys.argv)
    win = SoloWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
