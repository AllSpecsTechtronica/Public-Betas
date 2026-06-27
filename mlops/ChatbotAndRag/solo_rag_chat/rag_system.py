"""
Solo RAG Chat: standalone RAG stack (forked from Techtronica prime).
Local Retrieval-Augmented Generation. No imports from techtronica prime.
"""

import os
import sys
import json
import logging
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import asyncio
from datetime import datetime

# Explicitly check for sentence_transformers before importing langchain components
try:
    import sentence_transformers
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

# LangChain imports with graceful fallbacks (support langchain 0.2+ and legacy)
try:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore[reportMissingImports]
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
    except ImportError:
        from langchain.vectorstores import FAISS  # type: ignore[reportMissingImports]
        from langchain.document_loaders import PyMuPDFLoader, TextLoader  # type: ignore[reportMissingImports]
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[reportMissingImports]
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore[reportMissingImports]
    ChatOllama = None  # type: ignore[misc, assignment]
    try:
        from langchain_ollama import OllamaEmbeddings, ChatOllama  # type: ignore[reportMissingImports]
    except ImportError:
        try:
            from langchain_community.embeddings import OllamaEmbeddings
        except ImportError:
            OllamaEmbeddings = None  # type: ignore[misc, assignment]
        try:
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            pass  # ChatOllama already None
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.documents import Document
    except ImportError:
        from langchain.prompts import ChatPromptTemplate  # type: ignore[reportMissingImports]
        from langchain.schema import Document  # type: ignore[reportMissingImports]
    # Modern LCEL chains (replacement for deprecated RetrievalQA)
    # In langchain 1.x, chains module was removed, so we use manual LCEL
    LCEL_AVAILABLE = False
    RetrievalQA = None  # type: ignore[misc, assignment]
    try:
        # Check if we can use manual LCEL (works with langchain 1.x)
        try:
            from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # type: ignore[reportMissingImports]
            from langchain_core.prompts import ChatPromptTemplate  # Already imported above
            LCEL_AVAILABLE = True
        except ImportError:
            LCEL_AVAILABLE = False
        
        # Try to import helper functions if available (langchain < 1.0)
        create_retrieval_chain = None  # type: ignore[misc, assignment]
        create_stuff_documents_chain = None  # type: ignore[misc, assignment]
        if not LCEL_AVAILABLE:
            try:
                from langchain.chains.retrieval import create_retrieval_chain  # type: ignore[reportMissingImports]
                from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore[reportMissingImports]
                LCEL_AVAILABLE = True
            except ImportError:
                try:
                    from langchain.chains import create_retrieval_chain  # type: ignore[reportMissingImports]
                    from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore[reportMissingImports]
                    LCEL_AVAILABLE = True
                except ImportError:
                    pass
        
        # Legacy RetrievalQA import (for older langchain versions)
        if not LCEL_AVAILABLE:
            try:
                from langchain_community.chains.retrieval_qa.base import RetrievalQA  # type: ignore[reportMissingImports]
            except ImportError:
                try:
                    from langchain_community.chains import RetrievalQA  # type: ignore[reportMissingImports]
                except ImportError:
                    try:
                        from langchain.chains import RetrievalQA  # type: ignore[reportMissingImports]
                    except ImportError:
                        RetrievalQA = None
    except Exception:
        LCEL_AVAILABLE = False
        RetrievalQA = None
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[reportMissingImports]
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOpenAI
        except ImportError:
            from langchain.chat_models import ChatOpenAI  # type: ignore[reportMissingImports]
    import torch
    RAG_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"[RAG] Warning: RAG dependencies not available: {e}")
    RAG_DEPENDENCIES_AVAILABLE = False
    LCEL_AVAILABLE = False
    create_retrieval_chain = None  # type: ignore[misc, assignment]
    create_stuff_documents_chain = None  # type: ignore[misc, assignment]
    RetrievalQA = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


class SoloRAGSystem:
    """
    Solo RAG System - Local Retrieval-Augmented Generation (standalone package).
    Uses LM Studio or Ollama for generation and local embeddings for retrieval.
    """

    def __init__(
        self,
        lms_base_url: str = "http://localhost:1234/v1",
        lms_api_key: str = "lm-studio",
        model_id: str = "granite-3.1-2b-instruct-Q5_K_M",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_backend: str = "huggingface",
        ollama_base_url: str = "http://localhost:11434",
        device: str = "auto",
        rag_index_path: str = "rag_index",
        max_context_length: int = 4000,
    ):
        """
        Initialize solo RAG system.

        Args:
            lms_base_url: LM Studio API base URL
            lms_api_key: API key (dummy for LM Studio)
            model_id: Model ID from LM Studio
            embedding_model: HuggingFace model name or Ollama embedding model name
            embedding_backend: "huggingface" or "ollama"
            ollama_base_url: Ollama API base URL (when embedding_backend is "ollama")
            device: Device for embeddings (auto/cuda/cpu) when using HuggingFace
            rag_index_path: Path to store FAISS index
            max_context_length: Maximum context length for generation
        """
        if not RAG_DEPENDENCIES_AVAILABLE:
            raise ImportError("RAG dependencies not available. Install with: pip install sentence-transformers faiss-cpu pymupdf openai langchain langchain-community")
        
        # Check for modern LCEL chains or legacy RetrievalQA
        if not LCEL_AVAILABLE and RetrievalQA is None:
            raise ImportError("RAG chain not available. Install compatible langchain version: pip install langchain langchain-community")

        self.lms_base_url = lms_base_url
        self.lms_api_key = lms_api_key
        self.model_id = model_id
        self.embedding_model = embedding_model
        self.embedding_backend = (embedding_backend or "huggingface").lower()
        self.ollama_base_url = ollama_base_url or "http://localhost:11434"
        self.rag_index_path = rag_index_path
        self.max_context_length = max_context_length
        self.is_initialized = False
        self.vectorstore = None

        # Device detection for HuggingFace embeddings
        if device == "auto":
            try:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device = "cpu"
        else:
            self.device = device

        print(f"[RAG] Initializing solo RAG system...")
        print(f"[RAG] Model: {model_id}")
        print(f"[RAG] Embedding backend: {self.embedding_backend}")
        print(f"[RAG] Embedding model: {embedding_model}")
        print(f"[RAG] Device: {self.device}")

        # Initialize embeddings: Ollama or HuggingFace
        if self.embedding_backend == "ollama":
            if OllamaEmbeddings is None:
                raise ImportError("Ollama embeddings not available. Install: pip install langchain-ollama (or langchain-community)")
            try:
                self.embeddings = OllamaEmbeddings(
                    model=embedding_model,
                    base_url=self.ollama_base_url,
                )
                print(f"[RAG] Ollama embeddings initialized: {embedding_model}")
            except Exception as e:
                print(f"[RAG] Error initializing Ollama embeddings: {e}")
                raise
        else:
            # HuggingFace embeddings (check for sentence_transformers first)
            # Pre-import sentence_transformers to ensure it's available before HuggingFaceEmbeddings tries to use it
            try:
                import sentence_transformers
                print(f"[RAG] sentence_transformers imported successfully from: {sentence_transformers.__file__}")
            except ImportError as ie:
                # Check if it's a transformers version conflict
                if "PreTrainedModel" in str(ie) or "transformers" in str(ie).lower():
                    error_msg = f"Version conflict detected with transformers package.\n"
                    error_msg += f"Python executable: {sys.executable}\n"
                    error_msg += f"Error: {ie}\n\n"
                    error_msg += f"This usually means there are conflicting transformers versions.\n"
                    error_msg += f"Try: pip install --upgrade transformers sentence-transformers\n"
                    error_msg += f"Or check if you have multiple venv directories (.venv vs venv)"
                    print(f"[RAG] Error initializing embeddings: {error_msg}")
                    raise ImportError(error_msg)
                else:
                    error_msg = f"Could not import sentence_transformers python package.\n"
                    error_msg += f"Please ensure it's installed: pip install sentence-transformers\n"
                    error_msg += f"Python executable: {sys.executable}\n"
                    error_msg += f"First 5 sys.path entries: {sys.path[:5]}\n"
                    error_msg += f"Import error: {ie}"
                    print(f"[RAG] Error initializing embeddings: {error_msg}")
                    raise ImportError(error_msg)
            
            # HuggingFace embeddings (avoid meta tensor by retrying on CPU)
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model, model_kwargs={"device": self.device}
                )
                print(f"[RAG] Embeddings initialized successfully")
            except Exception as e:
                if "meta tensor" in str(e).lower() or "to_empty" in str(e).lower():
                    try:
                        self.embeddings = HuggingFaceEmbeddings(
                            model_name=embedding_model, model_kwargs={"device": "cpu"}
                        )
                        self.device = "cpu"
                        print(f"[RAG] Embeddings initialized on CPU (fallback)")
                    except Exception as e2:
                        print(f"[RAG] Error initializing embeddings: {e2}")
                        raise
                else:
                    print(f"[RAG] Error initializing embeddings: {e}")
                    raise

        # LLM for RAG answer step: use Ollama when embedding_backend is ollama (no LM Studio required), else LM Studio
        try:
            if self.embedding_backend == "ollama" and ChatOllama is not None:
                self.llm = ChatOllama(
                    base_url=self.ollama_base_url,
                    model=model_id,
                    temperature=0.1,
                    num_predict=2048,
                )
                print(f"[RAG] Ollama LLM initialized for answers: {model_id}")
            else:
                self.llm = ChatOpenAI(
                    base_url=lms_base_url,
                    api_key=lms_api_key,
                    model=model_id,
                    temperature=0.1,
                    max_tokens=2048,
                )
                print(f"[RAG] LM Studio client initialized")
        except Exception as e:
            print(f"[RAG] Error initializing RAG LLM: {e}")
            raise

        # RAG-specific prompt template: answer only from context, no refusals or meta-commentary
        self.prompt_template = """
Answer the question using ONLY the context below. Do not refuse, ask to rephrase, or explain what RAG is. If the context has relevant information, summarize or quote it. If the context is empty or irrelevant, say briefly: "No relevant documents in the knowledge base."

Context:
{context}

Question: {question}

Answer:
"""
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)

        self.vectorstore = None
        self.is_initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the RAG system
        Loads existing index if available
        """
        try:
            if os.path.exists(self.rag_index_path):
                await self.load_index()
                logger.info("RAG system initialized with existing index")
            else:
                logger.info("RAG system initialized - no existing index found")
            
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False

    @staticmethod
    def _emit(progress_cb: Optional[Any], message: str) -> None:
        """Best-effort progress reporting; also mirrors to the logger."""
        logger.info(message)
        if progress_cb is not None:
            try:
                progress_cb(message)
            except Exception:
                pass

    def _load_documents(
        self,
        context_files: List[Dict[str, Any]],
        progress_cb: Optional[Any] = None,
    ) -> List[Any]:
        """Load raw documents from context files, reporting per-file progress."""
        docs: List[Any] = []
        total = len(context_files)
        for idx, file_info in enumerate(context_files, start=1):
            file_path = file_info.get('path', '')
            file_name = file_info.get('name', '')
            file_content = file_info.get('content', '')
            source_type = "gallery" if str(file_path).startswith("gallery://") else "file"

            if not file_path or not file_content:
                logger.warning(f"Skipping file with missing data: {file_name}")
                continue

            try:
                if file_path.lower().endswith('.pdf'):
                    if os.path.exists(file_path):
                        self._emit(progress_cb, f"Loading PDF {idx}/{total}: {file_name}…")
                        loader = PyMuPDFLoader(file_path)
                        file_docs = loader.load()
                    else:
                        logger.warning(f"PDF file not found: {file_path}")
                        continue
                else:
                    file_docs = [Document(
                        page_content=file_content,
                        metadata={
                            "source": file_path,
                            "filename": file_name,
                            "file_type": Path(file_path).suffix,
                            "source_type": source_type,
                            "added_at": datetime.now().isoformat()
                        }
                    )]

                for doc in file_docs:
                    doc.metadata.update({
                        "filename": file_name,
                        "file_type": Path(file_path).suffix,
                        "source_type": source_type,
                        "added_at": datetime.now().isoformat()
                    })

                docs.extend(file_docs)
                self._emit(
                    progress_cb,
                    f"Read {idx}/{total}: {file_name} ({len(file_docs)} page/section docs)",
                )
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                self._emit(progress_cb, f"[skip] {file_name}: {e}")
        return docs

    def _embed_in_batches(
        self,
        texts: List[str],
        progress_cb: Optional[Any] = None,
        batch_size: int = 16,
    ) -> List[List[float]]:
        """Embed texts in batches, reporting throughput + ETA so long runs are visible."""
        total = len(texts)
        vectors: List[List[float]] = []
        start = time.time()
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            t0 = time.time()
            vectors.extend(self.embeddings.embed_documents(batch))
            done = min(i + batch_size, total)
            elapsed = max(1e-6, time.time() - start)
            rate = done / elapsed
            eta = (total - done) / max(1e-6, rate)
            self._emit(
                progress_cb,
                f"Embedded {done}/{total} chunks "
                f"({time.time() - t0:.1f}s/batch, ~{rate:.1f} chunks/s, ETA {eta:.0f}s)",
            )
        return vectors

    def _split_and_build(
        self,
        docs: List[Any],
        chunk_size: int,
        chunk_overlap: int,
        progress_cb: Optional[Any] = None,
    ) -> int:
        """Chunk docs, embed with progress, and (re)build the FAISS index. Returns chunk count."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            self._emit(progress_cb, "No chunks produced from documents.")
            return 0
        self._emit(
            progress_cb,
            f"Split into {len(splits)} chunks (size={chunk_size}, overlap={chunk_overlap}). "
            f"Embedding via {self.embedding_backend}:{self.embedding_model}…",
        )

        texts = [d.page_content for d in splits]
        metadatas = [d.metadata for d in splits]
        build_start = time.time()
        try:
            vectors = self._embed_in_batches(texts, progress_cb=progress_cb)
            self._emit(progress_cb, "Building FAISS index from embeddings…")
            self.vectorstore = FAISS.from_embeddings(
                list(zip(texts, vectors)), self.embeddings, metadatas=metadatas
            )
        except Exception as e:
            # Fall back to the all-at-once path if batched embedding is unsupported.
            self._emit(progress_cb, f"Batched embedding fell back ({e}); building in one pass…")
            self.vectorstore = FAISS.from_documents(splits, self.embeddings)

        os.makedirs(self.rag_index_path, exist_ok=True)
        self.vectorstore.save_local(self.rag_index_path)
        self._emit(
            progress_cb,
            f"Saved index ({len(splits)} chunks) in {time.time() - build_start:.1f}s "
            f"to {self.rag_index_path}",
        )
        return len(splits)

    async def build_index_from_context_files(
        self,
        context_files: List[Dict[str, Any]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        progress_cb: Optional[Any] = None,
    ) -> int:
        """
        Build FAISS index from context files (PDFs and text files)

        Args:
            context_files: List of context file dictionaries with 'path', 'name', 'content'
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            progress_cb: Optional callable(str) invoked with human-readable progress lines

        Returns:
            Number of chunks indexed
        """
        self._emit(progress_cb, f"Building RAG index from {len(context_files)} context file(s)…")
        docs = self._load_documents(context_files, progress_cb=progress_cb)
        if not docs:
            self._emit(progress_cb, "No documents processed successfully.")
            return 0
        return self._split_and_build(docs, chunk_size, chunk_overlap, progress_cb=progress_cb)

    async def load_index(self) -> bool:
        """
        Load existing FAISS index

        Returns:
            True if loaded successfully
        """
        try:
            if not os.path.exists(self.rag_index_path):
                logger.warning(f"RAG index path not found: {self.rag_index_path}")
                return False

            self.vectorstore = FAISS.load_local(
                self.rag_index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(f"RAG index loaded from {self.rag_index_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading RAG index: {e}")
            return False

    @staticmethod
    def _normalize_rag_filter_token(value: str) -> str:
        """Normalize a source-filter token for tolerant matching."""
        token = str(value or "").strip().lower()
        if not token:
            return ""
        token = token.replace("\\", "/").replace(" ", "_")
        return token

    def _normalize_rag_filter_list(self, values: Optional[List[str]]) -> List[str]:
        """Normalize and deduplicate source filter values."""
        out: List[str] = []
        seen = set()
        for raw in values or []:
            token = self._normalize_rag_filter_token(str(raw or ""))
            if not token or token in seen:
                continue
            out.append(token)
            seen.add(token)
        return out

    def _doc_matches_source_filters(
        self,
        doc: Any,
        source_types: List[str],
        source_filters: List[str],
    ) -> bool:
        """Return True when a retrieved document matches source-type/source-name filters."""
        metadata = getattr(doc, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        source_type = str(metadata.get("source_type") or "").strip().lower()
        if source_types and source_type not in source_types:
            return False
        if not source_filters:
            return True
        source = str(metadata.get("source") or "").strip()
        filename = str(metadata.get("filename") or "").strip()
        candidates = [
            source,
            filename,
            Path(source).name if source else "",
            Path(source).stem if source else "",
            Path(filename).name if filename else "",
            Path(filename).stem if filename else "",
        ]
        candidate_tokens: List[str] = []
        for raw in candidates:
            token = self._normalize_rag_filter_token(raw)
            if token:
                candidate_tokens.append(token)
        for needle in source_filters:
            if any(
                needle == token
                or needle in token
                or token.endswith("/" + needle)
                for token in candidate_tokens
            ):
                return True
        return False

    def _answer_from_docs(self, question: str, docs: List[Any]) -> str:
        """Generate an answer from an explicit list of retrieved docs."""
        context_budget = max(1200, int(self.max_context_length or 4000))
        context_parts: List[str] = []
        used = 0
        for doc in docs:
            text = str(getattr(doc, "page_content", "") or "").strip()
            if not text:
                continue
            remaining = context_budget - used
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = text[:remaining]
            context_parts.append(text)
            used += len(text) + 2
        context_blob = "\n\n".join(context_parts).strip()
        if not context_blob:
            return "No relevant documents in the knowledge base."
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You answer questions using ONLY the context below. "
                    "If the context is not relevant, reply briefly: "
                    "'No relevant documents in the knowledge base.' "
                    "Context:\n{context}",
                ),
                ("human", "{input}"),
            ]
        )
        messages = prompt.format_messages(context=context_blob, input=question)
        answer = self.llm.invoke(messages)
        if hasattr(answer, "content"):
            return str(answer.content or "").strip()
        return str(answer or "").strip()

    def _query_rag_with_source_filters(
        self,
        question: str,
        k: int,
        return_sources: bool,
        source_types: List[str],
        source_filters: List[str],
    ) -> Dict[str, Any]:
        """Run filtered retrieval by source metadata then answer from matched docs."""
        candidate_k = max(24, int(k or 4) * 12)
        candidate_k = min(candidate_k, 256)
        retrieved_docs = self.vectorstore.similarity_search(question, k=candidate_k)
        matched_docs = [
            doc
            for doc in retrieved_docs
            if self._doc_matches_source_filters(
                doc,
                source_types=source_types,
                source_filters=source_filters,
            )
        ]
        if not matched_docs:
            response: Dict[str, Any] = {
                "answer": "No relevant documents in the knowledge base.",
                "sources": [],
            }
            if source_types:
                response["source_types"] = list(source_types)
            if source_filters:
                response["source_filters"] = list(source_filters)
            return response

        selected_docs = matched_docs[: max(1, int(k or 4))]
        answer = self._answer_from_docs(question, selected_docs)
        response = {"answer": answer}
        if return_sources:
            response["sources"] = [
                {
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(getattr(doc, "page_content", "")) > 500
                        else getattr(doc, "page_content", "")
                    ),
                    "metadata": getattr(doc, "metadata", {}),
                }
                for doc in selected_docs
            ]
        if source_types:
            response["source_types"] = list(source_types)
        if source_filters:
            response["source_filters"] = list(source_filters)
        return response

    async def query_rag(
        self,
        question: str,
        k: int = 4,
        return_sources: bool = False,
        source_types: Optional[List[str]] = None,
        source_filters: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Query RAG system

        Args:
            question: User question
            k: Number of documents to retrieve
            return_sources: Include source documents in response
            source_types: Optional source-type filters (for example: file, gallery)
            source_filters: Optional file/source-name filters
            top_k: Compatibility alias for k

        Returns:
            Dict with answer and optionally sources
        """
        if not self.is_initialized:
            await self.initialize()

        if self.vectorstore is None:
            return {
                "answer": "No RAG index available. Please add context files first.",
                "error": True,
                "sources": []
            }

        try:
            if top_k is not None:
                try:
                    k = int(top_k)
                except Exception:
                    k = k
            k = max(1, int(k or 4))
            normalized_source_types = self._normalize_rag_filter_list(source_types)
            normalized_source_filters = self._normalize_rag_filter_list(source_filters)
            if normalized_source_types or normalized_source_filters:
                return self._query_rag_with_source_filters(
                    question=question,
                    k=k,
                    return_sources=return_sources,
                    source_types=normalized_source_types,
                    source_filters=normalized_source_filters,
                )

            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

            # Use modern LCEL approach if available, otherwise fall back to RetrievalQA
            if LCEL_AVAILABLE:
                # Check if we have helper functions (langchain < 1.0) or need manual LCEL (langchain 1.x)
                if create_retrieval_chain is not None and create_stuff_documents_chain is not None:
                    # Modern LCEL approach with helper functions (langchain 0.2+)
                    system_prompt = (
                        "You answer questions using ONLY the context below. Do not refuse, ask to rephrase, or explain what RAG is. "
                        "If the context contains relevant information, summarize or quote it clearly. "
                        "If the context is empty or does not contain anything relevant, reply briefly: 'No relevant documents in the knowledge base.' "
                        "Context:\n{context}"
                    )
                    lcel_prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    # Create document chain
                    question_answer_chain = create_stuff_documents_chain(self.llm, lcel_prompt)
                    
                    # Create retrieval chain
                    chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    # Invoke chain
                    result = chain.invoke({"input": question})
                    
                    # Extract answer (LCEL returns "answer" key)
                    answer = result.get("answer", result.get("result", ""))
                    response = {"answer": answer}
                    
                    # Extract source documents if requested
                    if return_sources:
                        sources = []
                        if "context" in result:
                            context_docs = result.get("context", [])
                            if isinstance(context_docs, list):
                                sources = [
                                    {
                                        "content": (doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content) if hasattr(doc, "page_content") else str(doc)[:500],
                                        "metadata": getattr(doc, "metadata", {}),
                                    }
                                    for doc in context_docs
                                ]
                        response["sources"] = sources
                    
                    return response
                else:
                    # Manual LCEL approach (langchain 1.x - no chains module)
                    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
                    from langchain_core.output_parsers import StrOutputParser
                    
                    # Format documents function
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    # Create prompt template
                    system_prompt = (
                        "You answer questions using ONLY the context below. Do not refuse, ask to rephrase, or explain what RAG is. "
                        "If the context contains relevant information, summarize or quote it clearly. "
                        "If the context is empty or does not contain anything relevant, reply briefly: 'No relevant documents in the knowledge base.' "
                        "Context:\n{context}"
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    # Build manual LCEL chain (langchain 1.x compatible)
                    # Format: retrieve docs -> format -> prompt -> llm -> parse
                    def rag_chain(question: str):
                        # Retrieve documents
                        docs = retriever.invoke(question)
                        # Format documents
                        context = format_docs(docs)
                        # Create prompt messages with context and question
                        messages = prompt.format_messages(context=context, input=question)
                        # Get answer from LLM
                        answer = self.llm.invoke(messages)
                        # Parse output
                        if hasattr(answer, "content"):
                            return answer.content
                        return str(answer)
                    
                    # Invoke chain
                    answer = rag_chain(question)
                    response = {"answer": answer}
                    
                    # Extract source documents if requested
                    if return_sources:
                        docs = retriever.invoke(question)
                        response["sources"] = [
                            {
                                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                                "metadata": getattr(doc, "metadata", {}),
                            }
                            for doc in docs
                        ]
                    
                    return response
            
            elif RetrievalQA is not None:
                # Legacy RetrievalQA approach (for older langchain versions)
                qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=retriever,
                    chain_type_kwargs={"prompt": self.prompt},
                    return_source_documents=return_sources,
                )

                result = qa_chain.invoke({"query": question})

                response = {"answer": result["result"]}

                if return_sources and "source_documents" in result:
                    response["sources"] = [
                        {
                            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                            "metadata": doc.metadata,
                        }
                        for doc in result["source_documents"]
                    ]

                return response
            else:
                return {
                    "answer": "RAG chain not available. Please install compatible langchain version: pip install langchain langchain-community",
                    "error": True,
                    "sources": []
                }

        except Exception as e:
            logger.error(f"Error querying RAG: {e}")
            return {"answer": f"RAG query error: {str(e)}", "error": True, "sources": []}

    async def add_context_files_to_index(
        self,
        context_files: List[Dict[str, Any]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        progress_cb: Optional[Any] = None,
    ) -> int:
        """
        Add more context files to existing index

        Args:
            context_files: List of context file dictionaries
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            progress_cb: Optional callable(str) invoked with human-readable progress lines

        Returns:
            Number of new chunks added
        """
        if not self.is_initialized:
            await self.initialize()

        if self.vectorstore is None:
            return await self.build_index_from_context_files(
                context_files, chunk_size, chunk_overlap, progress_cb=progress_cb
            )

        # Process new files (reuse the shared loader for per-file progress).
        self._emit(progress_cb, f"Adding {len(context_files)} file(s) to existing index…")
        docs = self._load_documents(context_files, progress_cb=progress_cb)

        if not docs:
            self._emit(progress_cb, "No new documents to add.")
            return 0

        # Chunk and add to existing index
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        splits = text_splitter.split_documents(docs)
        if not splits:
            self._emit(progress_cb, "No chunks produced from documents.")
            return 0

        self._emit(
            progress_cb,
            f"Embedding {len(splits)} new chunk(s) via {self.embedding_backend}:{self.embedding_model}…",
        )
        add_start = time.time()
        try:
            texts = [d.page_content for d in splits]
            metadatas = [d.metadata for d in splits]
            vectors = self._embed_in_batches(texts, progress_cb=progress_cb)
            self.vectorstore.add_embeddings(list(zip(texts, vectors)), metadatas=metadatas)
        except Exception as e:
            self._emit(progress_cb, f"Batched add fell back ({e}); adding in one pass…")
            self.vectorstore.add_documents(splits)
        self.vectorstore.save_local(self.rag_index_path)
        self._emit(
            progress_cb,
            f"Added {len(splits)} chunk(s) in {time.time() - add_start:.1f}s and saved index.",
        )

        return len(splits)

    async def clear_index(self) -> bool:
        """
        Clear the RAG index

        Returns:
            True if cleared successfully
        """
        try:
            if os.path.exists(self.rag_index_path):
                import shutil
                shutil.rmtree(self.rag_index_path)
                logger.info("RAG index cleared")
            
            self.vectorstore = None
            return True
        except Exception as e:
            logger.error(f"Error clearing RAG index: {e}")
            return False

    def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current RAG index

        Returns:
            Dict with index information
        """
        info = {
            "initialized": self.is_initialized,
            "has_index": self.vectorstore is not None,
            "index_path": self.rag_index_path,
            "index_exists": os.path.exists(self.rag_index_path),
            "device": self.device,
            "model_id": self.model_id,
            "embedding_model": self.embedding_model,
            "embedding_backend": self.embedding_backend,
            "ollama_base_url": self.ollama_base_url,
        }

        if self.vectorstore:
            try:
                info["index_size"] = self.vectorstore.index.ntotal
            except:
                info["index_size"] = "unknown"

        return info


# ---------------------------------------------------------------------------
# Namespaced RAG registry
#
# The two RAG "structures" share ONE engine (model + embeddings + Ollama URL,
# held in ``_shared_rag_config``) but keep SEPARATE FAISS indexes, one per
# namespace. ``DEFAULT_NAMESPACE`` is the per-project chat RAG; ``NOTES_NAMESPACE``
# is the global, vault-wide notes RAG used as a secondary knowledge base.
#
# All public functions take an optional ``namespace`` defaulting to
# ``DEFAULT_NAMESPACE``, so every existing call site keeps its original behaviour.
# ---------------------------------------------------------------------------

DEFAULT_NAMESPACE = "default"
NOTES_NAMESPACE = "notes"

# Config keys that describe the shared engine (everything except the index path).
# These are pooled in ``_shared_rag_config`` so both namespaces use one engine.
_ENGINE_CONFIG_KEYS = frozenset({
    "lms_base_url",
    "lms_api_key",
    "model_id",
    "embedding_model",
    "embedding_backend",
    "ollama_base_url",
    "device",
    "max_context_length",
})

# One SoloRAGSystem instance per namespace.
_rag_systems: Dict[str, "SoloRAGSystem"] = {}
# Per-namespace overrides (in practice just ``rag_index_path``).
_rag_configs: Dict[str, Dict[str, Any]] = {}
# Engine config shared by every namespace.
_shared_rag_config: Dict[str, Any] = {}

# Backward-compatible module global: mirrors the default-namespace instance so
# any legacy reader of ``rag_system`` keeps working.
rag_system: Optional["SoloRAGSystem"] = None
# Legacy alias kept for callers that still read ``_rag_config`` directly; it now
# points at the merged effective config of the default namespace.
_rag_config: Dict[str, Any] = {}


def _effective_config(namespace: str) -> Dict[str, Any]:
    """Merge the shared engine config with this namespace's overrides."""
    merged = dict(_shared_rag_config)
    merged.update(_rag_configs.get(namespace, {}))
    return merged


def set_rag_config(
    lms_base_url: Optional[str] = None,
    lms_api_key: Optional[str] = None,
    model_id: Optional[str] = None,
    embedding_model: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    device: Optional[str] = None,
    rag_index_path: Optional[str] = None,
    max_context_length: Optional[int] = None,
    namespace: str = DEFAULT_NAMESPACE,
) -> None:
    """
    Set config used when creating a RAG system instance.

    Engine keys (model, embeddings, Ollama URL, device, context length) are
    written to the SHARED config so every namespace uses one engine. Only
    ``rag_index_path`` is stored per-namespace, giving each namespace its own
    on-disk index. Call before get_rag_system()/initialize_rag_system().
    """
    global _rag_config
    ns_overrides = _rag_configs.setdefault(namespace, {})

    def _put(key: str, value: Any) -> None:
        if value is None:
            return
        if key in _ENGINE_CONFIG_KEYS:
            _shared_rag_config[key] = value
        else:
            ns_overrides[key] = value

    _put("lms_base_url", lms_base_url)
    _put("lms_api_key", lms_api_key)
    _put("model_id", model_id)
    _put("embedding_model", embedding_model)
    _put("embedding_backend", embedding_backend)
    _put("ollama_base_url", ollama_base_url)
    _put("device", device)
    _put("rag_index_path", rag_index_path)
    _put("max_context_length", max_context_length)

    # Keep the legacy alias in sync with the default namespace.
    _rag_config = _effective_config(DEFAULT_NAMESPACE)


def reset_rag_system(namespace: str = DEFAULT_NAMESPACE) -> None:
    """
    Drop the cached RAG instance for ``namespace`` so the next get_rag_system()
    rebuilds it with the current config (e.g. after changing embedding model).
    Other namespaces are left untouched.
    """
    global rag_system
    _rag_systems.pop(namespace, None)
    if namespace == DEFAULT_NAMESPACE:
        rag_system = None


def reset_all_rag_systems() -> None:
    """Drop every cached RAG instance across all namespaces."""
    global rag_system
    _rag_systems.clear()
    rag_system = None


async def get_rag_system(namespace: str = DEFAULT_NAMESPACE) -> Optional[SoloRAGSystem]:
    """Get or create the RAG system instance for ``namespace``."""
    global rag_system
    if namespace not in _rag_systems:
        try:
            inst = SoloRAGSystem(**_effective_config(namespace))
            await inst.initialize()
            _rag_systems[namespace] = inst
            print(f"[RAG] RAG system instance created (namespace={namespace!r})")
        except Exception as e:
            print(f"[RAG] Error creating RAG system (namespace={namespace!r}): {e}")
            return None
    inst = _rag_systems.get(namespace)
    if namespace == DEFAULT_NAMESPACE:
        rag_system = inst
    return inst


async def initialize_rag_system(namespace: str = DEFAULT_NAMESPACE) -> bool:
    """
    Initialize the RAG system for ``namespace``.
    Returns True if successful, False otherwise.
    """
    try:
        global rag_system
        if namespace not in _rag_systems:
            _rag_systems[namespace] = SoloRAGSystem(**_effective_config(namespace))
        inst = _rag_systems[namespace]
        if namespace == DEFAULT_NAMESPACE:
            rag_system = inst
        success = await inst.initialize()
        if success:
            print(f"[RAG] Solo RAG system initialized successfully (namespace={namespace!r})")
        else:
            print(f"[RAG] RAG system initialization failed (namespace={namespace!r})")
        return success
    except Exception as e:
        print(f"[RAG] Error initializing RAG system (namespace={namespace!r}): {e}")
        return False


def get_rag_status(namespace: str = DEFAULT_NAMESPACE) -> Dict[str, Any]:
    """Return RAG system status dict for ``namespace``."""
    if not RAG_DEPENDENCIES_AVAILABLE:
        return {
            "available": False,
            "error": "RAG dependencies not installed",
            "dependencies_installed": False
        }

    inst = _rag_systems.get(namespace)
    if inst is None:
        return {
            "available": False,
            "error": "RAG system not initialized",
            "dependencies_installed": True,
            "namespace": namespace,
        }

    try:
        return {
            "available": True,
            "dependencies_installed": True,
            "namespace": namespace,
            "initialized": inst.is_initialized,
            "has_index": inst.vectorstore is not None,
            "device": inst.device,
            "model_id": inst.model_id,
            "embedding_model": inst.embedding_model,
            "embedding_backend": inst.embedding_backend,
            "ollama_base_url": inst.ollama_base_url,
            "index_path": inst.rag_index_path
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "dependencies_installed": True,
            "namespace": namespace,
        }
