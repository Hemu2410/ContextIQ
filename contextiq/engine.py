"""
ContextIQ High-Level Engine Interface.

Unified entrypoint coordinating ingestion, state graph execution,
and on-demand document updates.
"""
import os
from typing import List, Dict, Any, Optional

from .models import Document, DocumentChunk, ContextIQResponse
from .chunker import HierarchicalChunker, load_documents_from_json
from .index import BM25Index
from .threshold import AdaptiveConfidenceEvaluator
from .extractor import GroundedExtractor
from .vector_store import QdrantVectorStore, VectorSearchResult
from .reranker import SemanticReranker
from .graph import ContextIQStateGraph
from .session import SessionManager


class ContextIQEngine:
    """Production documentation Q&A engine with Qdrant vector database integration."""

    def __init__(self, docs_path: Optional[str] = None, qdrant_url: Optional[str] = None):
        self.chunker = HierarchicalChunker()
        self.index = BM25Index(title_weight=2.5)
        self.vector_store = QdrantVectorStore(
            collection_name="solstice_docs",
            in_memory=(qdrant_url is None),
            url=qdrant_url,
        )
        self.reranker = SemanticReranker()
        self.evaluator = AdaptiveConfidenceEvaluator(self.index)
        self.extractor = GroundedExtractor()
        self.session_manager = SessionManager()
        self.graph = ContextIQStateGraph(
            self.index,
            self.evaluator,
            self.extractor,
            session_manager=self.session_manager,
            vector_store=self.vector_store,
            reranker=self.reranker,
        )

        self.documents: List[Document] = []
        self.chunks: List[DocumentChunk] = []

        if docs_path:
            self.load_docs(docs_path)

    def load_docs(self, docs_path: str) -> int:
        """Loads and indexes documentation from a JSON file."""
        if not os.path.exists(docs_path):
            raise FileNotFoundError(f"Documentation file not found at: {docs_path}")

        self.documents = load_documents_from_json(docs_path)
        self.chunks = self.chunker.chunk_all(self.documents)
        self.index.build(self.chunks)
        self.vector_store.index_chunks(self.chunks)

        # Refresh evaluator and graph bindings with updated vocabulary
        self.evaluator = AdaptiveConfidenceEvaluator(self.index)
        self.graph = ContextIQStateGraph(
            self.index,
            self.evaluator,
            self.extractor,
            session_manager=self.session_manager,
            vector_store=self.vector_store,
            reranker=self.reranker,
        )

        return len(self.chunks)

    def ingest_document(self, doc: Document) -> int:
        """Ingests a single new document or updates an existing document on demand."""
        # Remove existing document with same doc_id if present (upsert)
        self.documents = [d for d in self.documents if d.doc_id != doc.doc_id]
        self.documents.append(doc)

        # Re-chunk and re-index
        self.chunks = self.chunker.chunk_all(self.documents)
        self.index.build(self.chunks)
        self.vector_store.index_chunks(self.chunks)

        self.evaluator = AdaptiveConfidenceEvaluator(self.index)
        self.graph = ContextIQStateGraph(
            self.index,
            self.evaluator,
            self.extractor,
            session_manager=self.session_manager,
            vector_store=self.vector_store,
            reranker=self.reranker,
        )

        return len(self.chunks)

    def ask(self, question: str, session_id: Optional[str] = None) -> ContextIQResponse:
        """Processes an administrative question through the state graph with session memory."""
        return self.graph.run(question, session_id=session_id)

    def vector_search(self, query: str, top_k: int = 3, doc_id_filter: Optional[str] = None) -> List[VectorSearchResult]:
        """Direct vector similarity search against Qdrant collection."""
        return self.vector_store.search(query, top_k=top_k, doc_id_filter=doc_id_filter)

    def get_stats(self) -> Dict[str, Any]:
        """Returns runtime statistics of the active index and Qdrant store."""
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "vocabulary_size": len(self.index.corpus_raw_vocab),
            "avg_chunk_length": round(self.index.avg_doc_len, 2),
            "active_sessions": self.session_manager.active_session_count(),
            "vector_database": self.vector_store.get_collection_info(),
        }
