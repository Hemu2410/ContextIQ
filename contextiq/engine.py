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
from .graph import ContextIQStateGraph


class ContextIQEngine:
    """Production documentation Q&A engine."""

    def __init__(self, docs_path: Optional[str] = None):
        self.chunker = HierarchicalChunker()
        self.index = BM25Index(title_weight=2.5)
        self.evaluator = AdaptiveConfidenceEvaluator(self.index)
        self.extractor = GroundedExtractor()
        self.graph = ContextIQStateGraph(self.index, self.evaluator, self.extractor)

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

        # Refresh evaluator and graph bindings with updated vocabulary
        self.evaluator = AdaptiveConfidenceEvaluator(self.index)
        self.graph = ContextIQStateGraph(self.index, self.evaluator, self.extractor)

        return len(self.chunks)

    def ingest_document(self, doc: Document) -> int:
        """Ingests a single new document or updates an existing document on demand."""
        # Remove existing document with same doc_id if present (upsert)
        self.documents = [d for d in self.documents if d.doc_id != doc.doc_id]
        self.documents.append(doc)

        # Re-chunk and re-index
        self.chunks = self.chunker.chunk_all(self.documents)
        self.index.build(self.chunks)

        self.evaluator = AdaptiveConfidenceEvaluator(self.index)
        self.graph = ContextIQStateGraph(self.index, self.evaluator, self.extractor)

        return len(self.chunks)

    def ask(self, question: str) -> ContextIQResponse:
        """Processes an administrative question through the state graph."""
        return self.graph.run(question)

    def get_stats(self) -> Dict[str, Any]:
        """Returns runtime statistics of the active index."""
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.chunks),
            "vocabulary_size": len(self.index.corpus_raw_vocab),
            "avg_chunk_length": round(self.index.avg_doc_len, 2),
        }
