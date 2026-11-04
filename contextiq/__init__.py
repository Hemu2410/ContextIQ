"""
ContextIQ: Enterprise Documentation Q&A Assistant.
High-precision, zero-hallucination grounded retrieval and answering engine.
"""

from .models import (
    Document,
    DocumentChunk,
    Citation,
    ContextIQResponse,
    Status,
    ConfidenceTier,
    IntentType,
    QueryAnalysis,
    TelemetryEvent,
)
from .chunker import HierarchicalChunker, load_documents_from_json
from .index import BM25Index
from .threshold import AdaptiveConfidenceEvaluator, ConfidenceEvaluation
from .extractor import GroundedExtractor, ExtractedAnswer
from .vector_store import QdrantVectorStore, VectorSearchResult
from .graph import ContextIQStateGraph
from .engine import ContextIQEngine

__all__ = [
    "Document",
    "DocumentChunk",
    "Citation",
    "ContextIQResponse",
    "Status",
    "ConfidenceTier",
    "IntentType",
    "QueryAnalysis",
    "TelemetryEvent",
    "HierarchicalChunker",
    "load_documents_from_json",
    "BM25Index",
    "QdrantVectorStore",
    "VectorSearchResult",
    "AdaptiveConfidenceEvaluator",
    "ConfidenceEvaluation",
    "GroundedExtractor",
    "ExtractedAnswer",
    "ContextIQStateGraph",
    "ContextIQEngine",
]
