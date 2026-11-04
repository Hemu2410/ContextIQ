"""
ContextIQ: Enterprise Documentation Q&A Assistant.
High-precision, zero-hallucination grounded retrieval and answering engine.
"""

from .models import Document, DocumentChunk, Citation, ContextIQResponse
from .chunker import HierarchicalChunker
from .index import BM25Index
from .threshold import AdaptiveConfidenceEvaluator, ConfidenceEvaluation
from .extractor import GroundedExtractor, ExtractedAnswer
from .engine import ContextIQEngine

__all__ = [
    "Document",
    "DocumentChunk",
    "Citation",
    "ContextIQResponse",
    "HierarchicalChunker",
    "BM25Index",
    "AdaptiveConfidenceEvaluator",
    "ConfidenceEvaluation",
    "GroundedExtractor",
    "ExtractedAnswer",
    "ContextIQEngine",
]
