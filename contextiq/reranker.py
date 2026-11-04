"""
Semantic Reranker for ContextIQ.

Fuses multi-channel retrieval signals (Lexical BM25 with Title Boosting
and Dense Semantic Cosine Similarity from Qdrant) using calibrated
hybrid rank fusion and intent-corroborated relevance scoring.
"""
from typing import List, Dict, Any, Optional, Tuple
from .models import DocumentChunk, QueryAnalysis, IntentType
from .vector_store import VectorSearchResult


class SemanticReranker:
    """
    Cross-channel semantic reranker that combines lexical BM25 precision
    with dense vector semantic similarity from Qdrant.
    """

    def __init__(
        self,
        weight_bm25: float = 0.75,
        weight_semantic: float = 0.25,
    ):
        self.weight_bm25 = weight_bm25
        self.weight_semantic = weight_semantic

    def rerank(
        self,
        query: str,
        bm25_results: List[Tuple[float, DocumentChunk, Dict[str, Any]]],
        vector_results: List[VectorSearchResult],
        query_analysis: Optional[QueryAnalysis] = None,
        top_k: int = 5,
    ) -> List[Tuple[float, DocumentChunk, Dict[str, Any]]]:
        """
        Fuses lexical BM25 and Qdrant semantic vector results.
        Returns a sorted list of (reranked_score, chunk, debug_meta).
        """
        if not bm25_results and not vector_results:
            return []

        # Index candidates by chunk_id
        candidates: Dict[str, Dict[str, Any]] = {}

        # 1. Ingest BM25 results
        for rank, (score, chunk, meta) in enumerate(bm25_results):
            candidates[chunk.chunk_id] = {
                "chunk": chunk,
                "bm25_score": score,
                "bm25_rank": rank + 1,
                "bm25_meta": meta,
                "vector_score": 0.0,
                "vector_rank": None,
            }

        # 2. Ingest Qdrant Vector results
        for rank, v_res in enumerate(vector_results):
            cid = v_res.chunk.chunk_id
            if cid in candidates:
                candidates[cid]["vector_score"] = v_res.score
                candidates[cid]["vector_rank"] = rank + 1
            else:
                candidates[cid] = {
                    "chunk": v_res.chunk,
                    "bm25_score": 0.0,
                    "bm25_rank": None,
                    "bm25_meta": {},
                    "vector_score": v_res.score,
                    "vector_rank": rank + 1,
                }

        reranked: List[Tuple[float, DocumentChunk, Dict[str, Any]]] = []

        for cid, data in candidates.items():
            chunk = data["chunk"]
            s_bm25 = data["bm25_score"]
            s_vec = data["vector_score"]

            # Hybrid score computation
            if s_bm25 > 0 and s_vec > 0:
                hybrid_score = (self.weight_bm25 * s_bm25) + (self.weight_semantic * s_vec)
            elif s_bm25 > 0:
                hybrid_score = s_bm25
            else:
                hybrid_score = s_vec

            # Corroboration bonus: boost if title keywords overlap with query terms
            q_lower = query.lower()
            title_terms = [t for t in chunk.title.lower().split() if len(t) > 3]
            title_match_count = sum(1 for t in title_terms if t in q_lower)
            if title_match_count > 0:
                title_boost = min(0.03 * title_match_count, 0.06)
                hybrid_score = min(1.0, hybrid_score + title_boost)

            meta = {
                "reranked": True,
                "hybrid_score": round(hybrid_score, 4),
                "bm25_score": round(s_bm25, 4),
                "bm25_rank": data["bm25_rank"],
                "vector_score": round(s_vec, 4),
                "vector_rank": data["vector_rank"],
                "channel_count": (1 if s_bm25 > 0 else 0) + (1 if s_vec > 0 else 0),
            }
            if data["bm25_meta"]:
                meta.update({k: v for k, v in data["bm25_meta"].items() if k not in meta})

            reranked.append((round(hybrid_score, 4), chunk, meta))

        # Sort descending by hybrid score
        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k]
