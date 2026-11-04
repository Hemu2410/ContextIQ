"""
ContextIQ REST API — FastAPI Microservice for Documentation Q&A.

Endpoints:
    POST /api/v1/ask         — Answer an admin question with grounded citations
    POST /api/v1/ingest      — Ingest a single document into the live corpus
    POST /api/v1/reload      — Reload documentation corpus from disk
    GET  /api/v1/evaluate    — Run benchmark evaluation scorecard
    GET  /api/v1/vector-search — Direct vector similarity search against Qdrant
    GET  /health             — Health check with runtime statistics

Interactive API documentation available at /docs (Swagger UI).
"""
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from contextiq.engine import ContextIQEngine
from contextiq.models import Document

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DOCS_PATH = os.path.join("starter_kit", "data", "docs.json")
DEFAULT_QUESTIONS_PATH = os.path.join("starter_kit", "data", "example_questions.json")

QDRANT_URL = os.environ.get("QDRANT_URL")  # None → in-memory mode

# ---------------------------------------------------------------------------
# Engine Singleton
# ---------------------------------------------------------------------------
engine = ContextIQEngine(docs_path=DEFAULT_DOCS_PATH, qdrant_url=QDRANT_URL)

# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ContextIQ Documentation Q&A API",
    description=(
        "Enterprise-grade, zero-hallucination documentation Q&A microservice "
        "for Solstice Learning. Powered by Okapi BM25 with 2.5× Title Boosting, "
        "Qdrant Vector Database, and a LangGraph-compatible state machine with "
        "adaptive confidence gating."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for web portal widget embedding
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic Request / Response Models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Submit an admin question to ContextIQ."""
    question: str = Field(..., min_length=1, examples=["How do I reset my password?"])
    session_id: Optional[str] = Field(None, description="Optional conversational session ID for multi-turn disambiguation")


class IngestRequest(BaseModel):
    """Ingest a single document into the live corpus."""
    doc_id: str = Field(..., examples=["D-07"])
    title: str = Field(..., examples=["Salesforce Integration Guide"])
    content: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = None


class ReloadRequest(BaseModel):
    """Reload the documentation corpus from disk."""
    docs_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Returns service health status and runtime statistics."""
    return {
        "status": "healthy",
        "stats": engine.get_stats(),
    }


@app.post("/api/v1/ask", tags=["Q&A"])
def ask_question(request: AskRequest) -> Dict[str, Any]:
    """
    Answer an administrative question using the ContextIQ RAG pipeline.

    Returns grounded answers with 3-tier citations (doc_id, title, verbatim
    quote, and UI navigation path), or a safe refusal when documentation
    does not cover the topic.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    response = engine.ask(request.question, session_id=request.session_id)
    return response.to_dict()


@app.post("/api/v1/ingest", tags=["Corpus Management"])
def ingest_document(request: IngestRequest) -> Dict[str, Any]:
    """
    Ingest a single document (or upsert an existing one by doc_id) into the
    live corpus. The BM25 index and Qdrant vectors are rebuilt atomically.
    """
    doc = Document(
        doc_id=request.doc_id,
        title=request.title,
        content=request.content,
        metadata=request.metadata or {},
    )
    total_chunks = engine.ingest_document(doc)
    return {
        "status": "ingested",
        "doc_id": request.doc_id,
        "title": request.title,
        "total_chunks": total_chunks,
    }


@app.post("/api/v1/reload", tags=["Corpus Management"])
def reload_corpus(request: ReloadRequest) -> Dict[str, Any]:
    """
    Reload the documentation corpus from disk. Optionally specify a custom
    docs_path; defaults to starter_kit/data/docs.json.
    """
    path = request.docs_path or DEFAULT_DOCS_PATH
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Documentation file not found: {path}")

    chunk_count = engine.load_docs(path)
    return {
        "status": "reloaded",
        "docs_path": path,
        "chunk_count": chunk_count,
    }


@app.get("/api/v1/evaluate", tags=["Benchmark"])
def run_evaluation() -> Dict[str, Any]:
    """
    Execute all benchmark questions from example_questions.json and return
    a JSON scorecard with accuracy, hallucination rate, and per-question
    breakdown.
    """
    if not os.path.exists(DEFAULT_QUESTIONS_PATH):
        raise HTTPException(status_code=404, detail="Evaluation file not found.")

    with open(DEFAULT_QUESTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("example_questions", [])
    total = len(questions)
    passed = 0
    hallucinations = 0
    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(questions, 1):
        q_id = item.get("question_id", f"Q-{idx}")
        q_text = item.get("question", "")
        expected_doc_id = item.get("expected_doc_id")

        resp = engine.ask(q_text)

        if expected_doc_id is None:
            is_correct = resp.status == "REFUSED"
            if not is_correct:
                hallucinations += 1
        else:
            is_correct = (
                resp.status == "ANSWERED"
                and resp.citation is not None
                and resp.citation.doc_id == expected_doc_id
            )

        if is_correct:
            passed += 1

        results.append({
            "question_id": q_id,
            "question": q_text,
            "expected_doc_id": expected_doc_id,
            "actual_doc_id": resp.citation.doc_id if resp.citation else None,
            "status": resp.status if isinstance(resp.status, str) else resp.status.value,
            "confidence_score": round(resp.confidence_score, 4),
            "verdict": "PASS" if is_correct else "FAIL",
        })

    accuracy = (passed / total) * 100 if total > 0 else 0
    hallucination_rate = (hallucinations / total) * 100 if total > 0 else 0

    return {
        "total_questions": total,
        "passed": passed,
        "accuracy": f"{accuracy:.1f}%",
        "hallucination_rate": f"{hallucination_rate:.1f}%",
        "results": results,
    }


@app.get("/api/v1/vector-search", tags=["Vector Database"])
def vector_search(
    query: str = Query(..., min_length=1, description="Search query text"),
    top_k: int = Query(3, ge=1, le=20, description="Number of results to return"),
    doc_id: Optional[str] = Query(None, description="Filter results by document ID"),
) -> Dict[str, Any]:
    """
    Direct vector similarity search against the Qdrant collection.
    Bypasses BM25 to show raw cosine similarity scores from the vector
    database layer.
    """
    results = engine.vector_search(query, top_k=top_k, doc_id_filter=doc_id)
    return {
        "query": query,
        "top_k": top_k,
        "doc_id_filter": doc_id,
        "results": [
            {
                "score": r.score,
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.doc_id,
                "title": r.chunk.title,
                "text_preview": r.chunk.text[:200] + "..." if len(r.chunk.text) > 200 else r.chunk.text,
            }
            for r in results
        ],
    }
