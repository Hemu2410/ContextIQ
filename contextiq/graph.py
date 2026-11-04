"""
StateGraph Orchestrator for ContextIQ.

Models the question-answering workflow as a directed acyclic state graph:
START -> analyze_query -> retrieve_candidates -> [confidence_gate]
         ├── (Pass)   -> extract_grounded_answer -> END
         └── (Refuse) -> generate_refusal        -> END

Compatible with LangGraph StateGraph, while providing a self-contained
deterministic execution fallback for environments without external packages.
"""
from typing import Dict, Any, List, Optional, TypedDict

from .models import DocumentChunk, Citation, ContextIQResponse
from .index import BM25Index
from .threshold import AdaptiveConfidenceEvaluator, ConfidenceEvaluation
from .extractor import GroundedExtractor, ExtractedAnswer

STANDARD_REFUSAL_MESSAGE = "I don't have an answer for that in the Solstice Learning documentation."
SUPPORT_ESCALATION_NOTE = "If you need immediate assistance with this topic, please contact support@solsticelearning.com."


class AssistantState(TypedDict):
    question: str
    query_valid: bool
    top_score: float
    top_chunk: Optional[DocumentChunk]
    debug_meta: Dict[str, Any]
    evaluation: Optional[ConfidenceEvaluation]
    status: str
    answer: str
    citation: Optional[Citation]
    rejection_reason: Optional[str]
    escalation_path: Optional[str]


class ContextIQStateGraph:
    """Directed state graph coordinating the analysis, retrieval, gating, and extraction nodes."""

    def __init__(self, index: BM25Index, evaluator: AdaptiveConfidenceEvaluator, extractor: GroundedExtractor):
        self.index = index
        self.evaluator = evaluator
        self.extractor = extractor

    def node_analyze_query(self, state: AssistantState) -> AssistantState:
        """Node 1: Validates query input."""
        q = state["question"].strip() if state["question"] else ""
        state["query_valid"] = bool(q)
        return state

    def node_retrieve(self, state: AssistantState) -> AssistantState:
        """Node 2: Retrieves top matching candidate from the BM25 index."""
        if not state["query_valid"]:
            state["top_score"] = 0.0
            state["top_chunk"] = None
            return state

        results = self.index.search(state["question"], top_k=1)
        if results:
            score, chunk, meta = results[0]
            state["top_score"] = score
            state["top_chunk"] = chunk
            state["debug_meta"] = meta
        else:
            state["top_score"] = 0.0
            state["top_chunk"] = None
        return state

    def edge_confidence_gate(self, state: AssistantState) -> str:
        """Conditional Edge: Decides whether to route to answer extraction or refusal."""
        if not state["query_valid"]:
            return "route_refusal"

        if state["top_chunk"] is None or state["top_score"] <= 0.0:
            return "route_refusal"

        # Evaluate against the multi-signal rejection boundary
        evaluation = self.evaluator.evaluate(state["question"], state["top_score"], state["top_chunk"])
        state["evaluation"] = evaluation

        if evaluation.decision == "ACCEPT":
            return "route_extract"
        return "route_refusal"

    def node_extract_answer(self, state: AssistantState) -> AssistantState:
        """Node 3A: Extracts grounded answer and formats 3-tier citation."""
        extracted: ExtractedAnswer = self.extractor.extract(state["question"], state["top_chunk"])
        state["status"] = "ANSWERED"
        state["answer"] = extracted.direct_answer
        state["citation"] = extracted.citation
        state["rejection_reason"] = None
        return state

    def node_generate_refusal(self, state: AssistantState) -> AssistantState:
        """Node 3B: Generates standardized refusal and appends support escalation."""
        state["status"] = "REFUSED"
        state["answer"] = f"{STANDARD_REFUSAL_MESSAGE}\n\n{SUPPORT_ESCALATION_NOTE}"
        state["citation"] = None
        state["escalation_path"] = "support@solsticelearning.com"

        if not state["query_valid"]:
            state["rejection_reason"] = "Empty or whitespace-only query"
        elif state.get("evaluation") and state["evaluation"].rejection_reason:
            state["rejection_reason"] = state["evaluation"].rejection_reason
        else:
            state["rejection_reason"] = "No matching documentation found"
        return state

    def run(self, question: str) -> ContextIQResponse:
        """Executes the full graph pipeline for an incoming inquiry."""
        state: AssistantState = {
            "question": question,
            "query_valid": False,
            "top_score": 0.0,
            "top_chunk": None,
            "debug_meta": {},
            "evaluation": None,
            "status": "REFUSED",
            "answer": "",
            "citation": None,
            "rejection_reason": None,
            "escalation_path": None,
        }

        # Step 1: Query analysis
        state = self.node_analyze_query(state)

        # Step 2: Retrieval
        state = self.node_retrieve(state)

        # Step 3: Conditional Routing
        decision = self.edge_confidence_gate(state)

        # Step 4: Terminal Node Execution
        if decision == "route_extract":
            state = self.node_extract_answer(state)
        else:
            state = self.node_generate_refusal(state)

        # Return standardized response contract
        eval_obj = state.get("evaluation")
        tier = eval_obj.confidence_tier if eval_obj else "REFUSAL"
        threshold = eval_obj.effective_threshold if eval_obj else 0.40

        return ContextIQResponse(
            status=state["status"],
            question=question,
            answer=state["answer"],
            citation=state["citation"],
            confidence_tier=tier,
            confidence_score=round(state["top_score"], 3),
            effective_threshold=round(threshold, 3),
            rejection_reason=state.get("rejection_reason"),
            escalation_path=state.get("escalation_path"),
        )
