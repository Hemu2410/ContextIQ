"""
StateGraph Orchestrator for ContextIQ.

Models the question-answering workflow as a directed acyclic state graph:
START -> analyze_query -> retrieve_candidates -> evaluate_confidence -> [confidence_gate]
         ├── (Pass)   -> extract_grounded_answer -> END
         └── (Refuse) -> generate_refusal        -> END

Compatible with LangGraph StateGraph, while providing a self-contained
deterministic execution fallback for environments without external packages.
"""
import datetime
from typing import Dict, Any, List, Optional, TypedDict

from .models import (
    DocumentChunk,
    Citation,
    ContextIQResponse,
    QueryAnalysis,
    IntentType,
    Status,
    ConfidenceTier,
    TelemetryEvent,
)
from .index import BM25Index
from .threshold import AdaptiveConfidenceEvaluator, ConfidenceEvaluation
from .extractor import GroundedExtractor, ExtractedAnswer

STANDARD_REFUSAL_MESSAGE = "I don't have an answer for that in the Solstice Learning documentation."
SUPPORT_ESCALATION_NOTE = "If you need immediate assistance with this topic, please contact support@solsticelearning.com."


class AssistantState(TypedDict):
    question: str
    query_valid: bool
    intent: Optional[IntentType]
    query_analysis: Optional[QueryAnalysis]
    top_score: float
    top_chunk: Optional[DocumentChunk]
    debug_meta: Dict[str, Any]
    evaluation: Optional[ConfidenceEvaluation]
    status: str
    answer: str
    citation: Optional[Citation]
    rejection_reason: Optional[str]
    escalation_path: Optional[str]
    telemetry_log: List[TelemetryEvent]


class ContextIQStateGraph:
    """Directed state graph coordinating the analysis, retrieval, gating, and extraction nodes."""

    def __init__(self, index: BM25Index, evaluator: AdaptiveConfidenceEvaluator, extractor: GroundedExtractor):
        self.index = index
        self.evaluator = evaluator
        self.extractor = extractor

    # ------------------------------------------------------------------
    # Node 1 – Query analysis & intent classification
    # ------------------------------------------------------------------

    def node_analyze_query(self, state: AssistantState) -> AssistantState:
        """Validates query input and classifies intent via the evaluator."""
        q = state["question"].strip() if state["question"] else ""
        state["query_valid"] = bool(q)

        if state["query_valid"]:
            analysis = self.evaluator.classify_intent(q)
            state["query_analysis"] = analysis
            state["intent"] = analysis.intent
        else:
            state["query_analysis"] = None
            state["intent"] = None

        return state

    # ------------------------------------------------------------------
    # Node 2 – Retrieval (top_k=3)
    # ------------------------------------------------------------------

    def node_retrieve(self, state: AssistantState) -> AssistantState:
        """Retrieves top matching candidates from the BM25 index."""
        if not state["query_valid"]:
            state["top_score"] = 0.0
            state["top_chunk"] = None
            return state

        results = self.index.search(state["question"], top_k=3)
        if results:
            score, chunk, meta = results[0]
            state["top_score"] = score
            state["top_chunk"] = chunk
            state["debug_meta"] = meta
        else:
            state["top_score"] = 0.0
            state["top_chunk"] = None
        return state

    # ------------------------------------------------------------------
    # Node 3 – Confidence evaluation (moved OUT of the edge)
    # ------------------------------------------------------------------

    def node_evaluate_confidence(self, state: AssistantState) -> AssistantState:
        """Evaluates retrieval confidence using the multi-signal rejection boundary."""
        if not state["query_valid"] or state["top_chunk"] is None or state["top_score"] <= 0.0:
            state["evaluation"] = None
            return state

        evaluation = self.evaluator.evaluate(
            state["question"],
            state["top_score"],
            state["top_chunk"],
            query_analysis=state.get("query_analysis"),
        )
        state["evaluation"] = evaluation
        return state

    # ------------------------------------------------------------------
    # Conditional edge – pure routing, no mutation
    # ------------------------------------------------------------------

    def edge_confidence_gate(self, state: AssistantState) -> str:
        """Conditional Edge: Decides whether to route to answer extraction or refusal."""
        if not state["query_valid"]:
            return "route_refusal"

        if state["top_chunk"] is None or state["top_score"] <= 0.0:
            return "route_refusal"

        evaluation = state.get("evaluation")
        if evaluation and evaluation.decision == "ACCEPT":
            return "route_extract"
        return "route_refusal"

    # ------------------------------------------------------------------
    # Terminal Node A – Extract answer
    # ------------------------------------------------------------------

    def node_extract_answer(self, state: AssistantState) -> AssistantState:
        """Extracts grounded answer and formats 3-tier citation."""
        intent_type = state.get("intent") or IntentType.UNKNOWN
        extracted: ExtractedAnswer = self.extractor.extract(
            state["question"],
            state["top_chunk"],
            intent_type=intent_type,
        )
        state["status"] = "ANSWERED"
        state["answer"] = extracted.direct_answer
        state["citation"] = extracted.citation
        state["rejection_reason"] = None

        # Telemetry
        eval_obj = state.get("evaluation")
        event = TelemetryEvent(
            event_type="ANSWERED",
            question=state["question"],
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            confidence_score=round(state["top_score"], 3) if state["top_score"] else None,
            confidence_tier=eval_obj.confidence_tier if eval_obj else None,
        )
        state["telemetry_log"].append(event)

        return state

    # ------------------------------------------------------------------
    # Terminal Node B – Generate refusal
    # ------------------------------------------------------------------

    def node_generate_refusal(self, state: AssistantState) -> AssistantState:
        """Generates standardized refusal and appends support escalation."""
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

        # Telemetry
        event = TelemetryEvent(
            event_type="REFUSED",
            question=state["question"],
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rejection_reason=state["rejection_reason"],
        )
        state["telemetry_log"].append(event)

        return state

    # ------------------------------------------------------------------
    # Pipeline execution
    # ------------------------------------------------------------------

    def run(self, question: str) -> ContextIQResponse:
        """Executes the full graph pipeline for an incoming inquiry."""
        state: AssistantState = {
            "question": question,
            "query_valid": False,
            "intent": None,
            "query_analysis": None,
            "top_score": 0.0,
            "top_chunk": None,
            "debug_meta": {},
            "evaluation": None,
            "status": "REFUSED",
            "answer": "",
            "citation": None,
            "rejection_reason": None,
            "escalation_path": None,
            "telemetry_log": [],
        }

        # Step 1: Query analysis & intent classification
        state = self.node_analyze_query(state)

        # Step 2: Retrieval
        state = self.node_retrieve(state)

        # Step 3: Confidence evaluation
        state = self.node_evaluate_confidence(state)

        # Step 4: Conditional Routing
        decision = self.edge_confidence_gate(state)

        # Step 5: Terminal Node Execution
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
