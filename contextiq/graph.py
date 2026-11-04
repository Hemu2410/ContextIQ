"""
StateGraph Orchestrator for ContextIQ.

Models the question-answering workflow as a directed acyclic state graph:
START -> check_session_pending -> analyze_query -> retrieve_candidates -> evaluate_confidence -> [confidence_gate]
         ├── (Pass)         -> extract_grounded_answer    -> END
         ├── (Disambiguate) -> generate_disambiguation    -> END (saves pending options to session)
         └── (Refuse)       -> generate_refusal           -> END

Compatible with LangGraph StateGraph, while providing a self-contained
deterministic execution fallback with Short-Term Session Memory and Single-Shot
Bounded Disambiguation / Human-in-the-Loop (HITL).
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
from .index import BM25Index, tokenize
from .threshold import AdaptiveConfidenceEvaluator, ConfidenceEvaluation
from .extractor import GroundedExtractor, ExtractedAnswer
from .session import SessionManager, SessionState

STANDARD_REFUSAL_MESSAGE = "I don't have an answer for that in the Solstice Learning documentation."
SUPPORT_ESCALATION_NOTE = "If you need immediate assistance with this topic, please contact support@solsticelearning.com."


class AssistantState(TypedDict):
    question: str
    session_id: Optional[str]
    query_valid: bool
    intent: Optional[IntentType]
    query_analysis: Optional[QueryAnalysis]
    top_score: float
    top_chunk: Optional[DocumentChunk]
    search_results: List[Any]
    debug_meta: Dict[str, Any]
    evaluation: Optional[ConfidenceEvaluation]
    status: str
    answer: str
    citation: Optional[Citation]
    rejection_reason: Optional[str]
    escalation_path: Optional[str]
    options: Optional[List[Dict[str, str]]]
    telemetry_log: List[TelemetryEvent]


class ContextIQStateGraph:
    """Directed state graph coordinating analysis, retrieval, gating, disambiguation, and extraction."""

    def __init__(
        self,
        index: BM25Index,
        evaluator: AdaptiveConfidenceEvaluator,
        extractor: GroundedExtractor,
        session_manager: Optional[SessionManager] = None,
    ):
        self.index = index
        self.evaluator = evaluator
        self.extractor = extractor
        self.session_manager = session_manager or SessionManager()

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
    # Node 2 – Retrieval (top_k=5 for disambiguation depth)
    # ------------------------------------------------------------------

    def node_retrieve(self, state: AssistantState) -> AssistantState:
        """Retrieves top matching candidates from the BM25 index."""
        if not state["query_valid"]:
            state["top_score"] = 0.0
            state["top_chunk"] = None
            state["search_results"] = []
            return state

        results = self.index.search(state["question"], top_k=5)
        state["search_results"] = results
        if results:
            score, chunk, meta = results[0]
            state["top_score"] = score
            state["top_chunk"] = chunk
            state["debug_meta"] = meta
        else:
            state["top_score"] = 0.0
            state["top_chunk"] = None
            state["debug_meta"] = {}
        return state

    # ------------------------------------------------------------------
    # Node 3 – Confidence & Ambiguity evaluation
    # ------------------------------------------------------------------

    def node_evaluate_confidence(self, state: AssistantState) -> AssistantState:
        """Evaluates retrieval confidence and checks for ambiguous near-ties."""
        if not state["query_valid"] or state["top_chunk"] is None or state["top_score"] <= 0.0:
            state["evaluation"] = None
            return state

        evaluation = self.evaluator.evaluate(
            state["question"],
            state["top_score"],
            state["top_chunk"],
            query_analysis=state.get("query_analysis"),
            search_results=state.get("search_results"),
        )
        state["evaluation"] = evaluation
        if evaluation.options:
            state["options"] = evaluation.options
        return state

    # ------------------------------------------------------------------
    # Conditional edge – pure routing, no mutation
    # ------------------------------------------------------------------

    def edge_confidence_gate(self, state: AssistantState) -> str:
        """Conditional Edge: Routes to extraction, disambiguation, or refusal."""
        if not state["query_valid"]:
            return "route_refusal"

        if state["top_chunk"] is None or state["top_score"] <= 0.0:
            return "route_refusal"

        evaluation = state.get("evaluation")
        if evaluation:
            if evaluation.decision == "DISAMBIGUATE":
                return "route_disambiguate"
            if evaluation.decision == "ACCEPT":
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
    # Terminal Node C – Generate single-shot disambiguation (HITL)
    # ------------------------------------------------------------------

    def node_generate_disambiguation(self, state: AssistantState) -> AssistantState:
        """Generates structured clarification choices and records pending options in session."""
        options = state.get("options") or []
        opt_lines = []
        for opt in options:
            key = opt.get("option", "A")
            title = opt.get("title", "")
            desc = opt.get("description", "")
            opt_lines.append(f"  [{key}] {title} - {desc}")

        options_text = "\n".join(opt_lines)
        state["status"] = "DISAMBIGUATION"
        state["answer"] = (
            f"Your inquiry could refer to multiple topics in Solstice documentation. Did you mean:\n\n"
            f"{options_text}\n\n"
            f"Please reply with 'A' or 'B' (or clarify your topic) to see the verified instructions."
        )
        state["citation"] = None
        state["rejection_reason"] = "Query is ambiguous across multiple documentation topics"

        # Save pending disambiguation into the active session for follow-up resolution
        sid = state.get("session_id")
        if sid and self.session_manager:
            self.session_manager.set_pending_disambiguation(
                session_id=sid,
                original_query=state["question"],
                options=options,
            )

        # Telemetry
        event = TelemetryEvent(
            event_type="DISAMBIGUATION",
            question=state["question"],
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            rejection_reason=state["rejection_reason"],
        )
        state["telemetry_log"].append(event)
        return state

    # ------------------------------------------------------------------
    # Pipeline execution (Session-Aware)
    # ------------------------------------------------------------------

    def run(self, question: str, session_id: Optional[str] = None) -> ContextIQResponse:
        """Executes the full graph pipeline for an incoming inquiry with session memory."""
        # 1. Retrieve or initialize short-term session state
        session = self.session_manager.get_or_create_session(session_id)
        actual_sid = session.session_id

        # 2. Check for Turn-2 Follow-Up Resolution (Single-Shot Bounded Disambiguation)
        if session.pending_disambiguation:
            pending = session.pending_disambiguation
            options = pending.get("options", [])
            q_clean = question.strip().lower()

            chosen_option = None
            # Check for direct key match ('A', 'Option A', '1', 'first')
            if len(options) >= 1 and q_clean in {"a", "option a", "a.", "[a]", "1", "first", "the first one"}:
                chosen_option = options[0]
            # Check for direct key match ('B', 'Option B', '2', 'second')
            elif len(options) >= 2 and q_clean in {"b", "option b", "b.", "[b]", "2", "second", "the second one"}:
                chosen_option = options[1]
            else:
                # Check for semantic keyword overlap with Option A or B titles
                for opt in options:
                    opt_title_terms = set(tokenize(opt["title"], remove_stopwords=True))
                    q_terms = set(tokenize(q_clean, remove_stopwords=True))
                    if q_terms and opt_title_terms.intersection(q_terms):
                        chosen_option = opt
                        break

            # Strictly bounded: clear pending disambiguation regardless of outcome
            self.session_manager.clear_pending(actual_sid)

            if chosen_option:
                target_doc_id = chosen_option["doc_id"]
                target_chunks = [c for c in self.index.chunks if c.doc_id == target_doc_id]
                if target_chunks:
                    chosen_chunk = target_chunks[0]
                    extracted = self.extractor.extract(
                        chosen_option["title"],
                        chosen_chunk,
                        intent_type=IntentType.PROCEDURAL,
                    )
                    return ContextIQResponse(
                        status=Status.ANSWERED,
                        question=question,
                        answer=extracted.direct_answer,
                        citation=extracted.citation,
                        confidence_tier=ConfidenceTier.HIGH,
                        confidence_score=0.95,
                        effective_threshold=0.40,
                        session_id=actual_sid,
                    )

        # 3. Standard Graph Execution (Turn-1 or Unambiguous Follow-Up)
        state: AssistantState = {
            "question": question,
            "session_id": actual_sid,
            "query_valid": False,
            "intent": None,
            "query_analysis": None,
            "top_score": 0.0,
            "top_chunk": None,
            "search_results": [],
            "debug_meta": {},
            "evaluation": None,
            "status": "REFUSED",
            "answer": "",
            "citation": None,
            "rejection_reason": None,
            "escalation_path": None,
            "options": None,
            "telemetry_log": [],
        }

        # Step 1: Query analysis & intent classification
        state = self.node_analyze_query(state)

        # Step 2: Retrieval
        state = self.node_retrieve(state)

        # Step 3: Confidence & Ambiguity evaluation
        state = self.node_evaluate_confidence(state)

        # Step 4: Conditional Routing
        decision = self.edge_confidence_gate(state)

        # Step 5: Terminal Node Execution
        if decision == "route_extract":
            state = self.node_extract_answer(state)
        elif decision == "route_disambiguate":
            state = self.node_generate_disambiguation(state)
        else:
            state = self.node_generate_refusal(state)

        # Return standardized response contract
        eval_obj = state.get("evaluation")
        tier = eval_obj.confidence_tier if eval_obj else "REFUSAL"
        threshold = eval_obj.effective_threshold if eval_obj else 0.40

        status_enum = Status(state["status"]) if state["status"] in Status._value2member_map_ else Status.REFUSED

        return ContextIQResponse(
            status=status_enum,
            question=question,
            answer=state["answer"],
            citation=state["citation"],
            confidence_tier=ConfidenceTier(tier) if tier in ConfidenceTier._value2member_map_ else ConfidenceTier.REFUSAL,
            confidence_score=round(state["top_score"], 3),
            effective_threshold=round(threshold, 3),
            rejection_reason=state.get("rejection_reason"),
            escalation_path=state.get("escalation_path"),
            session_id=actual_sid,
            options=state.get("options"),
        )
