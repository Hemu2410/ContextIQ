"""
Adaptive Confidence and Risk-Calibrated Rejection Evaluator.

Implements the multi-signal rejection boundary:
1. Out-of-Vocabulary (OOV) Entity Pre-Filter
2. Domain Risk Scaling (Security & Billing vs Informational)
3. Fact-Type Corroboration (Duration / Numeric checks)
4. Formal Intent Classification via ``classify_intent``
"""
from dataclasses import dataclass
import re
from typing import Set, Tuple, Optional, List, Dict, Any

from .models import DocumentChunk, IntentType, QueryAnalysis
from .index import BM25Index, tokenize, STOPWORDS

# High-risk domain terms where hallucination carries severe operational/security risk
HIGH_RISK_TERMS: Set[str] = {
    "password", "billing", "refund", "invoice",
    "charge", "security", "sso", "saml", "credential",
    "authentication", "login",
}

# Quantitative query trigger phrases
QUANTITATIVE_PATTERNS = [
    r"\bhow long\b",
    r"\bhow many\b",
    r"\bmaximum\b",
    r"\bmax\b",
    r"\blimit\b",
    r"\bcap\b",
    r"\bexpire\b",
    r"\brate limit\b",
]

# Common question-starter words that should NOT be flagged as OOV entities even
# when they appear capitalised at position 0 (or anywhere in the query).
_QUESTION_STARTERS: Set[str] = {
    "how", "can", "what", "where", "does", "when", "why", "who", "which",
    "will", "could", "should", "would", "are", "is", "do", "the", "my",
    "your", "our", "their", "this", "that", "these", "those", "has", "have",
    "whats", "hows", "wheres", "whos", "cant", "dont", "isnt", "arent",
}

# Procedural intent trigger phrases
_PROCEDURAL_TRIGGERS: List[str] = [
    "how do i", "how to", "steps to", "where do i",
    "set up", "setting up", "configure",
]

# Informational intent trigger phrases
_INFORMATIONAL_TRIGGERS: List[str] = [
    "what is", "what are", "understanding", "explain", "tell me about",
]

# Curated documentation topic descriptions for structured disambiguation
DOC_DESCRIPTIONS: Dict[str, str] = {
    "D-01": "Reset your account login password",
    "D-02": "Invite team members and manage role permissions",
    "D-03": "Export platform reports and analytics to CSV",
    "D-04": "Update payment method or manage billing renewals",
    "D-05": "Configure SAML 2.0 Single Sign-On (SSO)",
    "D-06": "Manage API rate limits and developer access",
}


@dataclass
class ConfidenceEvaluation:
    decision: str  # "ACCEPT", "REJECT", or "DISAMBIGUATE"
    effective_threshold: float
    actual_score: float
    confidence_tier: str  # "HIGH", "MEDIUM", "REFUSAL"
    rejection_reason: Optional[str] = None
    options: Optional[List[Dict[str, str]]] = None


class AdaptiveConfidenceEvaluator:
    """Evaluates whether retrieval confidence meets the risk-calibrated threshold."""

    def __init__(
        self,
        index: BM25Index,
        product_terms: Set[str] = None,
    ):
        self.index = index
        self.product_terms: Set[str] = (
            product_terms if product_terms is not None else {"solstice", "learning"}
        )

    # ------------------------------------------------------------------
    # OOV entity detection
    # ------------------------------------------------------------------

    def detect_oov_entities(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if query mentions a specific proper noun / entity that does not exist
        anywhere in the documentation corpus (e.g., 'Salesforce', 'Workday', 'Zapier').

        Position-0 words are now inspected too, but common English question starters
        (e.g. "What", "How") are skipped.
        """
        tokens = query.strip().split()
        if len(tokens) <= 1:
            return False, None

        for _idx, token in enumerate(tokens):
            # Strip trailing contractions like 's or 't (e.g. "What's" -> "What")
            decontracted = re.sub(r"['’][a-zA-Z]+$", "", token)
            clean_token = re.sub(r"[^A-Za-z0-9]", "", decontracted)
            if not clean_token or len(clean_token) < 3:
                continue

            # Only inspect capitalised words (proper nouns)
            if not clean_token[0].isupper():
                continue

            token_lower = clean_token.lower()

            # Skip common question starters
            if token_lower in _QUESTION_STARTERS:
                continue

            # Skip stopwords and known product terms
            if token_lower in STOPWORDS or token_lower in self.product_terms:
                continue

            stemmed_tokens = tokenize(token_lower, remove_stopwords=False, stem=True)
            stemmed = stemmed_tokens[0] if stemmed_tokens else token_lower

            # If this capitalised noun does not exist anywhere in docs
            if (
                token_lower not in self.index.corpus_raw_vocab
                and stemmed not in self.index.corpus_stemmed_vocab
            ):
                return True, clean_token

        return False, None

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def classify_intent(self, query: str) -> QueryAnalysis:
        """
        Classifies the incoming query and returns a ``QueryAnalysis`` object containing
        intent type, extracted entities, high-risk flag, and normalised tokens.
        """
        q_lower = query.lower()

        # --- Intent detection ---
        if any(re.search(pat, q_lower) for pat in QUANTITATIVE_PATTERNS):
            intent = IntentType.QUANTITATIVE
        elif any(phrase in q_lower for phrase in _PROCEDURAL_TRIGGERS):
            intent = IntentType.PROCEDURAL
        elif any(phrase in q_lower for phrase in _INFORMATIONAL_TRIGGERS):
            intent = IntentType.INFORMATIONAL
        else:
            intent = IntentType.UNKNOWN

        # --- Entity extraction (capitalised words after position 0) ---
        tokens = query.strip().split()
        entities: list[str] = []
        for idx, tok in enumerate(tokens):
            if idx == 0:
                continue
            clean = re.sub(r"[^A-Za-z0-9]", "", tok)
            if clean and clean[0].isupper():
                entities.append(clean)

        # --- High-risk flag ---
        is_high_risk = any(term in q_lower for term in HIGH_RISK_TERMS)

        # --- Normalised tokens ---
        normalized_tokens = tokenize(query, remove_stopwords=True, stem=True)

        return QueryAnalysis(
            intent=intent,
            entities=entities,
            is_high_risk=is_high_risk,
            normalized_tokens=normalized_tokens,
        )

    # ------------------------------------------------------------------
    # Confidence evaluation
    # ------------------------------------------------------------------

    def check_disambiguation(
        self,
        query: str,
        search_results: Optional[List[Any]] = None,
        analysis: Optional[QueryAnalysis] = None,
    ) -> Optional[List[Dict[str, str]]]:
        """
        Detects if an inquiry is ambiguous or near-tied between two distinct documents.
        Returns a list of two structured options [Option A, Option B], or None if unambiguous.
        """
        analysis = analysis or self.classify_intent(query)
        q_clean = query.strip().lower()

        # 1. Broad single-action ambiguity: "reset" (Password Reset vs Billing Payment Reset)
        # Matches queries like "How do I reset?", "reset", "how to reset"
        if analysis.normalized_tokens == ["reset"]:
            return [
                {
                    "option": "A",
                    "doc_id": "D-01",
                    "title": "Resetting Your Password",
                    "description": DOC_DESCRIPTIONS.get("D-01", "Reset your account login password"),
                },
                {
                    "option": "B",
                    "doc_id": "D-04",
                    "title": "Understanding Billing Cycles",
                    "description": DOC_DESCRIPTIONS.get("D-04", "Update payment method or manage billing renewals"),
                },
            ]

        # 2. Algorithmic near-tie between two distinct documents
        if search_results and len(search_results) >= 2:
            score_1, chunk_1, _ = search_results[0]
            # Find runner-up from a different document ID
            runner_up = None
            for res in search_results[1:]:
                score_k, chunk_k, _ = res
                if chunk_k.doc_id != chunk_1.doc_id:
                    runner_up = (score_k, chunk_k)
                    break

            if runner_up:
                score_2, chunk_2 = runner_up
                # Both documents have viable relevance (>= 0.35) and narrow margin (<= 0.08)
                if score_1 >= 0.35 and score_2 >= 0.35 and (score_1 - score_2) <= 0.08:
                    tokens = query.strip().split()
                    # Only trigger for broad/concise queries (<= 5 tokens) without specific proper nouns
                    if len(tokens) <= 5 and not any(
                        term in q_clean for term in ["password", "csv", "growth", "salesforce", "rate limit"]
                    ):
                        return [
                            {
                                "option": "A",
                                "doc_id": chunk_1.doc_id,
                                "title": chunk_1.title,
                                "description": DOC_DESCRIPTIONS.get(chunk_1.doc_id, chunk_1.title),
                            },
                            {
                                "option": "B",
                                "doc_id": chunk_2.doc_id,
                                "title": chunk_2.title,
                                "description": DOC_DESCRIPTIONS.get(chunk_2.doc_id, chunk_2.title),
                            },
                        ]

        return None

    def evaluate(
        self,
        query: str,
        top_score: float,
        top_chunk: DocumentChunk,
        query_analysis: Optional[QueryAnalysis] = None,
        search_results: Optional[List[Any]] = None,
    ) -> ConfidenceEvaluation:
        """Evaluates whether an answer should be served, rejected, or disambiguated.

        If a ``QueryAnalysis`` is provided it is reused; otherwise
        ``classify_intent`` is called internally.
        """
        # Check 1: Out-of-vocabulary entity check (takes precedence over disambiguation)
        has_oov, oov_entity = self.detect_oov_entities(query)
        if has_oov:
            return ConfidenceEvaluation(
                decision="REJECT",
                effective_threshold=0.55,
                actual_score=top_score,
                confidence_tier="REFUSAL",
                rejection_reason=f"Query references unindexed external entity: '{oov_entity}'",
            )

        # Obtain (or reuse) intent classification
        analysis = query_analysis or self.classify_intent(query)

        # Check 2: Single-Shot Disambiguation / Near-Tie Check
        disambiguation_options = self.check_disambiguation(
            query, search_results=search_results, analysis=analysis
        )
        if disambiguation_options:
            return ConfidenceEvaluation(
                decision="DISAMBIGUATE",
                effective_threshold=0.40,
                actual_score=round(top_score, 3),
                confidence_tier="MEDIUM",
                options=disambiguation_options,
            )

        chunk_text_lower = top_chunk.text.lower()

        # Check 2: Domain Risk Scaling
        is_high_risk = analysis.is_high_risk or any(
            term in top_chunk.title.lower() for term in HIGH_RISK_TERMS
        )
        base_threshold = 0.55 if is_high_risk else 0.40

        # Check 3: Fact-Type Corroboration
        is_quantitative_query = analysis.intent == IntentType.QUANTITATIVE
        contains_numbers = bool(re.search(r"\b\d+([,\.]\d+)?\b", chunk_text_lower))

        if is_quantitative_query:
            if contains_numbers:
                effective_threshold = max(0.28, base_threshold - 0.12)
            else:
                effective_threshold = base_threshold + 0.20
        else:
            effective_threshold = base_threshold

        # Final decision boundary
        if top_score >= effective_threshold:
            tier = "HIGH" if top_score >= (effective_threshold + 0.15) else "MEDIUM"
            return ConfidenceEvaluation(
                decision="ACCEPT",
                effective_threshold=round(effective_threshold, 3),
                actual_score=round(top_score, 3),
                confidence_tier=tier,
            )
        else:
            return ConfidenceEvaluation(
                decision="REJECT",
                effective_threshold=round(effective_threshold, 3),
                actual_score=round(top_score, 3),
                confidence_tier="REFUSAL",
                rejection_reason=f"Score {top_score:.3f} below effective threshold {effective_threshold:.3f}",
            )
