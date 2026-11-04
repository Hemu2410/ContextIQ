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
from typing import Set, Tuple, Optional, List

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


@dataclass
class ConfidenceEvaluation:
    decision: str  # "ACCEPT" or "REJECT"
    effective_threshold: float
    actual_score: float
    confidence_tier: str  # "HIGH", "MEDIUM", "REFUSAL"
    rejection_reason: Optional[str] = None


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

    def evaluate(
        self,
        query: str,
        top_score: float,
        top_chunk: DocumentChunk,
        query_analysis: Optional[QueryAnalysis] = None,
    ) -> ConfidenceEvaluation:
        """Evaluates whether an answer should be served or rejected.

        If a ``QueryAnalysis`` is provided it is reused; otherwise
        ``classify_intent`` is called internally.
        """
        # Check 1: Out-of-vocabulary entity check
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
