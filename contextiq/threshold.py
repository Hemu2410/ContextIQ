"""
Adaptive Confidence and Risk-Calibrated Rejection Evaluator.

Implements the multi-signal rejection boundary:
1. Out-of-Vocabulary (OOV) Entity Pre-Filter
2. Domain Risk Scaling (Security & Billing vs Informational)
3. Fact-Type Corroboration (Duration / Numeric checks)
"""
from dataclasses import dataclass
import re
from typing import Set, Tuple, Optional
from .models import DocumentChunk
from .index import BM25Index, tokenize, STOPWORDS

# High-risk domain terms where hallucination carries severe operational/security risk
HIGH_RISK_TERMS: Set[str] = {
    "password", "reset", "billing", "refund", "invoice",
    "charge", "security", "sso", "saml", "credential"
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


@dataclass
class ConfidenceEvaluation:
    decision: str  # "ACCEPT" or "REJECT"
    effective_threshold: float
    actual_score: float
    confidence_tier: str  # "HIGH", "MEDIUM", "REFUSAL"
    rejection_reason: Optional[str] = None


class AdaptiveConfidenceEvaluator:
    """Evaluates whether retrieval confidence meets the risk-calibrated threshold."""

    def __init__(self, index: BM25Index):
        self.index = index

    def detect_oov_entities(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Detects if query mentions a specific proper noun / entity that does not exist
        anywhere in the documentation corpus (e.g., 'Salesforce', 'Workday', 'Zapier').
        """
        # Find capitalized words that are not the first word of the query
        tokens = query.strip().split()
        if len(tokens) <= 1:
            return False, None

        for idx, token in enumerate(tokens):
            clean_token = re.sub(r"[^A-Za-z0-9]", "", token)
            if not clean_token or len(clean_token) < 3:
                continue

            # Check capitalized proper nouns (excluding the first word of the query)
            if clean_token[0].isupper() and idx > 0:
                token_lower = clean_token.lower()
                if token_lower in STOPWORDS or token_lower in {"solstice", "learning"}:
                    continue

                stemmed_tokens = tokenize(token_lower, remove_stopwords=False, stem=True)
                stemmed = stemmed_tokens[0] if stemmed_tokens else token_lower

                # If this capitalized noun does not exist anywhere in docs
                if token_lower not in self.index.corpus_raw_vocab and stemmed not in self.index.corpus_stemmed_vocab:
                    return True, clean_token

        return False, None

    def evaluate(self, query: str, top_score: float, top_chunk: DocumentChunk) -> ConfidenceEvaluation:
        """Evaluates whether an answer should be served or rejected."""
        # Check 1: Out-of-vocabulary entity check
        has_oov, oov_entity = self.detect_oov_entities(query)
        if has_oov:
            return ConfidenceEvaluation(
                decision="REJECT",
                effective_threshold=0.50,
                actual_score=top_score,
                confidence_tier="REFUSAL",
                rejection_reason=f"Query references unindexed external entity: '{oov_entity}'",
            )

        q_lower = query.lower()
        chunk_text_lower = top_chunk.text.lower()

        # Check 2: Domain Risk Scaling
        is_high_risk = any(term in q_lower or term in top_chunk.title.lower() for term in HIGH_RISK_TERMS)
        base_threshold = 0.50 if is_high_risk else 0.35

        # Check 3: Fact-Type Corroboration
        is_quantitative_query = any(re.search(pat, q_lower) for pat in QUANTITATIVE_PATTERNS)
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
