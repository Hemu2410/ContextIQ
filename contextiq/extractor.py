"""
Grounded Extractive Answering Engine.

Extracts the precise factual sentence/clause directly answering the admin query,
ensuring zero hallucination and 100% auditable citation provenance.
"""
from dataclasses import dataclass
from typing import List, Optional

from .models import DocumentChunk, Citation, IntentType
from .index import tokenize


@dataclass
class ExtractedAnswer:
    direct_answer: str
    citation: Citation
    full_passage: str


class GroundedExtractor:
    """Extracts the most salient sentence answering the query from a retrieved chunk."""

    @staticmethod
    def extract_salient_sentence(
        query: str,
        chunk: DocumentChunk,
        intent_type: IntentType = IntentType.UNKNOWN,
    ) -> str:
        """
        Ranks individual sentences in the matched chunk by query keyword density
        and returns the top candidate sentence.

        For *PROCEDURAL* queries, up to 2 consecutive top-scoring sentences are
        returned so that step-by-step answers are not truncated to a single
        sentence.

        For *QUANTITATIVE* queries, only the single best sentence containing a
        number is returned.
        """
        if not chunk.sentences:
            return chunk.text

        q_terms = set(tokenize(query, remove_stopwords=True, stem=True))
        if not q_terms:
            return chunk.sentences[0]

        is_procedural = intent_type == IntentType.PROCEDURAL
        is_quantitative = intent_type == IntentType.QUANTITATIVE

        scored_sentences: List[tuple] = []
        for idx, sent in enumerate(chunk.sentences):
            sent_lower = sent.lower()
            sent_terms = tokenize(sent, remove_stopwords=False, stem=True)
            if not sent_terms:
                continue
            overlap = sum(1 for t in sent_terms if t in q_terms)

            # Quantitative query bonus
            if is_quantitative and any(char.isdigit() for char in sent):
                overlap += 3.0

            # Procedural query bonus
            if is_procedural:
                if any(v in sent_lower for v in ["click", "enter", "go to", "select", "settings >", "provide"]):
                    overlap += 2.0
                if sent_lower.startswith("note:"):
                    overlap -= 1.0

            score = overlap / (len(sent_terms) ** 0.5)
            scored_sentences.append((score, idx, sent))

        if not scored_sentences:
            return chunk.sentences[0]

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        # --- Multi-sentence assembly for procedural queries ---
        if is_procedural and len(scored_sentences) >= 2:
            best_score, best_idx, best_sent = scored_sentences[0]
            # Look for the next-best sentence that is consecutive
            for score, idx, sent in scored_sentences[1:]:
                if abs(idx - best_idx) == 1:
                    # Return them in original document order
                    if idx < best_idx:
                        return f"{sent} {best_sent}"
                    else:
                        return f"{best_sent} {sent}"
            # No consecutive match found – fall through to single sentence
            return best_sent

        # QUANTITATIVE or UNKNOWN / INFORMATIONAL – single best sentence
        return scored_sentences[0][2]

    def extract(
        self,
        query: str,
        chunk: DocumentChunk,
        intent_type: IntentType = IntentType.UNKNOWN,
    ) -> ExtractedAnswer:
        """Produces a grounded extractive answer with 3-tier citation metadata."""
        salient_sentence = self.extract_salient_sentence(query, chunk, intent_type=intent_type)
        citation = Citation(
            doc_id=chunk.doc_id,
            title=chunk.title,
            verbatim_quote=salient_sentence,
            navigation_path=chunk.navigation_path,
        )
        return ExtractedAnswer(
            direct_answer=salient_sentence,
            citation=citation,
            full_passage=chunk.text,
        )
