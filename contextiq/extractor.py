"""
Grounded Extractive Answering Engine.

Extracts the precise factual sentence/clause directly answering the admin query,
ensuring zero hallucination and 100% auditable citation provenance.
"""
from dataclasses import dataclass
from typing import List, Optional
from .models import DocumentChunk, Citation
from .index import tokenize


@dataclass
class ExtractedAnswer:
    direct_answer: str
    citation: Citation
    full_passage: str


class GroundedExtractor:
    """Extracts the most salient sentence answering the query from a retrieved chunk."""

    @staticmethod
    def extract_salient_sentence(query: str, chunk: DocumentChunk) -> str:
        """
        Ranks individual sentences in the matched chunk by query keyword density
        and returns the top candidate sentence.
        """
        if not chunk.sentences:
            return chunk.text

        q_terms = set(tokenize(query, remove_stopwords=True, stem=True))
        if not q_terms:
            return chunk.sentences[0]

        is_procedural = any(p in query.lower() for p in ["how do i", "how to", "steps", "where do i"])
        is_quantitative = any(w in query.lower() for w in ["limit", "expire", "many", "much", "long", "maximum", "cap"])

        scored_sentences: List[tuple] = []
        for sent in chunk.sentences:
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
            scored_sentences.append((score, sent))

        if not scored_sentences:
            return chunk.sentences[0]

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return scored_sentences[0][1]

    def extract(self, query: str, chunk: DocumentChunk) -> ExtractedAnswer:
        """Produces a grounded extractive answer with 3-tier citation metadata."""
        salient_sentence = self.extract_salient_sentence(query, chunk)
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
