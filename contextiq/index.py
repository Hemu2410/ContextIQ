"""
BM25 Search Index with Title Weighting and Corpus Vocabulary Registry.

Implements Okapi BM25 scoring with:
1. Title Field Boosting (2.5x weight)
2. Corpus Vocabulary & Entity Registry for Out-of-Vocabulary (OOV) Pre-Filtering
3. Score Normalization for Calibrated Thresholding
"""
from collections import Counter
import math
import re
from typing import List, Dict, Set, Tuple, Any
from .models import DocumentChunk

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "where", "which", "while", "who", "whom", "why", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def simple_stem(token: str) -> str:
    """Lightweight rule-based suffix normalization for inflections and plurals."""
    t = token.lower()
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 3 and not t.endswith(("ses", "zes")):
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
        return t[:-1]
    if t.endswith("ing") and len(t) > 5:
        return t[:-3]
    if t.endswith("ed") and len(t) > 4:
        return t[:-2]
    return t


def tokenize(text: str, remove_stopwords: bool = False, stem: bool = True) -> List[str]:
    """Tokenizes text, strips punctuation, and optionally stems tokens."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s_\-]", " ", text)
    tokens = [t.lower() for t in cleaned.split() if t.strip()]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [simple_stem(t) for t in tokens]
    return tokens


class BM25Index:
    """
    Okapi BM25 Index with:
    - Title Field Weighting (boost for terms in the document title)
    - Vocabulary and Proper-Noun Registry for Out-of-Vocabulary (OOV) pre-filtering
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, title_weight: float = 2.5):
        self.k1 = k1
        self.b = b
        self.title_weight = title_weight

        self.chunks: List[DocumentChunk] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.corpus_size: int = 0

        # Term frequencies per chunk
        self.doc_term_freqs: List[Counter] = []
        # Title terms per chunk
        self.title_term_sets: List[Set[str]] = []
        # Inverted index: term -> list of chunk indices
        self.inverted_index: Dict[str, List[int]] = {}
        # Inverse document frequencies: term -> float
        self.idf: Dict[str, float] = {}

        # Complete vocabulary of raw terms and stemmed terms across the entire corpus
        self.corpus_raw_vocab: Set[str] = set()
        self.corpus_stemmed_vocab: Set[str] = set()

    def build(self, chunks: List[DocumentChunk]) -> None:
        """Indexes document chunks and computes BM25 statistics."""
        self.chunks = chunks
        self.corpus_size = len(chunks)
        self.doc_len = []
        self.doc_term_freqs = []
        self.title_term_sets = []
        self.inverted_index = {}
        self.corpus_raw_vocab = set()
        self.corpus_stemmed_vocab = set()

        total_length = 0
        for idx, chunk in enumerate(chunks):
            body_tokens = tokenize(chunk.text, remove_stopwords=False, stem=True)
            title_tokens = tokenize(chunk.title, remove_stopwords=False, stem=True)

            raw_body = tokenize(chunk.text, remove_stopwords=False, stem=False)
            raw_title = tokenize(chunk.title, remove_stopwords=False, stem=False)
            self.corpus_raw_vocab.update(raw_body + raw_title)
            self.corpus_stemmed_vocab.update(body_tokens + title_tokens)

            length = len(body_tokens)
            self.doc_len.append(length)
            total_length += length

            tf = Counter(body_tokens)
            self.doc_term_freqs.append(tf)
            self.title_term_sets.append(set(title_tokens))

            unique_terms = set(body_tokens) | set(title_tokens)
            for term in unique_terms:
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(idx)

        self.avg_doc_len = (total_length / self.corpus_size) if self.corpus_size > 0 else 0.0

        # Calculate IDF with standard Okapi smoothing
        self.idf = {}
        for term, posting_list in self.inverted_index.items():
            df = len(posting_list)
            val = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[term] = max(val, 0.05)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, DocumentChunk, Dict[str, Any]]]:
        """
        Scores all chunks against the query using BM25 with title boosting.
        Returns list of (score, chunk, metadata) sorted descending.
        """
        q_tokens = tokenize(query, remove_stopwords=True, stem=True)
        if not q_tokens or self.corpus_size == 0:
            return []

        scores: List[float] = [0.0] * self.corpus_size
        matched_terms_per_chunk: List[List[str]] = [[] for _ in range(self.corpus_size)]

        for term in q_tokens:
            if term not in self.inverted_index:
                continue
            term_idf = self.idf.get(term, 0.0)
            postings = self.inverted_index[term]

            for chunk_idx in postings:
                tf = self.doc_term_freqs[chunk_idx].get(term, 0)
                # Apply title boost multiplier if term appears in title
                if term in self.title_term_sets[chunk_idx]:
                    tf = tf + (self.title_weight * 2.0)

                doc_l = self.doc_len[chunk_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_l / (self.avg_doc_len or 1.0)))
                term_score = term_idf * ((tf * (self.k1 + 1.0)) / (denom or 1.0))

                scores[chunk_idx] += term_score
                matched_terms_per_chunk[chunk_idx].append(term)

        # Max theoretical score for query normalization
        max_possible_score = sum(self.idf.get(term, 1.0) * (self.k1 + 1.0) for term in q_tokens)
        max_possible_score = max(max_possible_score, 1.0)

        results = []
        for idx, score in enumerate(scores):
            if score > 0:
                normalized_score = min(score / max_possible_score, 1.0)
                meta = {
                    "raw_score": score,
                    "normalized_score": round(normalized_score, 4),
                    "matched_terms": matched_terms_per_chunk[idx],
                    "query_terms_count": len(q_tokens),
                }
                results.append((normalized_score, self.chunks[idx], meta))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
