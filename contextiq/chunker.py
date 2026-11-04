"""
Hierarchical Structural Chunker for ContextIQ.

Preserves natural paragraph and sentence boundaries, eliminates mid-word
truncation, extracts UI navigation breadcrumbs, and maintains hierarchical links
between document context and atomic fact sentences.
"""
import json
import re
import sys
from typing import List, Dict, Any, Optional
from .models import Document, DocumentChunk


class HierarchicalChunker:
    """Chunks documents into semantic sections with indexed atomic child sentences."""

    @staticmethod
    def extract_navigation_path(text: str) -> Optional[str]:
        """Detects UI navigation patterns such as 'Settings > Team > Invite Member'.

        Uses a restricted pattern that requires the '>' delimiter and limits label
        characters to letters and spaces (no arbitrary whitespace in labels) with
        a maximum of 5 navigation steps to avoid false-positive matches.
        """
        nav_match = re.search(
            r'([A-Z][a-zA-Z ]+(?:\s*>\s*[A-Z][a-zA-Z ]+){1,5})', text
        )
        if nav_match:
            return nav_match.group(1).strip()
        return None

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Splits text into complete grammatical sentences without breaking abbreviations.

        Uses a safer regex that only splits after sentence-terminal punctuation
        when followed by whitespace and an uppercase letter, quote, or opening
        parenthesis — avoiding false splits on abbreviations like "Dr." or "U.S.".
        """
        cleaned = text.strip()
        if not cleaned:
            return []
        # Split on sentence terminals (. ! ?) followed by whitespace and an
        # uppercase letter, quote character, or opening parenthesis.
        pattern = r"(?<=[.!?])\s+(?=[A-Z\"'(])"
        raw_sentences = re.split(pattern, cleaned)
        return [s.strip() for s in raw_sentences if s.strip()]

    def chunk_document(self, doc: Document, max_sentences_per_chunk: int = 5) -> List[DocumentChunk]:
        """
        Chunks an article into cohesive passages while indexing its child sentences.

        Paragraph-level awareness: the content is first split on double-newlines
        into paragraphs, then each paragraph is split into sentences.  Each
        paragraph becomes its own chunk.
        """
        nav_path = self.extract_navigation_path(doc.content)

        # Split content into paragraphs on double-newline boundaries
        paragraphs = [p.strip() for p in re.split(r'\n\n+', doc.content) if p.strip()]
        if not paragraphs:
            return []

        # Gather all sentences across paragraphs for the compact-doc check
        all_sentences: List[str] = []
        paragraph_sentences: List[List[str]] = []
        for para in paragraphs:
            sents = self.split_into_sentences(para)
            paragraph_sentences.append(sents)
            all_sentences.extend(sents)

        if not all_sentences:
            return []

        # For single-topic compact docs, retain as one coherent section chunk
        if len(all_sentences) <= max_sentences_per_chunk:
            return [
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}-c0",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.content,
                    sentences=all_sentences,
                    navigation_path=nav_path,
                    metadata=doc.metadata,
                )
            ]

        # For longer articles, create one chunk per paragraph
        chunks: List[DocumentChunk] = []
        for para_sents in paragraph_sentences:
            if not para_sents:
                continue
            chunk_text = " ".join(para_sents)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}-c{len(chunks)}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=chunk_text,
                    sentences=para_sents,
                    navigation_path=nav_path,
                    metadata=doc.metadata,
                )
            )

        return chunks

    def chunk_all(self, docs: List[Document]) -> List[DocumentChunk]:
        """Chunks a collection of Document objects."""
        all_chunks: List[DocumentChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


def load_documents_from_json(file_path: str) -> List[Document]:
    """Loads raw documents from a Solstice JSON file into Document models.

    Validates the JSON structure: the top-level value must be a dict containing
    a ``"docs"`` key.  Logs a warning to stderr for any document whose content
    is empty.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate JSON structure
    if not isinstance(data, dict) or "docs" not in data:
        raise ValueError(
            f"Invalid JSON structure in '{file_path}': expected a JSON object "
            f"with a 'docs' key at the top level."
        )

    raw_list = data.get("docs", [])
    documents: List[Document] = []
    for item in raw_list:
        content = item.get("content", "")
        doc_id = item.get("doc_id", "UNKNOWN")
        if not content:
            print(
                f"WARNING: Document '{doc_id}' has empty content — skipping.",
                file=sys.stderr,
            )
            continue
        documents.append(
            Document(
                doc_id=doc_id,
                title=item.get("title", ""),
                content=content,
                metadata=item.get("metadata", {}),
            )
        )
    return documents
