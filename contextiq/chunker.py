"""
Hierarchical Structural Chunker for ContextIQ.

Preserves natural paragraph and sentence boundaries, eliminates mid-word
truncation, extracts UI navigation breadcrumbs, and maintains hierarchical links
between document context and atomic fact sentences.
"""
import json
import re
from typing import List, Dict, Any, Optional
from .models import Document, DocumentChunk


class HierarchicalChunker:
    """Chunks documents into semantic sections with indexed atomic child sentences."""

    @staticmethod
    def extract_navigation_path(text: str) -> Optional[str]:
        """Detects UI navigation patterns such as 'Settings > Team > Invite Member'."""
        nav_match = re.search(r"([A-Z][a-zA-Z\s]+(?:\s*>\s*[A-Z][a-zA-Z\s]+)+)", text)
        if nav_match:
            return nav_match.group(1).strip()
        return None

    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Splits text into complete grammatical sentences without breaking numbers/abbreviations."""
        cleaned = text.strip()
        if not cleaned:
            return []
        # Split on sentence terminals (. ! ?) followed by whitespace and a capital letter/number
        pattern = r"(?<=[.!?])\s+(?=[A-Z0-9'\"`])"
        raw_sentences = re.split(pattern, cleaned)
        return [s.strip() for s in raw_sentences if s.strip()]

    def chunk_document(self, doc: Document, max_sentences_per_chunk: int = 5) -> List[DocumentChunk]:
        """
        Chunks an article into cohesive passages while indexing its child sentences.
        Preserves complete paragraphs for context.
        """
        sentences = self.split_into_sentences(doc.content)
        if not sentences:
            return []

        nav_path = self.extract_navigation_path(doc.content)

        # For single-topic compact docs (like Solstice's articles), retain as coherent section chunk
        if len(sentences) <= max_sentences_per_chunk + 2:
            return [
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}-c0",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.content,
                    sentences=sentences,
                    navigation_path=nav_path,
                    metadata=doc.metadata,
                )
            ]

        # For longer articles, chunk with 1-sentence overlap to maintain context
        chunks: List[DocumentChunk] = []
        step = max(1, max_sentences_per_chunk - 1)
        for i in range(0, len(sentences), step):
            window = sentences[i : i + max_sentences_per_chunk]
            if not window:
                continue
            chunk_text = " ".join(window)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}-c{len(chunks)}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=chunk_text,
                    sentences=window,
                    navigation_path=nav_path,
                    metadata=doc.metadata,
                )
            )
            if i + max_sentences_per_chunk >= len(sentences):
                break

        return chunks

    def chunk_all(self, docs: List[Document]) -> List[DocumentChunk]:
        """Chunks a collection of Document objects."""
        all_chunks: List[DocumentChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks


def load_documents_from_json(file_path: str) -> List[Document]:
    """Loads raw documents from a Solstice JSON file into Document models."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_list = data.get("docs", [])
    documents: List[Document] = []
    for item in raw_list:
        documents.append(
            Document(
                doc_id=item.get("doc_id", "UNKNOWN"),
                title=item.get("title", ""),
                content=item.get("content", ""),
                metadata=item.get("metadata", {}),
            )
        )
    return documents
