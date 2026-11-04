"""
Data models and typed contracts for ContextIQ.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class Document:
    doc_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    sentences: List[str] = field(default_factory=list)
    navigation_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Citation:
    doc_id: str
    title: str
    verbatim_quote: str
    navigation_path: Optional[str] = None


@dataclass
class ContextIQResponse:
    status: str  # "ANSWERED" | "REFUSED"
    question: str
    answer: str
    citation: Optional[Citation]
    confidence_tier: str  # "HIGH" | "MEDIUM" | "REFUSAL"
    confidence_score: float
    effective_threshold: float
    rejection_reason: Optional[str] = None
    escalation_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data
