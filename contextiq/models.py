"""
Data models and typed contracts for ContextIQ.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any


class Status(str, Enum):
    ANSWERED = "ANSWERED"
    REFUSED = "REFUSED"


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    REFUSAL = "REFUSAL"


class IntentType(str, Enum):
    QUANTITATIVE = "QUANTITATIVE"
    PROCEDURAL = "PROCEDURAL"
    INFORMATIONAL = "INFORMATIONAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class QueryAnalysis:
    intent: IntentType = IntentType.UNKNOWN
    entities: List[str] = field(default_factory=list)
    is_high_risk: bool = False
    normalized_tokens: List[str] = field(default_factory=list)

    @property
    def intent_type(self) -> IntentType:
        return self.intent

    @property
    def extracted_entities(self) -> List[str]:
        return self.entities


@dataclass
class TelemetryEvent:
    timestamp: str
    question: str
    status: Optional[Status] = None
    event_type: Optional[str] = None
    doc_id: Optional[str] = None
    confidence_score: Optional[float] = None
    confidence_tier: Optional[str] = None
    rejection_reason: Optional[str] = None


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
    status: Status
    question: str
    answer: str
    citation: Optional[Citation]
    confidence_tier: ConfidenceTier
    confidence_score: float
    effective_threshold: float
    rejection_reason: Optional[str] = None
    escalation_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        def _convert(obj: Any) -> Any:
            """Recursively convert enum values to their string representations."""
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            if isinstance(obj, Enum):
                return obj.value
            return obj

        data = asdict(self)
        return _convert(data)
