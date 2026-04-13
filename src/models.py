from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LeadSearchRequest:
    company_name: str
    target_country: str
    product_focus: str
    desired_traits: List[str]
    max_leads: int = 5

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SearchEvidence:
    title: str
    url: str
    snippet: str
    page_excerpt: str = ""

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceSnippet:
    company_name: str
    url: str
    page_title: str
    page_type: str
    snippet: str
    source: str
    retrieval_score: float = 0.0

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedChunk:
    company_name: str
    company_url: str
    country: str
    inferred_category: str
    source_url: str
    page_title: str
    page_type: str
    chunk_text: str
    evidence_snippet: str
    retrieval_score: float
    scraped_at: str

    def model_dump(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DistributorLead:
    company_name: str
    country: str
    website: str
    summary: str
    score: float
    score_justifications: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    evidence_snippets: List[EvidenceSnippet] = field(default_factory=list)
    provenance: str = "fresh_web"
    request_context: Optional[LeadSearchRequest] = None

    def __post_init__(self) -> None:
        if isinstance(self.request_context, dict):
            self.request_context = LeadSearchRequest(**self.request_context)
        self.evidence_snippets = [
            item if isinstance(item, EvidenceSnippet) else EvidenceSnippet(**item)
            for item in self.evidence_snippets
        ]

    def model_dump(self) -> Dict[str, Any]:
        data = asdict(self)
        data["request_context"] = (
            self.request_context.model_dump() if self.request_context else None
        )
        return data
