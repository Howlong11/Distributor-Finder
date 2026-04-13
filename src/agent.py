from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List

from src.chunking import chunk_text
from src.config import AppConfig
from src.embeddings import LocalEmbeddingModel
from src.gemini_client import GeminiClient
from src.knowledge_store import KnowledgeStore
from src.models import DistributorLead, EvidenceSnippet, LeadSearchRequest, SearchEvidence
from src.research import PublicWebResearcher
from src.retrieval import LocalRetriever


class DistributorAgent:
    """Application orchestrator for distributor discovery, retrieval, and drafting."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.gemini = GeminiClient(config)
        self.researcher = PublicWebResearcher(config)
        self.knowledge_store = KnowledgeStore(config.knowledge_db_path)
        self.embedding_model = LocalEmbeddingModel(config.embedding_dimensions)
        self.retriever = LocalRetriever(self.knowledge_store, self.embedding_model)
        self.last_search_debug: Dict[str, Any] = {}

    def retrieve_country_results(self, request: LeadSearchRequest) -> List[SearchEvidence]:
        """Run the existing SerpApi-based discovery flow."""
        evidence = self.researcher.search(request)
        self.last_search_debug = {
            "request": request.model_dump(),
            "queries": self.researcher.build_queries(request),
            "research_debug": self.researcher.last_debug,
            "retrieved_result_count": len(evidence),
            "filter_raw_response": "",
            "score_raw_response": "",
            "filtered_count": 0,
            "scored_count": 0,
            "result_source": "fresh_web_search",
        }
        return evidence

    def filter_real_distributors(
        self, request: LeadSearchRequest, raw_evidence: List[SearchEvidence]
    ) -> List[DistributorLead]:
        if not self.last_search_debug:
            self.last_search_debug = {
                "request": request.model_dump(),
                "queries": self.researcher.build_queries(request),
                "research_debug": self.researcher.last_debug,
                "retrieved_result_count": len(raw_evidence),
                "filter_raw_response": "",
                "score_raw_response": "",
                "filtered_count": 0,
                "scored_count": 0,
            }
        if not raw_evidence:
            return []

        schema = {
            "type": "OBJECT",
            "properties": {
                "leads": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "company_name": {"type": "STRING"},
                            "country": {"type": "STRING"},
                            "website": {"type": "STRING"},
                            "summary": {"type": "STRING"},
                            "evidence": {"type": "ARRAY", "items": {"type": "STRING"}},
                            "sources": {"type": "ARRAY", "items": {"type": "STRING"}},
                        },
                        "required": [
                            "company_name",
                            "country",
                            "website",
                            "summary",
                            "evidence",
                            "sources",
                        ],
                    },
                }
            },
            "required": ["leads"],
        }
        evidence_text = self._format_evidence(raw_evidence)
        prompt = f"""
You are filtering public web search results to keep only real comfortable shoe distributors in {request.target_country}.

Goal:
- Keep only real distributors, wholesalers, importers, or official channel partners for comfortable shoes, comfort footwear, walking shoes, orthopedic comfort shoes, or adjacent comfort-footwear categories in {request.target_country}.

Exclude:
- blogs
- articles
- directories
- marketplaces
- "find distributors" websites
- sourcing portals
- software companies
- logistics providers
- consultants
- retailers with no sign of distribution or wholesale activity

Use only the public evidence below.

Evidence:
{evidence_text}

Return valid JSON only with:
{{
  "leads": [
    {{
      "company_name": "...",
      "country": "...",
      "website": "...",
      "summary": "...",
      "evidence": ["..."],
      "sources": ["..."]
    }}
  ]
}}

Return only companies that are plausibly real distributors based on the evidence.
"""
        payload = self.gemini.generate_json(prompt, schema)
        self.last_search_debug["filter_raw_response"] = self.gemini.last_response_text

        leads = [
            DistributorLead(
                company_name=item.get("company_name", "Unknown distributor"),
                country=item.get("country", request.target_country),
                website=item.get("website", ""),
                summary=item.get("summary", ""),
                score=0.0,
                score_justifications=[],
                evidence=item.get("evidence", []),
                sources=item.get("sources", []),
                evidence_snippets=[],
                provenance="fresh_web",
                request_context=request,
            )
            for item in payload.get("leads", [])[: request.max_leads]
        ]
        self.last_search_debug["filtered_count"] = len(leads)
        return leads

    def score_distributors(
        self, request: LeadSearchRequest, filtered_leads: List[DistributorLead]
    ) -> List[DistributorLead]:
        """Score leads using retrieved local evidence when available."""
        if not self.last_search_debug:
            self.last_search_debug = {
                "request": request.model_dump(),
                "queries": [],
                "research_debug": {},
                "retrieved_result_count": 0,
                "filter_raw_response": "",
                "score_raw_response": "",
                "filtered_count": len(filtered_leads),
                "scored_count": 0,
                "result_source": "unknown",
            }
        if not filtered_leads:
            return []

        enriched_leads = self.enrich_leads_with_local_evidence(filtered_leads, request)
        schema = {
            "type": "OBJECT",
            "properties": {
                "leads": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "company_name": {"type": "STRING"},
                            "score": {"type": "NUMBER"},
                            "score_justifications": {
                                "type": "ARRAY",
                                "items": {"type": "STRING"},
                            },
                        },
                        "required": ["company_name", "score", "score_justifications"],
                    },
                }
            },
            "required": ["leads"],
        }

        distributor_text = "\n\n".join(
            [
                (
                    f"Company: {lead.company_name}\n"
                    f"Country: {lead.country}\n"
                    f"Website: {lead.website}\n"
                    f"Summary: {lead.summary}\n"
                    f"Evidence: {lead.evidence}\n"
                    f"Sources: {lead.sources}\n"
                    "Retrieved evidence:\n"
                    + "\n".join(
                        [
                            (
                                f"- [{snippet.page_type}] {snippet.snippet} "
                                f"(source: {snippet.url})"
                            )
                            for snippet in lead.evidence_snippets
                        ]
                    )
                )
                for lead in enriched_leads
            ]
        )
        prompt = f"""
You are scoring real distributors for a shoe company.

Target country: {request.target_country}
Brand seeking distributors: {request.company_name}
Desired distributor features: {request.desired_traits}

Use only the public evidence below. Do not invent facts. Score each distributor from 0 to 100 based only on how well the publicly available evidence matches the desired features.

Distributors to score:
{distributor_text}

Return valid JSON only with:
{{
  "leads": [
    {{
      "company_name": "...",
      "score": 0,
      "score_justifications": ["..."]
    }}
  ]
}}
"""
        payload = self.gemini.generate_json(prompt, schema)
        self.last_search_debug["score_raw_response"] = self.gemini.last_response_text

        score_map = {
            item.get("company_name", ""): {
                "score": float(item.get("score", 0)),
                "score_justifications": item.get("score_justifications", []),
            }
            for item in payload.get("leads", [])
        }

        scored: List[DistributorLead] = []
        for lead in enriched_leads:
            score_data = score_map.get(lead.company_name, {})
            scored.append(
                DistributorLead(
                    company_name=lead.company_name,
                    country=lead.country,
                    website=lead.website,
                    summary=lead.summary,
                    score=float(score_data.get("score", 0)),
                    score_justifications=score_data.get("score_justifications", []),
                    evidence=lead.evidence,
                    sources=lead.sources,
                    evidence_snippets=lead.evidence_snippets,
                    provenance=lead.provenance,
                    request_context=request,
                )
            )

        scored.sort(key=lambda lead: lead.score, reverse=True)
        self.last_search_debug["scored_count"] = len(scored)
        return scored[: request.max_leads]

    def find_and_rank_distributors(
        self,
        request: LeadSearchRequest,
        use_local_first: bool = True,
    ) -> List[DistributorLead]:
        """Find and rank distributors, preferring local retrieval when available."""
        if use_local_first:
            local_leads = self.find_local_distributors(request)
            if local_leads:
                self.last_search_debug = {
                    "request": request.model_dump(),
                    "queries": [],
                    "research_debug": {},
                    "retrieved_result_count": 0,
                    "filter_raw_response": "Local retrieval reused indexed distributor evidence.",
                    "score_raw_response": "",
                    "filtered_count": len(local_leads),
                    "scored_count": len(local_leads),
                    "result_source": "local_retrieval",
                }
                return self.score_distributors(request, local_leads)

        raw_evidence = self.retrieve_country_results(request)
        filtered = self.filter_real_distributors(request, raw_evidence)
        return self.score_distributors(request, filtered)

    def get_gemini_health(self) -> Dict[str, Any]:
        return self.gemini.health_check()

    def get_knowledge_stats(self) -> Dict[str, int]:
        """Return counts for locally indexed distributor evidence."""
        return self.knowledge_store.get_index_stats()

    def find_local_distributors(self, request: LeadSearchRequest) -> List[DistributorLead]:
        """Return candidate distributors from the local knowledge base."""
        query = self._build_retrieval_query(request)
        retrieved = self.retriever.search(
            query=query,
            top_k=max(request.max_leads * 4, self.config.local_retrieval_top_k),
            country=request.target_country,
        )
        grouped: Dict[str, List[EvidenceSnippet]] = defaultdict(list)
        urls: Dict[str, str] = {}
        for item in retrieved:
            grouped[item.company_name].append(
                EvidenceSnippet(
                    company_name=item.company_name,
                    url=item.source_url,
                    page_title=item.page_title,
                    page_type=item.page_type,
                    snippet=item.evidence_snippet,
                    source="local_retrieval",
                    retrieval_score=item.retrieval_score,
                )
            )
            urls[item.company_name] = item.company_url

        leads: List[DistributorLead] = []
        for company_name, snippets in grouped.items():
            deduped_sources = list(OrderedDict((snippet.url, None) for snippet in snippets).keys())
            deduped_evidence = list(
                OrderedDict((snippet.snippet, None) for snippet in snippets).keys()
            )
            leads.append(
                DistributorLead(
                    company_name=company_name,
                    country=request.target_country,
                    website=urls.get(company_name, ""),
                    summary=deduped_evidence[0] if deduped_evidence else "Indexed company evidence available.",
                    score=0.0,
                    score_justifications=[],
                    evidence=deduped_evidence[:4],
                    sources=deduped_sources[:4],
                    evidence_snippets=snippets[:4],
                    provenance="local_retrieval",
                    request_context=request,
                )
            )
        return leads[: request.max_leads]

    def index_distributor_leads(self, leads: List[DistributorLead]) -> Dict[str, int]:
        """Scrape and locally index evidence for discovered distributors."""
        indexed_companies = 0
        indexed_pages = 0
        indexed_chunks = 0

        for lead in leads:
            if not lead.website:
                continue
            pages = self.researcher.scrape_company_pages(
                company_name=lead.company_name,
                website=lead.website,
                seed_urls=lead.sources,
            )
            if not pages:
                continue
            company_id = self.knowledge_store.upsert_company(
                company_name=lead.company_name,
                website=lead.website,
                country=lead.country,
                inferred_category="comfort_footwear_distributor",
                timestamp=pages[0]["scraped_at"],
            )
            indexed_companies += 1
            for page in pages:
                page_id, changed = self.knowledge_store.upsert_page(
                    company_id=company_id,
                    url=page["url"],
                    page_title=page["page_title"],
                    page_type=page["page_type"],
                    scraped_at=page["scraped_at"],
                    cleaned_text=page["cleaned_text"],
                    evidence_snippet=page["evidence_snippet"],
                    text_hash=page["text_hash"],
                )
                indexed_pages += 1
                if not changed:
                    continue

                chunks = chunk_text(
                    page["cleaned_text"],
                    chunk_size=self.config.chunk_size_words,
                    overlap=self.config.chunk_overlap_words,
                )
                embeddings = self.embedding_model.embed_many(chunks)
                self.knowledge_store.replace_chunks(
                    page_id,
                    [
                        (index, chunk, chunk[:240], embedding)
                        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                    ],
                )
                indexed_chunks += len(chunks)

        return {
            "companies": indexed_companies,
            "pages": indexed_pages,
            "chunks": indexed_chunks,
        }

    def enrich_leads_with_local_evidence(
        self,
        leads: List[DistributorLead],
        request: LeadSearchRequest,
    ) -> List[DistributorLead]:
        """Attach local retrieved evidence snippets to leads when available."""
        enriched: List[DistributorLead] = []
        for lead in leads:
            snippets = self.retriever.retrieve_company_facts(
                company_name=lead.company_name,
                country=lead.country or request.target_country,
                query=self._build_retrieval_query(request),
                top_k=self.config.local_retrieval_top_k,
            )
            evidence = list(OrderedDict((item, None) for item in lead.evidence).keys())
            sources = list(OrderedDict((item, None) for item in lead.sources).keys())
            for snippet in snippets:
                if snippet.snippet not in evidence:
                    evidence.append(snippet.snippet)
                if snippet.url not in sources:
                    sources.append(snippet.url)
            provenance = lead.provenance
            if snippets and provenance == "fresh_web":
                provenance = "fresh_web+local_retrieval"
            enriched.append(
                DistributorLead(
                    company_name=lead.company_name,
                    country=lead.country,
                    website=lead.website,
                    summary=lead.summary,
                    score=lead.score,
                    score_justifications=lead.score_justifications,
                    evidence=evidence,
                    sources=sources,
                    evidence_snippets=snippets or lead.evidence_snippets,
                    provenance=provenance,
                    request_context=lead.request_context or request,
                )
            )
        return enriched

    def search_local_knowledge(
        self,
        query: str,
        country: str = "",
        top_k: int = 5,
    ) -> List[EvidenceSnippet]:
        """Search the local knowledge base for evidence snippets."""
        return [
            EvidenceSnippet(
                company_name=item.company_name,
                url=item.source_url,
                page_title=item.page_title,
                page_type=item.page_type,
                snippet=item.evidence_snippet,
                source="local_retrieval",
                retrieval_score=item.retrieval_score,
            )
            for item in self.retriever.search(
                query=query,
                top_k=top_k,
                country=country or None,
            )
        ]

    def generate_outreach(
        self,
        lead: DistributorLead,
        brand_name: str,
        sender_name: str,
        sender_role: str,
        outreach_goal: str,
    ) -> str:
        facts = self.retriever.retrieve_company_facts(
            company_name=lead.company_name,
            country=lead.country,
            query=(
                f"{lead.company_name} distributor profile {lead.country} "
                f"{outreach_goal} comfort footwear"
            ),
            top_k=self.config.local_retrieval_top_k,
        )
        fact_lines = "\n".join(
            [
                f"- {fact.snippet} (source: {fact.url})"
                for fact in facts
            ]
        )
        prompt = f"""
You are writing a commercially credible outreach email for a shoe company seeking a distributor partnership.

Brand: {brand_name}
Sender name: {sender_name}
Sender role: {sender_role}
Goal: {outreach_goal}

Target distributor:
- Company: {lead.company_name}
- Country: {lead.country}
- Website: {lead.website}
- Summary: {lead.summary}
- Score reasons: {lead.score_justifications}
- Evidence: {lead.evidence}
- Retrieved facts: {fact_lines or "No indexed local evidence available."}

Write a concise outreach email with:
1. A subject line
2. A short greeting
3. A body that references the distributor's apparent strengths from the evidence
4. A realistic partnership angle for a shoe brand
5. A direct but polite call to action

Avoid fake claims, invented numbers, and exaggerated flattery.
Only mention facts supported by the provided evidence and retrieved facts.
"""
        return self.gemini.generate_text(prompt, temperature=0.7)

    def respond_to_reply(self, lead: DistributorLead, distributor_reply: str) -> Dict[str, Any]:
        schema = {
            "type": "OBJECT",
            "properties": {
                "response": {"type": "STRING"},
                "action_plan": {"type": "ARRAY", "items": {"type": "STRING"}},
                "notes": {"type": "ARRAY", "items": {"type": "STRING"}},
            },
            "required": ["response", "action_plan", "notes"],
        }
        prompt = f"""
You are helping a shoe company respond to a distributor.

Distributor profile:
- Company: {lead.company_name}
- Country: {lead.country}
- Summary: {lead.summary}
- Qualification reasons: {lead.score_justifications}

Distributor reply:
{distributor_reply}

Return JSON with:
- response: a polished draft reply
- action_plan: short next-step bullets for the human team
- notes: risks, open questions, or commercial cautions

Keep the response commercially credible and avoid inventing facts.
Format the response like a real business email with paragraph breaks between greeting, body paragraphs, and closing.
"""
        result = self.gemini.generate_json(prompt, schema)
        result["response"] = self._format_email_text(result.get("response", ""))
        return result

    def _format_evidence(self, raw_evidence: List[SearchEvidence]) -> str:
        return "\n\n".join(
            [
                (
                    f"Title: {item.title}\n"
                    f"URL: {item.url}\n"
                    f"Snippet: {item.snippet}\n"
                    f"Page excerpt: {item.page_excerpt}"
                )
                for item in raw_evidence[:25]
            ]
        )

    def _format_email_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if "\n" not in text:
            text = re.sub(r"(?<=[.!?])\s+(?=[A-Z])", "\n\n", text)
        return text

    def _build_retrieval_query(self, request: LeadSearchRequest) -> str:
        desired = ", ".join(request.desired_traits)
        return (
            f"{request.product_focus} distributor in {request.target_country}. "
            f"Desired traits: {desired}."
        )
