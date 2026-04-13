from __future__ import annotations

from collections import OrderedDict
from typing import List, Optional

from src.embeddings import LocalEmbeddingModel
from src.knowledge_store import KnowledgeStore
from src.models import EvidenceSnippet, RetrievedChunk


class LocalRetriever:
    """Semantic retrieval over locally stored distributor evidence."""

    def __init__(self, store: KnowledgeStore, embedding_model: LocalEmbeddingModel) -> None:
        self.store = store
        self.embedding_model = embedding_model

    def search(
        self,
        query: str,
        top_k: int = 5,
        country: Optional[str] = None,
        company_name: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve top-k evidence chunks using semantic similarity plus metadata filters."""
        query_embedding = self.embedding_model.embed_text(query)
        return self.store.search_chunks(
            query_embedding=query_embedding,
            top_k=top_k,
            country=country,
            company_name=company_name,
        )

    def retrieve_company_facts(
        self,
        company_name: str,
        country: Optional[str],
        query: str,
        top_k: int = 4,
    ) -> List[EvidenceSnippet]:
        """Return deduplicated snippets for one company."""
        retrieved = self.search(
            query=query,
            top_k=max(top_k * 2, top_k),
            country=country,
            company_name=company_name,
        )
        unique = OrderedDict()
        for item in retrieved:
            key = (item.source_url, item.evidence_snippet)
            if key in unique:
                continue
            unique[key] = EvidenceSnippet(
                company_name=item.company_name,
                url=item.source_url,
                page_title=item.page_title,
                page_type=item.page_type,
                snippet=item.evidence_snippet,
                source="local_retrieval",
                retrieval_score=item.retrieval_score,
            )
            if len(unique) >= top_k:
                break
        return list(unique.values())

