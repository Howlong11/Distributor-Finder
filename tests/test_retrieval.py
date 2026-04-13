from __future__ import annotations

import unittest
from pathlib import Path

from src.embeddings import LocalEmbeddingModel
from src.knowledge_store import KnowledgeStore
from src.retrieval import LocalRetriever


class RetrievalTests(unittest.TestCase):
    def test_country_filtered_retrieval_prefers_matching_company(self) -> None:
        db_path = Path("tests/.tmp_knowledge.db")
        if db_path.exists():
            db_path.unlink()

        store = KnowledgeStore(str(db_path))
        embedding_model = LocalEmbeddingModel(dimensions=96)
        retriever = LocalRetriever(store, embedding_model)

        try:
            company_id = store.upsert_company(
                company_name="Tokyo Comfort Trading",
                website="https://tokyocomfort.example",
                country="Japan",
                inferred_category="comfort_footwear_distributor",
                timestamp="2026-04-14T00:00:00+00:00",
            )
            page_id, _ = store.upsert_page(
                company_id=company_id,
                url="https://tokyocomfort.example/about",
                page_title="About",
                page_type="about",
                scraped_at="2026-04-14T00:00:00+00:00",
                cleaned_text="Tokyo Comfort Trading distributes walking shoes and comfort footwear across Japan.",
                evidence_snippet="Tokyo Comfort Trading distributes walking shoes and comfort footwear across Japan.",
                text_hash="hash-1",
            )
            chunks = [
                "Tokyo Comfort Trading distributes walking shoes and comfort footwear across Japan.",
            ]
            embeddings = embedding_model.embed_many(chunks)
            store.replace_chunks(
                page_id,
                [
                    (index, chunk, chunk[:120], embedding)
                    for index, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ],
            )

            other_id = store.upsert_company(
                company_name="Berlin Industrial Supply",
                website="https://berlinindustrial.example",
                country="Germany",
                inferred_category="industrial_supplier",
                timestamp="2026-04-14T00:00:00+00:00",
            )
            other_page_id, _ = store.upsert_page(
                company_id=other_id,
                url="https://berlinindustrial.example/about",
                page_title="About",
                page_type="about",
                scraped_at="2026-04-14T00:00:00+00:00",
                cleaned_text="Berlin Industrial Supply distributes factory safety equipment in Germany.",
                evidence_snippet="Berlin Industrial Supply distributes factory safety equipment in Germany.",
                text_hash="hash-2",
            )
            other_chunks = ["Berlin Industrial Supply distributes factory safety equipment in Germany."]
            other_embeddings = embedding_model.embed_many(other_chunks)
            store.replace_chunks(
                other_page_id,
                [
                    (index, chunk, chunk[:120], embedding)
                    for index, (chunk, embedding) in enumerate(zip(other_chunks, other_embeddings))
                ],
            )

            results = retriever.search(
                query="Find comfort footwear distributors in Japan",
                country="Japan",
                top_k=3,
            )

            self.assertEqual(1, len(results))
            self.assertEqual("Tokyo Comfort Trading", results[0].company_name)
        finally:
            if db_path.exists():
                db_path.unlink()


if __name__ == "__main__":
    unittest.main()
