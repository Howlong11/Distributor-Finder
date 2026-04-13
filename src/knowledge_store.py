from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple

from src.models import RetrievedChunk


class KnowledgeStore:
    """SQLite-backed store for scraped distributor pages and chunk embeddings."""

    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS companies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_name TEXT NOT NULL,
                    website TEXT NOT NULL UNIQUE,
                    country TEXT NOT NULL,
                    inferred_category TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_pages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    company_id INTEGER NOT NULL,
                    url TEXT NOT NULL UNIQUE,
                    page_title TEXT NOT NULL,
                    page_type TEXT NOT NULL,
                    scraped_at TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    evidence_snippet TEXT NOT NULL,
                    FOREIGN KEY(company_id) REFERENCES companies(id)
                );

                CREATE TABLE IF NOT EXISTS text_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    evidence_snippet TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY(page_id) REFERENCES source_pages(id)
                );

                CREATE INDEX IF NOT EXISTS idx_companies_country ON companies(country);
                CREATE INDEX IF NOT EXISTS idx_source_pages_company ON source_pages(company_id);
                CREATE INDEX IF NOT EXISTS idx_text_chunks_page ON text_chunks(page_id);
                """
            )

    def upsert_company(
        self,
        company_name: str,
        website: str,
        country: str,
        inferred_category: str,
        timestamp: str,
    ) -> int:
        """Insert or update a company and return its id."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM companies WHERE website = ?",
                (website,),
            ).fetchone()
            if row:
                connection.execute(
                    """
                    UPDATE companies
                    SET company_name = ?, country = ?, inferred_category = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (company_name, country, inferred_category, timestamp, row["id"]),
                )
                return int(row["id"])

            cursor = connection.execute(
                """
                INSERT INTO companies (
                    company_name, website, country, inferred_category, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (company_name, website, country, inferred_category, timestamp, timestamp),
            )
            return int(cursor.lastrowid)

    def upsert_page(
        self,
        company_id: int,
        url: str,
        page_title: str,
        page_type: str,
        scraped_at: str,
        cleaned_text: str,
        evidence_snippet: str,
        text_hash: str,
    ) -> Tuple[int, bool]:
        """Insert or update a source page and indicate whether content changed."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, text_hash FROM source_pages WHERE url = ?",
                (url,),
            ).fetchone()
            if row:
                changed = row["text_hash"] != text_hash
                connection.execute(
                    """
                    UPDATE source_pages
                    SET company_id = ?, page_title = ?, page_type = ?, scraped_at = ?,
                        cleaned_text = ?, evidence_snippet = ?, text_hash = ?
                    WHERE id = ?
                    """,
                    (
                        company_id,
                        page_title,
                        page_type,
                        scraped_at,
                        cleaned_text,
                        evidence_snippet,
                        text_hash,
                        row["id"],
                    ),
                )
                return int(row["id"]), bool(changed)

            cursor = connection.execute(
                """
                INSERT INTO source_pages (
                    company_id, url, page_title, page_type, scraped_at,
                    text_hash, cleaned_text, evidence_snippet
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    company_id,
                    url,
                    page_title,
                    page_type,
                    scraped_at,
                    text_hash,
                    cleaned_text,
                    evidence_snippet,
                ),
            )
            return int(cursor.lastrowid), True

    def replace_chunks(
        self,
        page_id: int,
        chunks: Sequence[Tuple[int, str, str, List[float]]],
    ) -> None:
        """Replace all chunks for a page."""
        with self._connect() as connection:
            connection.execute("DELETE FROM text_chunks WHERE page_id = ?", (page_id,))
            connection.executemany(
                """
                INSERT INTO text_chunks (page_id, chunk_index, chunk_text, evidence_snippet, embedding_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        page_id,
                        chunk_index,
                        chunk_text,
                        evidence_snippet,
                        json.dumps(embedding),
                    )
                    for chunk_index, chunk_text, evidence_snippet, embedding in chunks
                ],
            )

    def get_company_count(self, country: Optional[str] = None) -> int:
        """Return the number of stored companies, optionally filtered by country."""
        with self._connect() as connection:
            if country:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM companies WHERE lower(country) = lower(?)",
                    (country,),
                ).fetchone()
            else:
                row = connection.execute("SELECT COUNT(*) AS count FROM companies").fetchone()
            return int(row["count"] if row else 0)

    def get_index_stats(self) -> Dict[str, int]:
        """Return simple counts for the knowledge base."""
        with self._connect() as connection:
            company_count = connection.execute(
                "SELECT COUNT(*) AS count FROM companies"
            ).fetchone()["count"]
            page_count = connection.execute(
                "SELECT COUNT(*) AS count FROM source_pages"
            ).fetchone()["count"]
            chunk_count = connection.execute(
                "SELECT COUNT(*) AS count FROM text_chunks"
            ).fetchone()["count"]
        return {
            "companies": int(company_count),
            "pages": int(page_count),
            "chunks": int(chunk_count),
        }

    def search_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        country: Optional[str] = None,
        company_name: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Return chunk candidates for local semantic ranking."""
        conditions: List[str] = []
        parameters: List[str] = []

        if country:
            conditions.append("lower(c.country) = lower(?)")
            parameters.append(country)
        if company_name:
            conditions.append("lower(c.company_name) = lower(?)")
            parameters.append(company_name)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    c.company_name,
                    c.website AS company_url,
                    c.country,
                    c.inferred_category,
                    p.url AS source_url,
                    p.page_title,
                    p.page_type,
                    p.scraped_at,
                    ch.chunk_text,
                    ch.evidence_snippet,
                    ch.embedding_json
                FROM text_chunks ch
                JOIN source_pages p ON ch.page_id = p.id
                JOIN companies c ON p.company_id = c.id
                {where_clause}
                """,
                parameters,
            ).fetchall()

        ranked: List[RetrievedChunk] = []
        for row in rows:
            embedding = json.loads(row["embedding_json"])
            score = sum(a * b for a, b in zip(query_embedding, embedding))
            ranked.append(
                RetrievedChunk(
                    company_name=row["company_name"],
                    company_url=row["company_url"],
                    country=row["country"],
                    inferred_category=row["inferred_category"],
                    source_url=row["source_url"],
                    page_title=row["page_title"],
                    page_type=row["page_type"],
                    chunk_text=row["chunk_text"],
                    evidence_snippet=row["evidence_snippet"],
                    retrieval_score=float(score),
                    scraped_at=row["scraped_at"],
                )
            )

        ranked.sort(key=lambda item: item.retrieval_score, reverse=True)
        return ranked[:top_k]
