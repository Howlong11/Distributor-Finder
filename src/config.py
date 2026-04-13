from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass
class AppConfig:
    gemini_api_key: str
    serpapi_api_key: str
    gemini_model: str = "gemini-2.0-flash"
    search_results_per_query: int = 2
    page_timeout_seconds: int = 12
    gemini_max_retries: int = 3
    gemini_retry_delay_seconds: float = 2.0
    knowledge_db_path: str = "data/knowledge.db"
    chunk_size_words: int = 140
    chunk_overlap_words: int = 30
    embedding_dimensions: int = 192
    max_company_pages: int = 5
    min_page_characters: int = 280
    local_retrieval_top_k: int = 5

    @classmethod
    def from_env(cls) -> "AppConfig":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY. Add it to a .env file or your shell environment.")

        return cls(
            gemini_api_key=api_key,
            serpapi_api_key=os.getenv("SERPAPI_API_KEY", "").strip(),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash",
            search_results_per_query=max(1, int(os.getenv("SEARCH_RESULTS_PER_QUERY", "2"))),
            gemini_max_retries=max(1, int(os.getenv("GEMINI_MAX_RETRIES", "3"))),
            gemini_retry_delay_seconds=max(
                0.5, float(os.getenv("GEMINI_RETRY_DELAY_SECONDS", "2.0"))
            ),
            knowledge_db_path=os.getenv("KNOWLEDGE_DB_PATH", "data/knowledge.db").strip()
            or "data/knowledge.db",
            chunk_size_words=max(60, int(os.getenv("CHUNK_SIZE_WORDS", "140"))),
            chunk_overlap_words=max(10, int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))),
            embedding_dimensions=max(64, int(os.getenv("EMBEDDING_DIMENSIONS", "192"))),
            max_company_pages=max(2, int(os.getenv("MAX_COMPANY_PAGES", "5"))),
            min_page_characters=max(120, int(os.getenv("MIN_PAGE_CHARACTERS", "280"))),
            local_retrieval_top_k=max(1, int(os.getenv("LOCAL_RETRIEVAL_TOP_K", "5"))),
        )
