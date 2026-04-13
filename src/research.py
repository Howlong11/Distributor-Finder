from __future__ import annotations

import hashlib
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from src.chunking import normalize_text
from src.config import AppConfig
from src.models import LeadSearchRequest, SearchEvidence


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class PublicWebResearcher:
    """Search and scrape public distributor evidence from the web."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.last_debug: Dict[str, List[Dict[str, str | int]]] = {"per_query": []}
        self.page_keywords = ["about", "contact", "brand", "distribution", "wholesale", "partner"]

    def build_queries(self, request: LeadSearchRequest) -> List[str]:
        base_queries = [
            f"comfortable shoe distributor {request.target_country}",
            f"{request.target_country} comfortable shoe wholesaler",
            f"{request.target_country} comfort footwear importer",
            f"{request.target_country} comfort footwear b2b wholesale",
        ]

        unique_queries: List[str] = []
        seen = set()
        for query in base_queries:
            normalized = query.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_queries.append(query)
        return unique_queries

    def search(self, request: LeadSearchRequest) -> List[SearchEvidence]:
        if not self.config.serpapi_api_key:
            raise ValueError("Missing SERPAPI_API_KEY. Add it to your .env file to search Google via SerpAPI.")

        self.last_debug = {"per_query": []}
        evidence: List[SearchEvidence] = []
        for query in self.build_queries(request):
            query_debug: Dict[str, str | int] = {
                "query": query,
                "source": "serpapi_google",
                "results": 0,
                "error": "",
            }
            try:
                results = self._serpapi_google_search(query)
                evidence.extend(results)
                query_debug["results"] = len(results)
            except requests.RequestException as exc:
                query_debug["error"] = str(exc)
            self.last_debug["per_query"].append(query_debug)

        unique: Dict[str, SearchEvidence] = OrderedDict()
        for item in evidence:
            unique.setdefault(item.url, item)
        return list(unique.values())

    def _serpapi_google_search(self, query: str) -> List[SearchEvidence]:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google",
                "q": query,
                "api_key": self.config.serpapi_api_key,
                "num": self.config.search_results_per_query,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=self.config.page_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()

        results: List[SearchEvidence] = []
        for item in body.get("organic_results", [])[: self.config.search_results_per_query]:
            title = item.get("title", "").strip()
            link = item.get("link", "").strip()
            snippet = item.get("snippet", "").strip()
            if not link or self._is_noise_result(title, snippet, link):
                continue
            page_excerpt = self._fetch_page_excerpt(link)
            results.append(
                SearchEvidence(
                    title=title,
                    url=link,
                    snippet=snippet,
                    page_excerpt=page_excerpt,
                )
            )
        return results

    def _fetch_page_excerpt(self, url: str) -> str:
        """Fetch a readable excerpt for a single page."""
        text, _, _ = self._fetch_page_details(url)
        return text[:2500]

    def scrape_company_pages(
        self,
        company_name: str,
        website: str,
        seed_urls: List[str] | None = None,
    ) -> List[Dict[str, str]]:
        """Scrape multiple relevant pages for a company website."""
        normalized_website = self._normalize_url(website)
        if not normalized_website:
            return []

        candidates = OrderedDict()
        candidates[normalized_website] = self._infer_page_type(normalized_website)
        for url in seed_urls or []:
            normalized = self._normalize_url(url)
            if not normalized:
                continue
            if self._same_domain(normalized_website, normalized):
                candidates.setdefault(normalized, self._infer_page_type(normalized))

        homepage_text, homepage_title, homepage_links = self._fetch_page_details(normalized_website)
        if homepage_text:
            candidates[normalized_website] = "homepage"
            for link in homepage_links:
                normalized_link = self._normalize_url(urljoin(normalized_website, link))
                if not normalized_link or normalized_link in candidates:
                    continue
                if not self._same_domain(normalized_website, normalized_link):
                    continue
                if any(keyword in normalized_link.lower() for keyword in self.page_keywords):
                    candidates[normalized_link] = self._infer_page_type(normalized_link)

        pages: List[Dict[str, str]] = []
        for url, page_type in list(candidates.items())[: self.config.max_company_pages]:
            cleaned_text, title, _ = self._fetch_page_details(url)
            if not cleaned_text or self._is_low_value_page(cleaned_text):
                continue
            pages.append(
                {
                    "company_name": company_name,
                    "url": url,
                    "page_title": title or homepage_title or company_name,
                    "page_type": page_type,
                    "cleaned_text": cleaned_text,
                    "evidence_snippet": cleaned_text[:280],
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                    "text_hash": hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest(),
                }
            )
        return pages

    def _fetch_page_details(self, url: str) -> Tuple[str, str, List[str]]:
        """Fetch cleaned text, title, and links for a page."""
        try:
            response = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=self.config.page_timeout_seconds,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = normalize_text(" ".join(soup.stripped_strings))
            title = normalize_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
            links = [
                anchor.get("href", "").strip()
                for anchor in soup.find_all("a", href=True)
                if anchor.get("href", "").strip()
            ]
            return text, title, links
        except requests.RequestException:
            return "", "", []

    def _is_noise_result(self, title: str, snippet: str, url: str) -> bool:
        combined = f"{title} {snippet} {url}".lower()
        noise_terms = [
            "directory",
            "directories",
            "marketplace",
            "marketplaces",
            "trade portal",
            "listing",
            "list of",
            "top 10",
            "top 20",
            "blog",
            "news",
            "article",
            "find distributors",
            "supplier discovery",
            "yellow pages",
            "linkedin",
            "facebook",
            "instagram",
            "wikipedia",
        ]
        return any(term in combined for term in noise_terms)

    def _is_low_value_page(self, text: str) -> bool:
        """Return whether the page text is too short or repetitive to index."""
        if len(text) < self.config.min_page_characters:
            return True
        words = text.lower().split()
        unique_ratio = len(set(words)) / max(1, len(words))
        return unique_ratio < 0.22

    def _normalize_url(self, url: str) -> str:
        """Normalize URLs to reduce duplicate pages."""
        if not url:
            return ""
        parsed = urlparse(url.strip())
        if not parsed.scheme:
            parsed = urlparse(f"https://{url.strip()}")
        normalized = parsed._replace(
            params="",
            query="",
            fragment="",
            netloc=parsed.netloc.lower(),
            path=parsed.path.rstrip("/") or "/",
        )
        return urlunparse(normalized)

    def _same_domain(self, root_url: str, candidate_url: str) -> bool:
        return urlparse(root_url).netloc == urlparse(candidate_url).netloc

    def _infer_page_type(self, url: str) -> str:
        lowered = url.lower()
        for keyword in self.page_keywords:
            if keyword in lowered:
                return keyword
        return "homepage"
