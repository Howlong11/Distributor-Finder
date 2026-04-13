# Global Distributor Finder

Streamlit app for researching and ranking potential comfort-footwear distributors in a target country, then generating outreach and reply drafts grounded in the lead data.

## What the application does

The app has three working flows:

1. Lead discovery
   Searches Google via SerpAPI using distributor-focused queries for a target country, fetches public page excerpts, and uses Gemini to filter down to plausible distributors.
2. Lead scoring
   Scores shortlisted distributors against the user's desired traits and explains the reasoning behind each score.
3. Commercial drafting
   Generates personalized outreach emails and follow-up replies based on the selected distributor and the evidence gathered during discovery.

## Current product behavior

- Country-based distributor search for comfort shoes / comfort footwear
- Public-web evidence collection from search results and company pages
- AI-assisted filtering to remove obvious noise such as directories, blogs, marketplaces, and social profiles
- AI-assisted scoring with justification bullets
- Outreach email drafting for a selected distributor
- Reply assistant that returns:
  - a draft response
  - an action plan
  - risk flags / commercial notes
- Gemini connection health check in the sidebar
- Search debug panel showing:
  - queries used
  - result counts
  - research engine debug info
  - raw Gemini responses for filtering and scoring
- Local indexing for discovered distributor websites
- Local retrieval search over previously scraped distributor evidence

## Tech stack

- Python 3
- Streamlit for the UI
- Google Gemini API for filtering, scoring, outreach generation, and reply handling
- SerpAPI Google Search API for web result retrieval
- `requests` for API and webpage fetching
- `beautifulsoup4` for extracting readable page text
- `python-dotenv` for local environment variable loading
- SQLite for local knowledge persistence

## Project structure

```text
.
|-- app.py
|-- requirements.txt
`-- src/
    |-- agent.py
    |-- config.py
    |-- gemini_client.py
    |-- models.py
    |-- research.py
    `-- utils.py
```

## How it works

1. The user enters a target country, desired distributor traits, and the number of leads to return.
2. The app builds a small set of search queries related to comfort-footwear distribution.
3. SerpAPI returns Google organic results for those queries.
4. The app fetches each result page and extracts a text excerpt where possible.
5. Gemini filters the raw evidence into likely real distributors only.
6. Users can index discovered distributors into a local knowledge base that stores scraped pages, chunks, and embeddings.
7. The app can reuse that local evidence before making a fresh SerpAPI call.
8. Gemini scores the remaining leads from 0 to 100 using retrieved evidence when available.
9. The user can then generate outreach or draft a response to an incoming distributor reply.

## Configuration

Create a `.env` file in the project root with:

```env
GEMINI_API_KEY=your_gemini_api_key
SERPAPI_API_KEY=your_serpapi_api_key
GEMINI_MODEL=gemini-2.0-flash
SEARCH_RESULTS_PER_QUERY=2
GEMINI_MAX_RETRIES=3
GEMINI_RETRY_DELAY_SECONDS=2.0
KNOWLEDGE_DB_PATH=data/knowledge.db
CHUNK_SIZE_WORDS=140
CHUNK_OVERLAP_WORDS=30
MAX_COMPANY_PAGES=5
MIN_PAGE_CHARACTERS=280
LOCAL_RETRIEVAL_TOP_K=5
```

Notes:

- `GEMINI_API_KEY` is required for the app to start.
- `SERPAPI_API_KEY` is required for lead discovery to search Google.
- `GEMINI_MODEL` defaults to `gemini-2.0-flash`.
- `SEARCH_RESULTS_PER_QUERY` controls how many organic results are pulled per query.
- `KNOWLEDGE_DB_PATH` controls where the local evidence store is created.

## Local setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Key modules

- `app.py`: Streamlit UI, tabs, forms, session state, and rendering
- `src/agent.py`: Main orchestration for search, filtering, scoring, outreach, and reply handling
- `src/research.py`: SerpAPI queries, noise filtering, page fetching, and evidence gathering
- `src/knowledge_store.py`: SQLite persistence for companies, pages, chunks, and embeddings
- `src/retrieval.py`: local semantic retrieval over indexed evidence
- `src/chunking.py`: text normalization and chunking utilities
- `src/embeddings.py`: deterministic local embeddings for chunk search
- `src/gemini_client.py`: Gemini API client, retries, and health checks
- `src/config.py`: Environment-based runtime configuration
- `src/models.py`: Dataclasses for requests, evidence, and distributor leads

## Current limitations

- Search coverage is intentionally narrow and currently uses a small fixed query set
- Results depend heavily on public web visibility and page accessibility
- Distributor qualification is AI-assisted and should still be reviewed by a human
- The product focus is currently tailored to comfortable shoes / comfort footwear
- Local retrieval quality depends on the quality of the scraped distributor pages

## Human review is still needed

- Confirm the company is a real distributor, wholesaler, importer, or channel partner
- Verify geography, category coverage, and commercial fit
- Review outreach messaging before sending
- Validate legal, pricing, regulatory, and contractual details outside the app
