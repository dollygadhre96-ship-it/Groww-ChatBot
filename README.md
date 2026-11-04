# Groww Mutual Fund FAQ Assistant

A facts-only RAG demo for HDFC Mutual Fund FAQs. It is not investment advice.

## Setup

1. Create a virtual environment in `.venv`.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your Groq API key.
4. Start the app from PowerShell with `./scripts/run_app.ps1`. This always uses the project virtual environment, even if another Python or Streamlit installation is on your PATH. To start it automatically at sign-in and recover it if it exits, run `./scripts/install_auto_start.ps1` once. Windows then checks the app every minute and restarts it when needed.

## Project status

Phases 1–9 are implemented: ingestion, chunking, embeddings, ChromaDB retrieval, input guardrails, the Streamlit chat interface, and Groq-grounded answer generation with citation and sentence-count validation.

## Data and secrets

Local scraped data and ChromaDB files are excluded from Git. Never commit `.env` or API keys.


## Phase 2 — Scraping source documents

The scraper reads **only the five explicit URLs** listed in `config/config.yaml`.
It does not crawl internal links or discover additional pages.

Run from the project root:

```bash
python -m src.ingestion.scraper
```

It saves fetched HTML under `data/raw/` and cleaned text plus source metadata as JSON
under `data/processed/`. Run scraper tests with:

```bash
pytest -q tests/test_scraper.py
```

## Phase 2 — Scrape the configured sources

The scraper reads **only the five exact URLs** in `config/config.yaml`. It does not
inspect anchors, discover links, crawl related pages, or follow internal links.

Run from the project root:

```bash
python -m src.ingestion.scraper
```

It writes:
- `data/raw/<source_id>.html` — fetched HTML snapshots
- `data/processed/<source_id>.json` — extracted text and source metadata
- `data/processed/scrape_report.json` — per-URL success/failure summary

A failed URL is reported while other configured URLs continue. Re-running the scraper
overwrites the same source files (stable IDs based on URL).

## Phase 3 — Document loading

Load the JSON documents created by Phase 2 and normalize them into `Document(text, metadata)` objects. The loader skips `scrape_report.json`, validates required fields, skips malformed/empty/duplicate records, and logs why a file was skipped.

Run from the project root:

```bash
python -m src.ingestion.loader
```

The loader reads `data/processed/*.json`; it does not make network requests or discover sources.

## Phase 4 — Chunking

The chunker accepts `Document` objects from Phase 3, packs paragraph blocks into
configurable character-sized chunks, splits oversized blocks near sentence/word
boundaries, applies overlap, and preserves source metadata. Chunk IDs are stable
for the same source/content/index/text.

Run a quick chunking check after loading documents:

```python
from src.ingestion.loader import load_documents
from src.chunking import chunk_documents

documents = load_documents()
chunks = chunk_documents(documents, chunk_size=800, chunk_overlap=120)
print(f"{len(documents)} documents -> {len(chunks)} chunks")
```

Chunk settings are in `config/config.yaml`. Current values are character counts,
not token counts.

## Phase 5 — Embedding generation

`EmbeddingService` uses Sentence Transformers (`sentence-transformers/all-MiniLM-L6-v2`
by default), normalizes vectors, and uses the same model for chunks and queries.

```python
from src.ingestion.loader import load_documents
from src.chunking import chunk_documents
from src.embeddings import EmbeddingService
chunks = chunk_documents(load_documents())
embedder = EmbeddingService()
records = embedder.embed_chunks(chunks)
query_vector = embedder.embed_query("What is the expense ratio?")
print(len(records), len(query_vector))
```
The first run may download the model. Run from the project root.

### Phase 6 — ChromaDB vector storage

`src/vector_store/store.py` provides `ChromaStore`: persistent local ChromaDB with cosine distance, chunk/vector upserts (stable IDs prevent duplicates), nearest-neighbor search, count, and explicit collection reset. Search results include ID, text, metadata, and `distance` (lower is more relevant). `tests/test_vector_store.py` covers upsert/search, persistence, validation, and reset.

```python
from src.vector_store import ChromaStore
store = ChromaStore("data/chroma", "hdfc_mf_faq", "sentence-transformers/all-MiniLM-L6-v2")
store.upsert_chunks(chunks, embeddings)
results = store.query(query_vector, top_k=5)
```

## Phase 7 — Retrieval logic

`Retriever` embeds each non-empty question with the configured `EmbeddingService`, searches the persistent Chroma collection using `top_k`, filters results by cosine **distance** (`distance <= relevance_threshold`; lower is closer), removes duplicate chunk text, and assembles source-attributed evidence context. It returns `RetrievalResponse` with retrieved chunks, distance/relevance values, context, source links, and a `no_evidence` flag for safe fallback handling.

Example:

```python
from src.embeddings.service import EmbeddingService
from src.vector_store.store import ChromaStore
from src.retrieval import Retriever

store = ChromaStore("data/chroma", "hdfc_mf_faq")
embedder = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2")
retriever = Retriever(store, embedder, top_k=5, relevance_threshold=0.45)
result = retriever.retrieve("What is the expense ratio?")
if result.no_evidence:
    print("I couldn't find this in the available sources.")
else:
    print(result.context)
    print(result.sources)
```

Run retrieval unit tests with `pytest -q tests/test_retriever.py`. Retrieval does not call the LLM; answer generation and output validation are later pipeline stages.

### Retrieval behavior and outputs

- Query embedding uses the same configured Sentence Transformer as document indexing.
- Chroma is queried for configured `top_k`; cosine **distance** is filtered using `distance <= relevance_threshold` (smaller is better).
- Exact normalized duplicate chunk text is removed.
- Context includes evidence labels with source title and URL and is capped at `max_context_chars` (12,000 by default). Chunks that do not fit in full are omitted rather than truncated.
- `RetrievalResponse` exposes `query`, `chunks` (ID/text/metadata/distance/relevance), `context`, unique `sources`, `no_evidence`, and the applied `threshold`.
- Retrieval is intentionally separate from LLM answer generation. `no_evidence=True` means use the safe no-evidence response.

Run the focused tests: `pytest -q tests/test_retriever.py`.


## Phase 8 — Guardrails and Streamlit UI

### Input / output guardrails
`src/guardrails/rules.py` provides deterministic input checks for empty/oversized
queries, common PII formats, advice requests, prompt injection attempts, and
unsupported topics. `validate_answer` checks non-empty output, sentence count, and
presence of a retrieved source URL. These are baseline rules and should be reviewed
against a broader golden set before production use.

Run:
```bash
pytest -q tests/test_guardrails.py
```

### Streamlit chat interface
Run from project root:
```bash
./scripts/run_app.ps1
```
The UI includes a branded welcome screen, three suggested questions, chat history,
clickable source cards, clear chat, and a discreet facts-only footer. Pipeline diagnostics
remain internal and are not exposed in the user interface.

## Phase 9 — Groq-grounded answer generation

`src/generation.py` uses the configured Groq chat-completions model and the local `GROQ_API_KEY`. It answers only from retrieved evidence and asks the model to cite source URLs. The app checks citations and the configured sentence limit, then uses a safe fallback if generation fails or validation rejects the answer.

## Build the local ChromaDB index

From the project root, run the following after installing dependencies:

```powershell
python -m src.ingestion.scraper
python scripts/index_documents.py
./scripts/run_app.ps1
```

The indexing script loads JSON documents from `data/processed`, applies the configured chunk size/overlap, creates embeddings with the configured sentence-transformers model, and upserts them into the persistent ChromaDB collection. It does not delete existing records. Re-running it updates records with stable IDs.
