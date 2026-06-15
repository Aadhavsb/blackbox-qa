# blackbox-qa

Agentic question-answering over public **NTSB aviation incident reports**.

> **Status: scaffolding / work in progress.** Concept and architecture are locked; build is in progress phase by phase. Each phase leaves the repo in a runnable state.

## What it is

A natural-language question goes in (`"were there more engine failures on 737s or A320s since 2015?"`); a grounded, cited answer comes out. Under the hood:

- **One Postgres database** (via `pgvector`) holds the incident reports relationally, their narrative chunks as dense embeddings (HNSW index), and a `tsvector` full-text index — so the same store serves keyword search, vector search, and SQL aggregation.
- **Hybrid retrieval**: BM25-style keyword search (Postgres FTS) + dense vector search, fused with **Reciprocal Rank Fusion (RRF)**. Optional cross-encoder reranking stage (measured as an ablation).
- **A raw tool-calling agent** (no framework) that chooses between three tools at runtime:
  - `hybrid_search` — retrieve relevant report chunks
  - `sql_query` — read-only aggregate queries over the structured fields
  - `fetch_full_report` — pull a full report by ID
  - Low-confidence answers trigger one bounded query-rewrite + re-retrieval retry.
- **Observability**: every request traced in self-hosted **Langfuse** (optional Compose profile); LLM-as-judge scores posted back to the same traces via the Scores API.
- **Evaluation**: a frozen gold set drives a **manually-triggered** GitHub Actions pipeline (parameterized run modes: deterministic Recall@k, a temp-0 judge slice, a full frontier-judge run). Architected to drop in as a merge-blocking retrieval-regression gate, but not currently wired as one.
- **`docker compose up`** runs the whole thing.

## Architecture

```
question
   │
   ▼
┌─────────────┐   tool calls   ┌──────────────────────────────┐
│ agent loop  │ ─────────────▶ │ hybrid_search │ sql_query │ … │
│ (≤8 turns)  │ ◀───────────── │   (Postgres + pgvector)      │
└─────────────┘   tool results └──────────────────────────────┘
   │                                   ▲
   │ grounded answer                   │ traces + scores
   ▼                                   ▼
 client                          Langfuse (optional)
```

## Data

NTSB aviation accident data (1982–present) from <https://data.ntsb.gov/avdata/> — distributed as a Microsoft Access `.mdb` inside `avall.zip`. **Raw data is never committed.** `make ingest` downloads it and converts it to Postgres via `mdbtools`. Only the frozen gold set and a small (~20-report) CI fixture are committed.

## Quickstart

Dependencies are managed with [Poetry](https://python-poetry.org/) (`pyproject.toml` + `poetry.lock`).

```bash
poetry install                          # runtime + dev deps
# poetry install --extras observability # + Langfuse

make db-up        # start Postgres (pgvector) via docker compose
make ingest       # download NTSB data, convert .mdb, load + embed + index
make serve        # FastAPI app
# or:
poetry run blackbox-qa "your question here"
```

Prerequisites: Docker (Postgres) and `mdbtools` (`sudo apt install mdbtools`) for ingest.

## Phases

1. Ingest + hybrid retrieval + gold set (Recall@5 / MRR measured)
2. (1.5) Cross-encoder rerank stage, reported as ablation numbers
3. Agent loop — 3 tools, bounded iterations, arg validation, confidence-retry
4. Langfuse tracing + judge scores via Scores API
5. CI eval pipeline (GitHub Actions + pgvector service container, manually triggered)
6. README as engineering doc — measured numbers, failure modes, "at 100x scale"

## License

MIT — see [LICENSE](LICENSE).
