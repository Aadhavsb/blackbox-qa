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
  - Weak-evidence answers trigger one bounded query-rewrite + re-retrieval retry, gated on the **cross-encoder retrieval score** (a calibrated signal) rather than the model's self-report.
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

Prerequisites: Docker (Postgres) and `mdbtools` (`sudo apt install mdbtools`) for ingest. The agent needs an OpenAI-compatible LLM — copy `.env.example` to `.env` and add a (free) [Groq](https://console.groq.com) key (recommended; or Gemini/Ollama — any OpenAI-compatible endpoint works via env vars). The agent and judge use different models. Embeddings and the reranker run locally on CPU.

### The agent

A raw tool-calling loop (no framework) with three tools — `hybrid_search` (narrative search, reranked), `sql_query` (read-only `SELECT` over structured fields), and `fetch_full_report` (full report by `ev_id`). The loop is bounded (~8 turns + a total tool-call ceiling), validates tool arguments and feeds errors back for self-correction (unexpected tool failures are caught and returned, never crash the loop), and forces a text answer on the final turn (`tool_choice="none"`, with a repair step if the model still withholds text). The `sql_query` tool is guarded to single read-only `SELECT`s (rejected: multi-statement, DDL/DML, and side-effect functions like `pg_sleep`/`pg_read_file`) plus a DB-level read-only transaction and a least-privilege role. The LLM client retries on rate-limit (429 / Groq 413 TPM) with server-suggested backoff.

**Confidence gate.** The one query-rewrite retry fires on a *calibrated* signal: the top cross-encoder rerank score of the evidence the agent gathered. A low top score means nothing relevant was retrieved — a far more honest "should I retry?" input than the model's self-reported `CONFIDENCE` line, which tracks fluency, not evidence. The threshold is chosen against the gold set (`python -m evals.run --mode calibrate`): failed and successful retrievals separate cleanly (failures ≤ 3.21, successes ≥ 5.80), so the gate sits at **4.5**. The self-report is still recorded alongside the score so the two can be compared. When no search ran (a SQL-only answer), the gate falls back to the self-report.

### Observability (optional)

Self-hosted Langfuse, behind a Compose profile (`docker compose --profile observability up`): each agent run is a trace, with a generation per LLM turn (token usage → cost) and a span per tool call. Quality is scored out-of-band and attached to the run's trace by `trace_id` via the Scores API — a deterministic `citation_match` (did it cite a gold `ev_id`?) and an LLM-as-judge `answer_quality` (0–1). Tracing is async/batched so it adds no latency to the request; when `LANGFUSE_ENABLED=false` (default) all of it is a no-op. Run the judge-scored slice with:

```bash
poetry run python -m evals.run --mode judge-slice --limit 5
```

## Results

On a 2008 NTSB slice — 3,000 reports / ~14k narrative chunks, 12-query hand-curated gold set:

| Stage | Recall@5 | MRR |
|---|---|---|
| Hybrid (dense + FTS, RRF) | 0.75 | 0.64 |
| Hybrid + cross-encoder rerank | 0.75 | **0.75** |

Reranking can't add documents the first stage missed, so Recall@5 is unchanged; it reorders the candidate pool, lifting the right report higher and improving **MRR +0.11 (~17% relative)**. Queries are paraphrased so this reflects retrieval quality, not memorization. Reproduce:

```bash
poetry run python -m evals.run --mode retrieval --ablation --k 5
```

Baseline committed at `evals/baseline.json`, rerank ablation at `evals/ablation.json`.

## Phases

1. Ingest + hybrid retrieval + gold set (Recall@5 / MRR measured) ✅
2. (1.5) Cross-encoder rerank stage, reported as ablation numbers ✅
3. Agent loop — 3 tools, bounded iterations, arg validation, confidence-retry ✅
4. Langfuse tracing + judge scores via Scores API ✅
5. CI eval pipeline (GitHub Actions + pgvector service container, manually triggered)
6. Deploy heavy components (Langfuse, embeddings, reranker, Postgres) to a VM / EC2
7. README as engineering doc — measured numbers, failure modes, "at 100x scale"

## License

MIT — see [LICENSE](LICENSE).
