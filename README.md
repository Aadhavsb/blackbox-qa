# blackbox-qa

Agentic question-answering over public **NTSB aviation incident reports**.

> **Status: runs end-to-end.** Ingest → hybrid retrieval + rerank → tool-calling agent → self-hosted Langfuse tracing/scoring → a manually-triggered GitHub Actions eval pipeline are all built and green. Remaining work is a cloud deploy (phase 6).

## What it is

A natural-language question goes in (`"were there more engine failures on 737s or A320s since 2015?"`); a grounded, cited answer comes out. Under the hood:

- **One Postgres database** (via `pgvector`) holds the incident reports relationally, their narrative chunks as dense embeddings (HNSW index), and a `tsvector` full-text index — so the same store serves keyword search, vector search, and SQL aggregation.
- **Hybrid retrieval**: BM25-style keyword search (Postgres FTS) + dense vector search, fused with **Reciprocal Rank Fusion (RRF)**. Optional cross-encoder reranking stage (measured as an ablation).
- **A raw tool-calling agent** (no framework) that chooses between three tools at runtime:
  - `hybrid_search` — retrieve relevant report chunks
  - `sql_query` — read-only aggregate queries over the structured fields
  - `fetch_full_report` — pull a full report by ID
  - Weak-evidence answers trigger one bounded query-rewrite + re-retrieval retry, gated on the **cross-encoder retrieval score** (a calibrated signal) rather than the model's self-report.
- **Observability**: every request traced in self-hosted **Langfuse** (started by `make db-up`); LLM-as-judge scores posted back to the same traces via the Scores API.
- **Evaluation**: a frozen 25-query gold set drives a GitHub Actions pipeline with two triggers: a **Recall@5 regression gate on every PR** (deterministic, free, fast), and a manually-triggered mode for the LLM-as-judge tiers (slice and full).
- **`docker compose up`** runs the whole thing.

## Architecture

```
question
   │
   ▼
agent loop   (≤8 turns · ≤16 tool calls · 1 retry on weak retrieval evidence)
   │  picks one tool per turn
   ├──▶ hybrid_search       BM25 + dense → RRF → cross-encoder rerank
   ├──▶ sql_query           read-only, validated SELECT
   └──▶ fetch_full_report   full report by ev_id
                 │
                 ▼
        Postgres + pgvector   (HNSW vectors · tsvector full-text)
                 │
                 ▼
        grounded, cited answer

every run → Langfuse trace: per-turn token usage + out-of-band judge
scores (answer_quality · citation_match) on the same trace   (optional)
```

## Data

NTSB aviation accident data (1982–present) from <https://data.ntsb.gov/avdata/> — distributed as a Microsoft Access `.mdb` inside `avall.zip`. **Raw data is never committed.** `make ingest` downloads it and converts it to Postgres via `mdbtools`. Only the frozen gold set and a small (85-report: 25 gold + 60 distractors) CI fixture are committed.

## Quickstart

Dependencies are managed with [Poetry](https://python-poetry.org/) (`pyproject.toml` + `poetry.lock`).

```bash
poetry install                          # runtime + dev deps + Langfuse SDK

make db-up        # Postgres (pgvector) + self-hosted Langfuse stack
make ingest       # download NTSB data, convert .mdb, load + embed + index
make serve        # FastAPI app
# or:
poetry run blackbox-qa "your question here"
```

Prerequisites: Docker (Postgres) and `mdbtools` (`sudo apt install mdbtools`) for ingest. The agent needs an OpenAI-compatible LLM — copy `.env.example` to `.env` and add a (free) [Groq](https://console.groq.com) key (recommended; or Gemini/Ollama — any OpenAI-compatible endpoint works via env vars). The agent and judge use different models. Embeddings and the reranker run locally on CPU.

### The agent

A raw tool-calling loop (no framework) with three tools — `hybrid_search` (narrative search, reranked), `sql_query` (read-only `SELECT` over structured fields), and `fetch_full_report` (full report by `ev_id`). The loop is bounded (~8 turns + a total tool-call ceiling), validates tool arguments and feeds errors back for self-correction (unexpected tool failures are caught and returned, never crash the loop), and forces a text answer on the final turn (`tool_choice="none"`, with a repair step if the model still withholds text). The `sql_query` tool is guarded to single read-only `SELECT`s (rejected: multi-statement, DDL/DML, and side-effect functions like `pg_sleep`/`pg_read_file`) plus a DB-level read-only transaction and a least-privilege role. The LLM client retries on rate-limit (429 / Groq 413 TPM) with server-suggested backoff.

**Confidence gate.** The one query-rewrite retry fires on a *calibrated* signal: the top cross-encoder rerank score of the evidence the agent gathered. A low top score means nothing relevant was retrieved — a far more honest "should I retry?" input than the model's self-reported `CONFIDENCE` line, which tracks fluency, not evidence. The threshold is chosen against the gold set with a Youden's-J sweep (`python -m evals.run --mode calibrate`): across 25 queries the 3 failed retrievals top out at 3.21 while successes mostly sit far higher (mean ≈ 7.6), so the gate sits at **4.42** — catching every failed retrieval (TPR 1.0) at the cost of one needless retry on a genuinely-good answer (FPR 0.045). The self-report is still recorded alongside the score so the two can be compared. When no search ran (a SQL-only answer), the gate falls back to the self-report. (Calibration is on a small sample — n=25 with 3 failed retrievals — so treat 4.42 as a direction, not a tuned constant.)

### Observability (optional)

Self-hosted Langfuse (`make db-up` starts it alongside Postgres): each agent run is a trace, with a generation per LLM turn (token usage → cost) and a span per tool call. Quality is scored out-of-band and attached to the run's trace by `trace_id` via the Scores API — a deterministic `citation_match` (did it cite a gold `ev_id`?) and an LLM-as-judge `answer_quality` (0–1). Tracing is async/batched so it adds no latency to the request; set `LANGFUSE_ENABLED=false` in `.env` to disable. Run the judge-scored slice with:

```bash
poetry run python -m evals.run --mode judge-slice --limit 5
```

Example traces — the observation tree with per-turn token usage and both judge scores on one trace, plus a low-confidence failure case showing the gate — are in [`docs/tracing/`](docs/tracing/).

## Results

On a 2008–2009 NTSB slice — 3,000 reports / ~14k narrative chunks, 25-query hand-curated gold set:

| Stage | Recall@5 | MRR |
|---|---|---|
| BM25 keyword-only (naive baseline) | 0.04 | 0.04 |
| Hybrid (dense + BM25, RRF) | 0.88 | 0.75 |
| Hybrid + cross-encoder rerank | **0.88** | **0.88** |

BM25 alone nearly fails on this corpus (Recall@5 0.04) because the gold queries are paraphrased — exact-keyword matching breaks against natural-language reformulations. Adding dense vectors via RRF lifts Recall@5 to 0.88 (**+22× relative**). Reranking can't add documents the first stage missed, so Recall@5 holds; it reorders the pool, improving MRR by +0.13 (~17% relative). Reproduce:

```bash
poetry run python -m evals.run --ablation --k 5
```

Baselines committed at `evals/baseline.json` and `evals/ablation.json` (CI PR gate uses the hermetic fixture in `evals/fixtures/` instead — see `evals/README.md`).

**Full ingest** (30,646 reports / 125,213 chunks, same gold set): Recall@5 drops to **0.40** hybrid / **0.64** hybrid+rerank — more distractors at scale; rerank helps materially (+0.24 Recall@5). BM25-only stays at 0.04.

**End-to-end answer quality** (full 25-query gold set, LLM-as-judge, `evals/answers.json`):

| Metric | Score |
|---|---|
| `citation_match` rate | 0.84 |
| `answer_quality` mean | 0.856 |

`citation_match` is a lower bound — it fires only when the agent cites the exact gold `ev_id`; answers citing a different valid incident score 0 even when correct. `answer_quality` is a 0–1 judge score assessing answer accuracy and grounding. Reproduce:

```bash
poetry run python -m evals.run --mode full
```

## At 100x scale

At ~300x the data (~300k reports / ~1.4M chunks) the design holds because the heavy pieces are already isolated and swappable — scaling is mostly configuration, not a rewrite. The vector index is approximate ANN (HNSW, tunable `ef_search`); the embedder and cross-encoder are separate models behind env vars, so they move to a GPU/served endpoint by changing config; tracing is already async/batched with judge scoring out-of-band so observability adds no request latency; and the deterministic retrieval gate drops straight into a blocking PR check. The remaining knobs are standard ops, not redesign: bound the rerank pool, add connection pooling / read replicas for the SQL path, and partition the chunks table by year.

## Phases

1. Ingest + hybrid retrieval + gold set (Recall@5 / MRR measured) ✅
2. (1.5) Cross-encoder rerank stage, reported as ablation numbers ✅
3. Agent loop — 3 tools, bounded iterations, arg validation, confidence-retry ✅
4. Langfuse tracing + judge scores via Scores API ✅
5. CI eval pipeline — Recall@5 regression gate on every PR + manually-triggered judge tiers; calibrated retrieval-score confidence gate ✅
6. Deploy heavy components (Langfuse, embeddings, reranker, Postgres) to a VM / EC2
7. README as engineering doc — architecture diagram, measured numbers, failure modes, "at 100x scale", Langfuse trace screenshots ✅

## License

MIT — see [LICENSE](LICENSE).
