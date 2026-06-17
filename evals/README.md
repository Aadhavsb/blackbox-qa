# Evaluation

Frozen gold set + a parameterized runner.

## Gold set

- `gold/retrieval_gold.example.jsonl` — committed template showing the format.
- `gold/retrieval_gold.jsonl` — the real committed gold set (LLM-drafted, hand-verified). One JSON object per line; each has a `query` and either `ev_id` (single gold report) or `ev_ids` (list). `#`/blank lines ignored.

Populate it after ingest so the `ev_id`s point at real `reports.ev_id` values.

## Runner

```bash
python -m evals.run --mode retrieval --k 5
# gate-style compare against a committed baseline:
python -m evals.run --mode retrieval --baseline evals/baseline.json --max-drop 0.01
# choose the confidence-gate threshold from the gold set (no LLM):
python -m evals.run --mode calibrate --out evals/confidence_calibration.json
```

- `metrics.py` — pure Recall@k / MRR (unit tested, no DB).
- `run.py` — `--mode {retrieval,judge-slice,full,calibrate}`. With `--baseline`, a Recall@k drop beyond `--max-drop` exits non-zero, so the same script can back a blocking gate later.
  - `retrieval` — deterministic Recall@k / MRR (no LLM).
  - `judge-slice` — runs the agent end-to-end on a subset (default 12 cases) and posts `citation_match` + `answer_quality` to Langfuse by trace_id (needs an LLM).
  - `full` — same as `judge-slice` but over the **entire** gold set, using the same free judge model (no paid "frontier" judge — the project stays $0).
  - `calibrate` — measures each gold query's top rerank score + retrieval success and recommends the confidence-gate threshold (Youden's-J chooser); see `confidence_calibration.json`.

## CI fixture

`.github/workflows/eval.yml` is **manually triggered** (`workflow_dispatch`) and not a merge gate, but is architected to become one (uncomment the `pull_request` trigger — no script change).

CI runs against a small, deterministic, committed corpus instead of downloading/embedding NTSB data:

- `fixtures/seed.sql` — 12 gold reports + 60 distractors (with chunks + rounded embeddings), regenerable from a populated local DB via `python -m evals.fixtures.build_fixture`.
- `fixtures/ci_baseline.json` — the fixture's own Recall@5 / MRR baseline (distinct from the full-corpus `baseline.json`); the CI retrieval gate compares against it.
