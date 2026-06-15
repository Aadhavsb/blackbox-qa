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
```

- `metrics.py` — pure Recall@k / MRR (unit tested, no DB).
- `run.py` — `--mode {retrieval,judge-slice,full}`. `retrieval` is deterministic and implemented; the judge modes arrive in phase 5. With `--baseline`, a Recall@k drop beyond `--max-drop` exits non-zero, so the same script can back a blocking gate later.

The CI pipeline (`.github/workflows/eval.yml`) is **manually triggered** (`workflow_dispatch`) and not a merge gate, but is architected to become one (add a `pull_request` trigger).
