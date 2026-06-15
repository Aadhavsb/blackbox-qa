# Evaluation

Frozen gold set + a parameterized runner.

- `gold/` — committed gold set (LLM-drafted, hand-verified): retrieval gold (query → report ID) and end-to-end gold (query → reference answer).
- `run.py` — `--mode {retrieval,judge-slice,full}` (added in phase 5).

The CI pipeline (`.github/workflows/eval.yml`) is **manually triggered** and not a merge gate, but is architected to become one.
