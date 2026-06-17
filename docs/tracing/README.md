# Example Langfuse traces

Screenshots from the self-hosted Langfuse stack (`docker compose --profile observability up`),
showing real `agent.run` traces with LLM-as-judge scores attached.

| File | What it shows |
|---|---|
| [`trace-high-confidence.png`](trace-high-confidence.png) | A high-quality answer: the observation tree (one generation per turn + a span per `hybrid_search` call), token usage (8,084 → 323), and both judge scores on the **same** trace — `answer_quality 0.95`, `citation_match True`. |
| [`trace-low-confidence.png`](trace-low-confidence.png) | The confidence gate on weak retrieval (the *"747 lost all four generator control units"* query): low judge scores — `answer_quality 0.10`, `citation_match False`. |
| [`scores-tab.png`](scores-tab.png) | The Scores tab for the high-confidence trace: `answer_quality` (NUMERIC) and `citation_match` (BOOLEAN), posted out-of-band via the Scores API. |

Reproduce locally (with `LANGFUSE_ENABLED=true` and a project key pair in `.env`):

```bash
poetry run python -m evals.run --mode judge-slice --limit 3
```
