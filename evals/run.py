"""Parameterized eval runner.

Modes (matching the CI workflow's `mode` input):
  retrieval   - deterministic Recall@k / MRR over the gold set (no LLM). Default.
  judge-slice - small temp-0 judge-scored end-to-end slice (added in phase 5).
  full        - full frontier-judge run (added in phase 5).

Exit code is non-zero when a baseline is supplied and Recall@k regresses beyond
the tolerance, so this script can later back a blocking PR gate unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals import metrics

DEFAULT_GOLD = Path(__file__).resolve().parent / "gold" / "retrieval_gold.jsonl"


def load_gold(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def _relevant(row: dict) -> set[str]:
    if "ev_ids" in row:
        return set(row["ev_ids"])
    return {row["ev_id"]}


def run_retrieval(gold: list[dict], k: int) -> dict:
    # Imported here so the heavy retrieval stack only loads for this mode.
    from blackbox_qa.retrieval import search_reports

    recalls: list[float] = []
    rrs: list[float] = []
    for row in gold:
        relevant = _relevant(row)
        retrieved = search_reports(row["query"], top_k=k)
        recalls.append(metrics.recall_at_k(retrieved, relevant, k))
        rrs.append(metrics.reciprocal_rank(retrieved, relevant))
    return {
        "n": len(gold),
        f"recall@{k}": round(metrics.mean(recalls), 4),
        "mrr": round(metrics.mean(rrs), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="blackbox-qa eval runner")
    parser.add_argument("--mode", choices=("retrieval", "judge-slice", "full"), default="retrieval")
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--baseline", type=Path, default=None, help="baseline results JSON")
    parser.add_argument("--max-drop", type=float, default=0.01, help="allowed Recall@k drop")
    args = parser.parse_args()

    if args.mode != "retrieval":
        print(f"mode {args.mode!r} not implemented yet (phase 5).")
        return 0
    if not args.gold.exists():
        raise SystemExit(
            f"gold set not found: {args.gold}\n"
            f"Populate it from ingested data (see evals/gold/retrieval_gold.example.jsonl)."
        )

    result = run_retrieval(load_gold(args.gold), args.k)
    print(json.dumps(result, indent=2))

    if args.baseline and args.baseline.exists():
        base = json.loads(args.baseline.read_text())
        key = f"recall@{args.k}"
        drop = base.get(key, 0.0) - result[key]
        if drop > args.max_drop:
            print(f"REGRESSION: {key} dropped {drop:.4f} > tolerance {args.max_drop}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
