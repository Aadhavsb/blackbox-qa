"""Parameterized eval runner.

Modes (matching the CI workflow's `mode` input):
  retrieval   - deterministic Recall@k / MRR over the gold set (no LLM). Default.
  judge-slice - small temp-0 judge-scored end-to-end slice (added in phase 5).
  full        - full frontier-judge run (added in phase 5).
  calibrate   - choose the confidence-gate threshold from the gold set (no LLM).

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


def run_retrieval(gold: list[dict], k: int, use_rerank: bool = False) -> dict:
    # Imported here so the heavy retrieval stack only loads for this mode.
    from blackbox_qa.retrieval import search_reports

    recalls: list[float] = []
    rrs: list[float] = []
    for row in gold:
        relevant = _relevant(row)
        retrieved = search_reports(row["query"], top_k=k, use_rerank=use_rerank)
        recalls.append(metrics.recall_at_k(retrieved, relevant, k))
        rrs.append(metrics.reciprocal_rank(retrieved, relevant))
    return {
        "n": len(gold),
        f"recall@{k}": round(metrics.mean(recalls), 4),
        "mrr": round(metrics.mean(rrs), 4),
    }


def run_judge_slice(gold: list[dict], limit: int | None = None) -> dict:
    """Run the agent end-to-end on a slice, score each answer, post scores by trace_id.

    Two scores per question: `citation_match` (deterministic — did the agent cite a
    gold ev_id?) and `answer_quality` (LLM-as-judge, 0..1). Both attach to the run's
    Langfuse trace out-of-band via the Scores API.
    """
    from blackbox_qa import agent, judge, observability as obs

    rows = gold[: limit] if limit else gold
    results: list[dict] = []
    for row in rows:
        res = agent.run(row["query"])
        gold_ids = _relevant(row)
        cited = set(res.citations)
        citation_match = 1 if (cited & gold_ids) else 0
        quality, reason = judge.judge_answer(row["query"], res.answer)
        if res.trace_id:
            obs.score(
                "citation_match",
                citation_match,
                trace_id=res.trace_id,
                data_type="BOOLEAN",
                comment=f"gold={sorted(gold_ids)} cited={sorted(cited)}",
            )
            obs.score(
                "answer_quality", quality, trace_id=res.trace_id, data_type="NUMERIC", comment=reason
            )
        results.append(
            {
                "query": row["query"],
                "citation_match": citation_match,
                "answer_quality": quality,
                "confidence": res.confidence,
                "trace_id": res.trace_id,
            }
        )
    obs.flush()
    return {
        "summary": {
            "n": len(results),
            "citation_match_rate": round(metrics.mean([r["citation_match"] for r in results]), 4),
            "answer_quality_mean": round(metrics.mean([r["answer_quality"] for r in results]), 4),
        },
        "results": results,
    }


def choose_threshold(scored: list[tuple[float, bool]]) -> dict:
    """Pick a confidence-gate threshold from (top_score, retrieval_succeeded) pairs.

    The gate fires (triggers a retry) when top_score < threshold, i.e. it is a
    detector for *failed* retrievals. We sweep candidate cut points and maximise
    Youden's J = TPR - FPR: catch as many failures as possible (TPR) while
    rarely firing on successes (FPR, i.e. wasted retries). Pure / testable.
    """
    failures = [s for s, ok in scored if not ok]
    successes = [s for s, ok in scored if ok]
    all_scores = [s for s, _ in scored]
    if not failures:
        thr = (min(all_scores) - 1.0) if all_scores else 0.0
        return {"threshold": round(thr, 4), "note": "no failed retrievals in gold; gate effectively off"}
    if not successes:
        return {"threshold": round(max(all_scores) + 1.0, 4), "note": "no successful retrievals in gold"}

    uniq = sorted(set(all_scores))
    candidates = [uniq[0] - 1.0] + [(a + b) / 2 for a, b in zip(uniq, uniq[1:])] + [uniq[-1] + 1.0]
    best: dict | None = None
    for thr in candidates:
        tpr = sum(1 for s in failures if s < thr) / len(failures)
        fpr = sum(1 for s in successes if s < thr) / len(successes)
        cand = {
            "threshold": round(thr, 4),
            "tpr": round(tpr, 4),  # fraction of failures the gate catches
            "fpr": round(fpr, 4),  # fraction of successes wastefully retried
            "youden_j": round(tpr - fpr, 4),
        }
        if best is None or cand["youden_j"] > best["youden_j"] or (
            cand["youden_j"] == best["youden_j"] and cand["fpr"] < best["fpr"]
        ):
            best = cand
    assert best is not None
    return best


def _score_stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "mean": round(metrics.mean(vals), 4),
    }


def run_calibrate(gold: list[dict], k: int, candidates: int = 50) -> dict:
    """Measure each gold query's top rerank score + whether retrieval succeeded,
    then recommend a confidence-gate threshold. No LLM calls."""
    from blackbox_qa.rerank import rerank_hits
    from blackbox_qa.retrieval import hybrid_search

    per: list[dict] = []
    for row in gold:
        relevant = _relevant(row)
        pool = hybrid_search(row["query"], top_k=candidates, candidates=candidates)
        ranked = rerank_hits(row["query"], pool)
        top_score = float(ranked[0].score) if ranked else None
        ev_order: list[str] = []
        for h in ranked:
            if h.ev_id not in ev_order:
                ev_order.append(h.ev_id)
        success = bool(relevant & set(ev_order[:k]))
        per.append(
            {
                "query": row["query"],
                "top_score": round(top_score, 4) if top_score is not None else None,
                "success": success,
            }
        )
    scored = [(p["top_score"], p["success"]) for p in per if p["top_score"] is not None]
    return {
        "n": len(per),
        "k": k,
        "recommended_threshold": choose_threshold(scored),
        "success_scores": _score_stats([s for s, ok in scored if ok]),
        "failure_scores": _score_stats([s for s, ok in scored if not ok]),
        "per_query": per,
    }


def run_ablation(gold: list[dict], k: int) -> dict:
    """Hybrid vs. hybrid+rerank on the same gold set."""
    base = run_retrieval(gold, k, use_rerank=False)
    rerank = run_retrieval(gold, k, use_rerank=True)
    rk = f"recall@{k}"
    return {
        "n": len(gold),
        "hybrid": {rk: base[rk], "mrr": base["mrr"]},
        "hybrid+rerank": {rk: rerank[rk], "mrr": rerank["mrr"]},
        "delta": {
            rk: round(rerank[rk] - base[rk], 4),
            "mrr": round(rerank["mrr"] - base["mrr"], 4),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="blackbox-qa eval runner")
    parser.add_argument(
        "--mode",
        choices=("retrieval", "judge-slice", "full", "calibrate"),
        default="retrieval",
    )
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="add cross-encoder rerank stage")
    parser.add_argument("--ablation", action="store_true", help="compare hybrid vs hybrid+rerank")
    parser.add_argument("--baseline", type=Path, default=None, help="baseline results JSON")
    parser.add_argument("--max-drop", type=float, default=0.01, help="allowed Recall@k drop")
    parser.add_argument("--limit", type=int, default=None, help="cap rows (judge-slice)")
    parser.add_argument("--out", type=Path, default=None, help="write result JSON here (calibrate)")
    args = parser.parse_args()

    if args.mode == "full":
        print("mode 'full' not implemented yet (phase 5).")
        return 0
    if not args.gold.exists():
        raise SystemExit(
            f"gold set not found: {args.gold}\n"
            f"Populate it from ingested data (see evals/gold/retrieval_gold.example.jsonl)."
        )

    gold = load_gold(args.gold)
    if args.mode == "calibrate":
        result = run_calibrate(gold, args.k)
        print(json.dumps(result, indent=2))
        if args.out:
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(f"wrote {args.out}")
        return 0
    if args.mode == "judge-slice":
        print(json.dumps(run_judge_slice(gold, limit=args.limit), indent=2))
        return 0
    if args.ablation:
        print(json.dumps(run_ablation(gold, args.k), indent=2))
        return 0

    result = run_retrieval(gold, args.k, use_rerank=args.rerank)
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
