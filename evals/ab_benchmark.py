"""A/B benchmark: framework-free baseline (`agent.run`) vs the LangGraph
multi-agent graph, scored on the SAME gold set with the SAME two metrics as
`evals/run.py` — deterministic `citation_match` + LLM-as-judge `answer_quality`.

    poetry run python -m evals.ab_benchmark --limit 5
    poetry run python -m evals.ab_benchmark --limit 0 --out evals/ab.json   # full gold set

SQL approvals are auto-approved here (batch mode); the interactive approval
lives in `blackbox_qa.graph.run_hitl`. This is the number that decides whether
multi-agent actually beat the single-agent baseline — and by how much.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langgraph.types import Command

from blackbox_qa import agent, judge
from blackbox_qa.agent import _EV_ID_RE
from evals import metrics
from evals.run import DEFAULT_GOLD, _relevant, load_gold


def _run_baseline(question: str) -> tuple[str, set[str]]:
    res = agent.run(question)
    return res.answer, set(res.citations)


def _run_graph(graph, question: str, thread: str) -> tuple[str, set[str]]:
    cfg = {"configurable": {"thread_id": thread}}
    state = graph.invoke({"question": question}, cfg)
    guard = 0
    while "__interrupt__" in state and guard < 5:  # auto-approve every HITL pause
        state = graph.invoke(Command(resume="approve"), cfg)
        guard += 1
    answer = state["messages"][-1].content or ""
    return answer, set(_EV_ID_RE.findall(answer))


def _score(question: str, answer: str, citations: set[str], gold_ids: set[str]) -> dict:
    quality, reason = judge.judge_answer(question, answer)
    return {
        "citation_match": 1 if (citations & gold_ids) else 0,
        "answer_quality": quality,
        "reason": reason,
    }


def run(gold: list[dict], limit: int | None = None) -> dict:
    from blackbox_qa.graph.build import build

    graph = build()
    rows = gold[:limit] if limit else gold
    per: list[dict] = []

    for i, row in enumerate(rows):
        q = row["query"]
        gold_ids = _relevant(row)
        rec: dict = {"query": q}
        systems = (
            ("baseline", lambda: _run_baseline(q)),
            ("graph", lambda: _run_graph(graph, q, f"ab-{i}")),
        )
        for name, fn in systems:
            try:
                ans, cits = fn()
                rec[name] = _score(q, ans, cits, gold_ids)
            except Exception as exc:  # noqa: BLE001 - record and continue
                rec[name] = {"citation_match": 0, "answer_quality": 0.0, "error": str(exc)}
            print(
                f"[{i + 1}/{len(rows)}] {name}: q={rec[name].get('answer_quality')} "
                f"cite={rec[name].get('citation_match')}  {q[:55]}",
                flush=True,
            )
        per.append(rec)

    def summarize(key: str) -> dict:
        vals = [r[key] for r in per if key in r]
        return {
            "citation_match_rate": round(metrics.mean([v["citation_match"] for v in vals]), 4),
            "answer_quality_mean": round(metrics.mean([v["answer_quality"] for v in vals]), 4),
        }

    base_s, graph_s = summarize("baseline"), summarize("graph")
    return {
        "n": len(per),
        "baseline": base_s,
        "graph": graph_s,
        "delta": {
            "citation_match_rate": round(
                graph_s["citation_match_rate"] - base_s["citation_match_rate"], 4
            ),
            "answer_quality_mean": round(
                graph_s["answer_quality_mean"] - base_s["answer_quality_mean"], 4
            ),
        },
        "per_query": per,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="A/B: framework-free baseline vs LangGraph multi-agent")
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--limit", type=int, default=5, help="cap queries (0 = full gold set)")
    p.add_argument("--out", type=Path, default=None, help="write full result JSON here")
    args = p.parse_args()

    if not args.gold.exists():
        raise SystemExit(f"gold set not found: {args.gold}")

    limit = args.limit if args.limit and args.limit > 0 else None
    result = run(load_gold(args.gold), limit=limit)

    summary = {k: v for k, v in result.items() if k != "per_query"}
    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
