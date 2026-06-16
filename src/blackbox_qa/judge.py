"""LLM-as-judge: a temperature-0 chat completion that rates an answer.

A judge is "just another chat completion whose output you record" — here it
returns a 0..1 quality score that the eval runner posts to Langfuse via the
Scores API. Run at temp 0 with a tight JSON rubric for reproducibility; treat
the score as a noisy signal, not ground truth.
"""

from __future__ import annotations

import json
import re

from blackbox_qa import llm
from blackbox_qa.config import settings

JUDGE_SYSTEM = """You evaluate answers about NTSB aviation accident reports.
Given a QUESTION and an ANSWER, rate the ANSWER from 0.0 to 1.0 considering:
- groundedness: cites specific reports (ev_ids) / concrete facts, not vague claims;
- relevance: actually answers the question asked;
- non-evasiveness: does not punt ("I don't know") unless truly unanswerable.
Respond with ONLY compact JSON: {"score": <float 0..1>, "reason": "<one short sentence>"}."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(text: str) -> tuple[float, str]:
    m = _JSON_RE.search(text or "")
    if not m:
        return 0.0, "unparseable judge output"
    try:
        data = json.loads(m.group(0))
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        return score, str(data.get("reason", ""))[:300]
    except (ValueError, TypeError, json.JSONDecodeError):
        return 0.0, "unparseable judge output"


def judge_answer(question: str, answer: str) -> tuple[float, str]:
    """Return (score in 0..1, one-line reason). Uses JUDGE_MODEL if set, else LLM_MODEL."""
    resp = llm.chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nANSWER:\n{answer}"},
        ],
        model=settings.judge_model or None,
        temperature=0.0,
    )
    return _parse(resp.content or "")
