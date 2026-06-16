"""Raw tool-calling agent loop — no framework.

The loop is deliberately explicit so the engineered behaviour is visible:
bounded iterations, per-call argument validation with self-correction, a
`tool_choice="none"` final-turn floor that guarantees termination, and one
confidence-triggered query-rewrite retry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from blackbox_qa import llm, tools

MAX_TURNS = 8
MAX_RETRIES = 1

SYSTEM_PROMPT = """You are an aviation-safety analyst answering questions about NTSB accident reports.

Tools:
- hybrid_search: semantic + keyword search over accident narratives. Use for "what happened" / cause / how questions.
- sql_query: ONE read-only SELECT over the structured `reports` table. Use for counts, aggregations, and filtering (by year, aircraft make/model, state, injury severity, weather, etc.).
- fetch_full_report: pull one complete report by ev_id to confirm details after a search.

Rules:
- Ground every claim in tool results. Cite the ev_id(s) you relied on.
- Prefer sql_query for "how many / which / list" questions; hybrid_search for narrative questions.
- When you have enough evidence, stop calling tools and write the final answer.
- End the final answer with a line exactly "CONFIDENCE: high" or "CONFIDENCE: low" (low if the tools did not yield enough evidence)."""

_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(high|low)", re.IGNORECASE)
_EV_ID_RE = re.compile(r"\b\d{8}[A-Za-z]\d{5}\b")


@dataclass
class AgentResult:
    answer: str
    citations: list[str]
    turns: int
    confidence: str
    tool_log: list[dict[str, Any]] = field(default_factory=list)


def _assistant_msg(resp: llm.LLMResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": resp.content or "",
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in resp.tool_calls
        ],
    }


def _exec_tool(call: llm.ToolCall, tool_log: list[dict[str, Any]]) -> str:
    """Run one tool call; on any bad input return an error string for self-correction."""
    try:
        args = json.loads(call.arguments or "{}")
    except json.JSONDecodeError as exc:
        result = f"ERROR: arguments were not valid JSON ({exc}). Resend a valid JSON object."
        tool_log.append({"tool": call.name, "args_raw": call.arguments, "error": result})
        return result
    try:
        result = tools.dispatch(call.name, args)
        tool_log.append({"tool": call.name, "args": args, "ok": True})
        return result
    except tools.ToolError as exc:
        result = f"ERROR: {exc}"
        tool_log.append({"tool": call.name, "args": args, "error": str(exc)})
        return result


def _confidence(text: str) -> str:
    m = _CONFIDENCE_RE.search(text or "")
    return m.group(1).lower() if m else "high"


def _strip_confidence(text: str) -> str:
    return _CONFIDENCE_RE.sub("", text or "").strip()


def _citations(answer: str, tool_log: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for ev in _EV_ID_RE.findall(answer or ""):
        if ev not in seen:
            seen.append(ev)
    return seen


def run(question: str, max_turns: int = MAX_TURNS, max_retries: int = MAX_RETRIES) -> AgentResult:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_log: list[dict[str, Any]] = []
    retries_left = max_retries

    for turn in range(1, max_turns + 1):
        final_turn = turn == max_turns
        # Graceful floor: on the last turn forbid tools so the loop must answer.
        resp = llm.chat(
            messages,
            tools=tools.SCHEMAS,
            tool_choice="none" if final_turn else "auto",
        )

        if resp.wants_tool and not final_turn:
            messages.append(_assistant_msg(resp))
            for call in resp.tool_calls:
                output = _exec_tool(call, tool_log)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": output}
                )
            continue

        answer = resp.content or ""
        confidence = _confidence(answer)

        # One confidence-triggered retry: nudge a reformulation instead of giving up.
        if confidence == "low" and retries_left > 0 and not final_turn:
            retries_left -= 1
            messages.append({"role": "assistant", "content": answer})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That answer was low-confidence. Reformulate your search with "
                        "different keywords or filters and use the tools again before answering."
                    ),
                }
            )
            continue

        clean = _strip_confidence(answer)
        return AgentResult(
            answer=clean,
            citations=_citations(answer, tool_log),
            turns=turn,
            confidence=confidence,
            tool_log=tool_log,
        )

    # Unreachable: the final turn always returns. Defensive fallback.
    return AgentResult(
        answer="Unable to produce an answer within the turn budget.",
        citations=[],
        turns=max_turns,
        confidence="low",
        tool_log=tool_log,
    )
