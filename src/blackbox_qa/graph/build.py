"""The multi-agent LangGraph: planner-free spine + a human-approved SQL step.

    START -> retriever_node -> sql_agent -> critic -> answer -> END
                   ^                          |
                   +----- (verdict == low) ---+   (Command handoff / cycle)

Design notes mapped to the framework-free baseline (`agent.py`):
- `State` replaces the pile of `run()` locals (messages/docs/verdict/retries).
- `retriever_node` wraps a prebuilt ReAct agent over the two search tools;
  it is the graph-native form of the baseline's tool loop.
- `sql_agent` proposes ONE read-only SELECT, then `interrupt()`s for human
  approval before executing it — a capability the `for`-loop agent could not
  have (needs the checkpointer to persist the paused run).
- `critic` returns a `Command`: on weak evidence it hands control back to the
  retriever (the baseline's confidence-gated single retry), else it finalizes.

Models come from the same free OpenAI-compatible endpoint the baseline uses
(Groq / Gemini / Ollama via .env) — nothing new to pay for.
"""

from __future__ import annotations

import operator
from functools import cache
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command, interrupt

from blackbox_qa import tools as bbq
from blackbox_qa.config import settings
from blackbox_qa.graph.tools_lc import fetch_full_report, hybrid_search

MAX_RETRIES = 1

RETRIEVER_PROMPT = (
    "You are an aviation-safety analyst. Search the NTSB narrative corpus and "
    "report grounded findings, citing the ev_id(s) you relied on inline."
)
SQL_PROMPT = (
    "Write ONE read-only SELECT over the `reports` table that answers the "
    "question (counts / aggregations / filtering). Output only the SQL, no prose."
)
ANSWER_PROMPT = (
    "You are an aviation-safety analyst. Using ONLY the evidence below (narrative "
    "search results and any SQL results), write a final answer to the question. "
    "Ground every claim and cite the ev_id(s) you relied on inline, e.g. "
    "(ev_id 20080107X00027). If the evidence is insufficient, say so plainly."
)


@cache
def _model() -> ChatOpenAI:
    """Same free endpoint the baseline agent uses; cached so import is cheap."""
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=0,
    )


@cache
def _retriever_agent():
    # create_agent (langchain.agents) is the current, non-deprecated ReAct
    # factory; the returned object is itself a compiled ReAct subgraph.
    from langchain.agents import create_agent

    return create_agent(
        model=_model(),
        tools=[hybrid_search, fetch_full_report],
        system_prompt=RETRIEVER_PROMPT,
    )


class State(TypedDict):
    question: str
    docs: Annotated[list[str], operator.add]  # accumulate evidence across retries
    draft: str
    verdict: str
    retries: int
    messages: Annotated[list[AnyMessage], add_messages]


def retriever_node(state: State) -> dict[str, Any]:
    out = _retriever_agent().invoke(
        {"messages": [{"role": "user", "content": state["question"]}]}
    )
    final = out["messages"][-1].content or ""
    return {"docs": [final], "draft": final, "messages": out["messages"]}


def sql_agent(state: State) -> dict[str, Any]:
    proposed = (
        _model()
        .invoke(
            [
                {"role": "system", "content": SQL_PROMPT},
                {"role": "user", "content": state["question"]},
            ]
        )
        .content
        or ""
    ).strip()

    # --- human-in-the-loop: pause until a reviewer approves the SQL ---
    decision = interrupt({"action": "approve_sql", "sql": proposed})
    if str(decision).strip().lower() != "approve":
        return {
            "messages": [
                {"role": "assistant", "content": "SQL rejected by reviewer; using narrative evidence only."}
            ]
        }

    try:
        result = bbq.sql_query(proposed)  # validate_select + read-only txn still run
    except bbq.ToolError as exc:
        result = f"ERROR: {exc}"
    return {"messages": [{"role": "tool", "content": result}], "draft": result}


def critic(state: State) -> Command[Literal["retriever_node", "answer"]]:
    # Gate on RETRIEVAL quality (the docs channel), the graph-native analog of
    # the baseline's confidence-gated retry — not on the SQL draft.
    joined = " ".join(state.get("docs") or [])
    weak = (not joined.strip()) or ("No matching" in joined)
    if weak and state.get("retries", 0) < MAX_RETRIES:
        return Command(
            goto="retriever_node",
            update={
                "verdict": "low",
                "retries": state.get("retries", 0) + 1,
                "messages": [
                    {"role": "user", "content": "Low-confidence: reformulate the search with different keywords."}
                ],
            },
        )
    return Command(goto="answer", update={"verdict": "high"})


def answer(state: State) -> dict[str, Any]:
    # Synthesize a grounded final answer from all accumulated evidence
    # (narrative docs + any approved SQL result), so the output is comparable
    # to the baseline agent's final answer.
    evidence = "\n\n".join(state.get("docs") or [])
    if state.get("draft"):
        evidence += "\n\n[structured/SQL result]\n" + state["draft"]
    final = (
        _model()
        .invoke(
            [
                {"role": "system", "content": ANSWER_PROMPT},
                {"role": "user", "content": f"QUESTION:\n{state['question']}\n\nEVIDENCE:\n{evidence}"},
            ]
        )
        .content
        or (state.get("draft") or "No answer produced.")
    )
    return {"draft": final, "messages": [{"role": "assistant", "content": final}]}


def build(checkpointer: Any | None = None):
    """Compile the graph. A checkpointer is required for the HITL interrupt;
    defaults to in-memory (dev). Use a Postgres checkpointer in production."""
    b = StateGraph(State)
    b.add_node("retriever_node", retriever_node)
    b.add_node("sql_agent", sql_agent)
    b.add_node("critic", critic)
    b.add_node("answer", answer)
    b.add_edge(START, "retriever_node")
    b.add_edge("retriever_node", "sql_agent")
    b.add_edge("sql_agent", "critic")
    b.add_edge("answer", END)
    return b.compile(checkpointer=checkpointer or InMemorySaver())
