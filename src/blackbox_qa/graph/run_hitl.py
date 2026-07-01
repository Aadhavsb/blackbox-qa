"""Driver for the multi-agent graph with the human-in-the-loop SQL approval.

    poetry run python -m blackbox_qa.graph.run_hitl "How many icing accidents in 2008?"

The run pauses at the SQL step and prints the proposed SELECT; type `approve`
or `reject`. Pass --yes to auto-approve (useful for the A/B benchmark).
"""

from __future__ import annotations

import sys

from langgraph.types import Command

from blackbox_qa.graph.build import build


def ask(question: str, thread: str = "t1", auto_approve: bool = False) -> dict:
    graph = build()
    cfg = {"configurable": {"thread_id": thread}}
    state = graph.invoke({"question": question}, cfg)  # runs until the first interrupt

    while "__interrupt__" in state:
        payload = state["__interrupt__"][0].value
        print("\n--- HUMAN APPROVAL REQUIRED ---")
        print("Proposed SQL:\n  " + str(payload.get("sql")))
        decision = "approve" if auto_approve else input("approve / reject > ").strip()
        state = graph.invoke(Command(resume=decision), cfg)  # re-enters sql_agent

    print("\n=== ANSWER ===")
    print(state["messages"][-1].content)
    return state


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--yes"]
    auto = "--yes" in sys.argv[1:]
    question = " ".join(args) or "How many icing-related accidents were there in 2008?"
    ask(question, auto_approve=auto)


if __name__ == "__main__":
    main()
