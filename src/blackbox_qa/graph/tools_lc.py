"""LangChain-tool wrappers around the existing `tools.py` functions.

Only the dispatch/schema plumbing is re-expressed here; the safety logic
(`validate_select`, the forbidden-keyword/function regexes, the read-only
transaction) still lives in `tools.py` and runs inside these wrappers.

`sql_query` is deliberately NOT exposed as an auto-runnable tool: the graph's
`sql_agent` proposes SQL, a human approves it via `interrupt()`, and only then
does it call `tools.sql_query` directly (which still runs `validate_select`).
"""

from __future__ import annotations

from langchain_core.tools import tool

from blackbox_qa import tools as bbq


@tool
def hybrid_search(query: str, k: int = 5) -> str:
    """Semantic + keyword search over NTSB accident narratives.

    Use for "what happened" / cause / how questions. Returns ranked chunks,
    each tagged with its ev_id and a rerank score.
    """
    return bbq.hybrid_search(query, k)


@tool
def fetch_full_report(ev_id: str) -> str:
    """Fetch one complete NTSB report (structured fields + full narrative) by ev_id."""
    return bbq.fetch_full_report(ev_id)
