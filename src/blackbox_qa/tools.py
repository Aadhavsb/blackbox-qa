"""The three tools the agent can call, their JSON schemas, and a read-only SQL guard.

Tool functions return a plain string (the tool result fed back to the model).
Bad arguments raise ToolError, which the agent catches and feeds back so the
model can self-correct instead of crashing the loop.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from blackbox_qa import db, retrieval

MAX_SNIPPET_CHARS = 600
MAX_SQL_ROWS = 50
SQL_TIMEOUT_MS = 5000


class ToolError(Exception):
    """Raised for invalid tool arguments or disallowed operations."""


# --- read-only SQL guard --------------------------------------------------

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|"
    r"merge|vacuum|analyze|reindex|comment|call|do|set|begin|commit|rollback)\b",
    re.IGNORECASE,
)


def validate_select(sql: str) -> str:
    """Return a cleaned single read-only SELECT, or raise ToolError.

    Pure + deterministic so it can be unit-tested without a database. This is
    the demonstrable safety boundary: defense-in-depth over the DB-level
    read-only transaction in _run_select.
    """
    if not isinstance(sql, str):
        raise ToolError("sql must be a string")
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ToolError("empty SQL query")
    if ";" in cleaned:
        raise ToolError("multiple statements are not allowed; submit a single SELECT")
    low = cleaned.lower()
    if not (low.startswith("select") or low.startswith("with")):
        raise ToolError("only read-only SELECT (or WITH ... SELECT) queries are allowed")
    if _FORBIDDEN.search(cleaned):
        raise ToolError("query contains a forbidden keyword; only read-only SELECT is allowed")
    return cleaned


def _run_select(sql: str) -> str:
    with db.connect() as conn:
        # DB-level belt-and-suspenders on top of validate_select.
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute(f"SET LOCAL statement_timeout = {SQL_TIMEOUT_MS}")
        cur = conn.execute(sql)
        cols = [d.name for d in cur.description] if cur.description else []
        rows = cur.fetchmany(MAX_SQL_ROWS)
    if not rows:
        return "0 rows."
    header = " | ".join(cols)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in row) for row in rows)
    note = f"\n... (truncated at {MAX_SQL_ROWS} rows)" if len(rows) == MAX_SQL_ROWS else ""
    return f"{header}\n{body}{note}"


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_SNIPPET_CHARS else text[:MAX_SNIPPET_CHARS] + "..."


# --- tools ----------------------------------------------------------------


def hybrid_search(query: str, k: int = 5) -> str:
    """Semantic + keyword search over narrative chunks, reranked for precision."""
    if not isinstance(query, str) or not query.strip():
        raise ToolError("query must be a non-empty string")
    k = max(1, min(int(k), 10))
    pool = retrieval.hybrid_search(query, top_k=50, candidates=50)
    from blackbox_qa.rerank import rerank_hits

    hits = rerank_hits(query, pool, top_k=k)
    if not hits:
        return "No matching narrative chunks."
    return "\n\n".join(
        f"[{i}] ev_id={h.ev_id} (score={h.score:.3f})\n{_snippet(h.content)}"
        for i, h in enumerate(hits, start=1)
    )


def sql_query(sql: str) -> str:
    """Run one read-only SELECT against the structured `reports` table."""
    return _run_select(validate_select(sql))


def fetch_full_report(ev_id: str) -> str:
    """Fetch one full report (structured fields + complete narrative) by ev_id."""
    if not isinstance(ev_id, str) or not ev_id.strip():
        raise ToolError("ev_id must be a non-empty string")
    ev_id = ev_id.strip()
    with db.connect() as conn:
        cur = conn.execute("SELECT * FROM reports WHERE ev_id = %s", (ev_id,))
        cols = [d.name for d in cur.description]
        rep = cur.fetchone()
        if rep is None:
            raise ToolError(f"no report with ev_id={ev_id!r}")
        chunks = conn.execute(
            "SELECT source, content FROM report_chunks "
            "WHERE ev_id = %s ORDER BY source, chunk_index",
            (ev_id,),
        ).fetchall()
    fields = "\n".join(f"{c}: {v}" for c, v in zip(cols, rep, strict=True) if v is not None)
    narrative = "\n".join(f"[{src}] {content}" for src, content in chunks) or "(no narrative)"
    return f"== structured fields ==\n{fields}\n\n== narrative ==\n{narrative}"


# --- registry + schemas ---------------------------------------------------

REGISTRY: dict[str, Callable[..., str]] = {
    "hybrid_search": hybrid_search,
    "sql_query": sql_query,
    "fetch_full_report": fetch_full_report,
}

# Columns exposed to the model for sql_query (mirrors sql/schema.sql `reports`).
REPORTS_COLUMNS = (
    "ev_id, ntsb_no, ev_date, ev_year, ev_month, ev_type, ev_city, ev_state, "
    "ev_country, latitude, longitude, ev_highest_injury, inj_tot_fatal, "
    "inj_tot_serious, inj_tot_minor, inj_tot_none, inj_tot_total, wx_cond_basic, "
    "light_cond, acft_make, acft_model, acft_series, acft_category, far_part, damage"
)

SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "hybrid_search",
            "description": (
                "Semantic + keyword search over accident narrative text. Use for "
                "'what happened' / cause questions. Returns ranked chunks with ev_id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "natural-language search query"},
                    "k": {"type": "integer", "description": "results to return (1-10)", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": (
                "Run ONE read-only SELECT against the `reports` table for counts / "
                f"aggregations / filtering. Columns: {REPORTS_COLUMNS}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "a single SELECT statement"},
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_full_report",
            "description": (
                "Fetch one complete report (structured fields + full narrative) by "
                "its ev_id, e.g. after hybrid_search surfaces a candidate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ev_id": {"type": "string", "description": "the report's ev_id"},
                },
                "required": ["ev_id"],
            },
        },
    },
]


def dispatch(name: str, arguments: dict[str, Any]) -> str:
    """Invoke a tool by name with parsed JSON arguments. Raises ToolError on bad input."""
    fn = REGISTRY.get(name)
    if fn is None:
        raise ToolError(f"unknown tool {name!r}; available: {', '.join(REGISTRY)}")
    if not isinstance(arguments, dict):
        raise ToolError("tool arguments must be a JSON object")
    try:
        return fn(**arguments)
    except ToolError:
        raise
    except TypeError as exc:
        raise ToolError(f"invalid arguments for {name}: {exc}") from exc
