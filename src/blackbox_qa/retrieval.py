"""Hybrid retrieval: dense (pgvector) + keyword (Postgres FTS) fused with RRF."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from blackbox_qa import db, embeddings

RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    ev_id: str
    content: str
    score: float  # fused RRF score (set by hybrid_search)


def dense_search(conn: psycopg.Connection, query: str, limit: int) -> list[Hit]:
    qv = embeddings.embed_query(query)
    rows = conn.execute(
        """
        SELECT id, ev_id, content, (embedding <=> %(q)s) AS distance
        FROM report_chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {"q": qv, "k": limit},
    ).fetchall()
    return [Hit(chunk_id=r[0], ev_id=r[1], content=r[2], score=1.0 - float(r[3])) for r in rows]


def keyword_search(conn: psycopg.Connection, query: str, limit: int) -> list[Hit]:
    rows = conn.execute(
        """
        SELECT id, ev_id, content, ts_rank(tsv, q) AS rank
        FROM report_chunks, websearch_to_tsquery('english', %(query)s) q
        WHERE tsv @@ q
        ORDER BY rank DESC
        LIMIT %(k)s
        """,
        {"query": query, "k": limit},
    ).fetchall()
    return [Hit(chunk_id=r[0], ev_id=r[1], content=r[2], score=float(r[3])) for r in rows]


def rrf_fuse(*ranked_lists: list, k: int = RRF_K) -> list[tuple[object, float]]:
    """Reciprocal Rank Fusion.

    Each input is a list of keys in rank order (best first). Returns
    (key, score) pairs sorted by fused score descending. Pure / testable.
    """
    scores: dict[object, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
    """Fuse dense + keyword chunk rankings with RRF; return top_k chunks."""
    with db.connect() as conn:
        dense = dense_search(conn, query, candidates)
        keyword = keyword_search(conn, query, candidates)

    by_id = {h.chunk_id: h for h in (*dense, *keyword)}
    fused = rrf_fuse([h.chunk_id for h in dense], [h.chunk_id for h in keyword])
    out: list[Hit] = []
    for chunk_id, score in fused[:top_k]:
        hit = by_id[chunk_id]
        out.append(Hit(chunk_id=hit.chunk_id, ev_id=hit.ev_id, content=hit.content, score=score))
    return out


def search_reports(query: str, top_k: int = 5, candidates: int = 50) -> list[str]:
    """Report-level results: distinct ev_ids in fused order (for Recall@k)."""
    hits = hybrid_search(query, top_k=candidates, candidates=candidates)
    seen: list[str] = []
    for h in hits:
        if h.ev_id not in seen:
            seen.append(h.ev_id)
        if len(seen) >= top_k:
            break
    return seen
