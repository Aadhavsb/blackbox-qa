"""Cross-encoder reranking (local, CPU, free).

A cross-encoder scores (query, passage) pairs jointly — far more accurate than
the bi-encoder used for first-stage retrieval, but too slow for the whole
corpus, so it only ever sees the hybrid stage's shortlist.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

from blackbox_qa.config import settings
from blackbox_qa.retrieval import Hit


@cache
def _model():
    # Lazy import so torch only loads when reranking is actually used.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


def rerank_hits(query: str, hits: list[Hit], top_k: int | None = None) -> list[Hit]:
    """Rescore + reorder candidate hits by cross-encoder relevance.

    Returns hits sorted by rerank score (descending), with `score` replaced by
    the cross-encoder score. `top_k=None` keeps the whole reordered pool.
    """
    if not hits:
        return []
    scores = _model().predict([(query, h.content) for h in hits])
    ranked = sorted(zip(hits, scores, strict=True), key=lambda x: float(x[1]), reverse=True)
    out = [replace(hit, score=float(s)) for hit, s in ranked]
    return out if top_k is None else out[:top_k]
