"""Retrieval metrics. Pure functions, no DB — unit testable."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items found in the top-k.

    With a single gold doc per query this is 1.0 if the gold doc is in the
    top-k, else 0.0 — the standard retrieval-gate signal.
    """
    if not relevant:
        raise ValueError("relevant set must be non-empty")
    found = sum(1 for item in retrieved[:k] if item in relevant)
    return found / len(relevant)


def first_relevant_rank(retrieved: Sequence[str], relevant: set[str]) -> int | None:
    """1-based rank of the first relevant item, or None if absent."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return rank
    return None


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    rank = first_relevant_rank(retrieved, relevant)
    return 0.0 if rank is None else 1.0 / rank


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
