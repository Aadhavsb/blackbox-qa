import pytest

from evals.metrics import (
    first_relevant_rank,
    mean,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_hit_and_miss():
    assert recall_at_k(["a", "b", "c"], {"b"}, k=5) == 1.0
    assert recall_at_k(["a", "b", "c"], {"z"}, k=5) == 0.0


def test_recall_respects_k():
    assert recall_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert recall_at_k(["a", "b", "c"], {"c"}, k=3) == 1.0


def test_recall_fraction_multi_relevant():
    assert recall_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4) == 1.0
    assert recall_at_k(["a", "x", "y", "z"], {"a", "c"}, k=4) == 0.5


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], {"b"}) == 2
    assert first_relevant_rank(["a", "b"], {"z"}) is None


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["a"], {"z"}) == 0.0


def test_mean():
    assert mean([1.0, 0.0]) == 0.5
    assert mean([]) == 0.0


def test_empty_relevant_raises():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), k=5)
