"""Threshold selection for the confidence gate. Pure function, no DB/LLM."""

from evals.run import choose_threshold


def test_perfectly_separable_threshold_sits_between_classes():
    # Failures score low (0, 1), successes high (5, 6): cut belongs in (1, 5).
    scored = [(0.0, False), (1.0, False), (5.0, True), (6.0, True)]
    rec = choose_threshold(scored)
    assert 1.0 < rec["threshold"] <= 5.0
    assert rec["tpr"] == 1.0  # catches both failures
    assert rec["fpr"] == 0.0  # never fires on a success


def test_no_failures_gate_effectively_off():
    scored = [(2.0, True), (3.0, True)]
    rec = choose_threshold(scored)
    # Threshold below the minimum score => score < threshold is never true.
    assert rec["threshold"] < 2.0
    assert "note" in rec


def test_no_successes_gate_fires_for_all():
    scored = [(2.0, False), (3.0, False)]
    rec = choose_threshold(scored)
    assert rec["threshold"] > 3.0
    assert "note" in rec


def test_overlap_picks_best_tradeoff():
    # One failure sits above a success (overlap); J-max tolerates one miss/FP.
    scored = [(0.0, False), (4.0, False), (3.0, True), (6.0, True)]
    rec = choose_threshold(scored)
    assert -1.0 <= rec["threshold"] <= 7.0
    assert 0.0 <= rec["youden_j"] <= 1.0
