from blackbox_qa.retrieval import rrf_fuse


def test_single_list_preserves_order():
    fused = rrf_fuse(["a", "b", "c"])
    assert [k for k, _ in fused] == ["a", "b", "c"]


def test_items_in_both_lists_float_to_top():
    dense = ["a", "b", "c"]
    keyword = ["c", "d", "a"]
    fused = dict(rrf_fuse(dense, keyword))
    # 'a' (ranks 1 & 3) and 'c' (ranks 3 & 1) beat singletons 'b' and 'd'.
    assert fused["a"] > fused["b"]
    assert fused["c"] > fused["d"]


def test_score_formula():
    fused = dict(rrf_fuse(["x"], k=60))
    assert abs(fused["x"] - 1.0 / 61.0) < 1e-12


def test_agreement_beats_single_top_hit():
    # A doc both systems rank #2 should beat a doc only one system ranks #1.
    a = ["top1", "shared", "z"]
    b = ["other", "shared", "y"]
    fused = dict(rrf_fuse(a, b))
    assert fused["shared"] > fused["top1"]
