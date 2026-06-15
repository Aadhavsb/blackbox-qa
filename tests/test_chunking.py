from blackbox_qa.chunking import chunk_text, normalize


def test_normalize_collapses_whitespace():
    assert normalize("  a\n\t b   c ") == "a b c"


def test_empty_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_single_chunk():
    text = "engine failed on takeoff"
    assert chunk_text(text, target_words=200, overlap_words=40) == [text]


def test_windows_overlap_and_cover_all_words():
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = chunk_text(text, target_words=30, overlap_words=10)

    # Every original word appears in at least one chunk.
    seen = set()
    for c in chunks:
        seen.update(c.split(" "))
    assert seen == set(words)

    # Consecutive chunks share exactly `overlap_words` words at the seam.
    first = chunks[0].split(" ")
    second = chunks[1].split(" ")
    assert first[-10:] == second[:10]


def test_no_duplicate_final_window():
    text = " ".join(f"w{i}" for i in range(60))
    chunks = chunk_text(text, target_words=30, overlap_words=10)
    # step = 20 -> windows at 0,20,40; 40..70 clipped to 40..60. No empty tail.
    assert chunks[-1].split(" ")[-1] == "w59"
    assert all(c.strip() for c in chunks)


def test_invalid_params():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("x", target_words=0)
    with pytest.raises(ValueError):
        chunk_text("x", target_words=10, overlap_words=10)
