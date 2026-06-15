"""Deterministic narrative chunking.

Word-window chunker with overlap. Kept dependency-free and pure so it is unit
testable without a model or database. Targets fit comfortably under the
embedding model's 512-token limit (~200 words is well within it).
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


def chunk_text(text: str, target_words: int = 200, overlap_words: int = 40) -> list[str]:
    """Split text into overlapping word windows.

    Returns [] for empty input. The final window is kept even if short.
    """
    if target_words <= 0:
        raise ValueError("target_words must be positive")
    if not 0 <= overlap_words < target_words:
        raise ValueError("overlap_words must be in [0, target_words)")

    words = normalize(text).split(" ")
    if words == [""]:
        return []

    step = target_words - overlap_words
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + target_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + target_words >= len(words):
            break
    return chunks
