"""Local sentence-transformers embedding wrapper (BAAI/bge-small-en-v1.5).

bge-small is an asymmetric retrieval model: passages are embedded as-is, but
queries should carry a short instruction prefix. Embeddings are L2-normalized
so cosine similarity equals the dot product.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from blackbox_qa.config import settings

# Recommended query-side instruction for bge-*-en-v1.5 (passages get none).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 384


@cache
def _model():
    # Imported lazily so the rest of the package (and its tests) don't require torch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    return _model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    return _model().encode(
        QUERY_INSTRUCTION + text, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
