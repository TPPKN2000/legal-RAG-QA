"""
Embedding layer. Wraps `sentence-transformers` so the rest of the codebase
never touches the model directly (keeps model.py as the single place that
knows about device placement / batching / model names).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from backend import config


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    device = config.DEVICE
    try:
        import torch
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    except ImportError:
        device = "cpu"

    return SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)


def embed_texts(texts: Sequence[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
    """Embed a batch of texts, returning an (N, dim) float32 array.

    Embeddings are L2-normalized by default so that cosine similarity search
    (Pinecone metric="cosine") and dot-product scoring are equivalent.
    """
    if not texts:
        return np.zeros((0, config.EMBEDDING_DIM), dtype="float32")
    model = _get_embedder()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=normalize,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def embed_query(text: str) -> list[float]:
    """Convenience wrapper for a single query string -> plain python list,
    which is what Pinecone's query API expects."""
    return embed_texts([text])[0].tolist()
