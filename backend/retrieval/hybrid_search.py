"""
Hybrid search (design doc §3.1): fuse BM25 and Pinecone vector results with
Reciprocal Rank Fusion (RRF) rather than a weighted score sum, because BM25
scores and cosine similarities live on incomparable scales — RRF sidesteps
that by fusing on *rank* instead of raw score.

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

`k` (config.RRF_K) is the standard damping constant (60 is the commonly used
default from the original RRF paper) that keeps a single retriever's #1 hit
from completely dominating the fused ranking.
"""
from __future__ import annotations

from typing import Optional

from backend import config
from backend.indexing import bm25_index, vector_store
from backend.models import RetrievedChunk
from backend.retrieval.querry_transform import generate_hyde, rewrite_query


def _rrf_fuse(result_lists: list[list[RetrievedChunk]], k: int = config.RRF_K) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            # Keep the richest copy of the chunk (text/law_id/aid identical
            # across retrievers, but we still need one canonical object).
            best_chunk.setdefault(chunk.chunk_id, chunk)

    fused = [
        RetrievedChunk(
            chunk_id=cid,
            law_id=best_chunk[cid].law_id,
            aid=best_chunk[cid].aid,
            text=best_chunk[cid].text,
            score=score,
            source="fused",
        )
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


def hybrid_search(
    query: str,
    law_id: Optional[str] = None,
    require_active: bool = True,
    use_query_rewriting: bool = True,
    use_hyde: bool = True,
    top_k: int = 30,
) -> list[RetrievedChunk]:
    """Run BM25 + vector search (optionally over multiple query rewrites and
    a HyDE passage), then fuse everything with RRF.

    Metadata filtering (law_id / active-only) is pushed down into both the
    BM25 index and the Pinecone query themselves (design doc §3.2), not
    applied after fusion, so it never silently reduces an already-truncated
    top_k.
    """
    bm25 = bm25_index.get_bm25_index()

    queries = rewrite_query(query) if use_query_rewriting else [query]

    result_lists: list[list[RetrievedChunk]] = []
    for q in queries:
        result_lists.append(
            bm25.query(q, top_k=config.BM25_TOP_K, law_id=law_id, require_active=require_active)
        )
        result_lists.append(
            vector_store.query(q, top_k=config.VECTOR_TOP_K, law_id=law_id, require_active=require_active)
        )

    if use_hyde:
        hyde_text = generate_hyde(query)
        if hyde_text:
            result_lists.append(
                vector_store.query(hyde_text, top_k=config.VECTOR_TOP_K, law_id=law_id,
                                    require_active=require_active)
            )

    fused = _rrf_fuse(result_lists)
    return fused[:top_k]
