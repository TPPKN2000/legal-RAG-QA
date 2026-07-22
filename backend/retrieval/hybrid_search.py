"""
Hybrid search (design doc §3.1): fuse BM25 and Pinecone vector results with
Reciprocal Rank Fusion (RRF) rather than a weighted score sum, because BM25
scores and cosine similarities live on incomparable scales — RRF sidesteps
that by fusing on *rank* instead of raw score.

    RRF(d) = sum over retrievers r of  weight_r / (k + rank_r(d))

`k` (config.RRF_K) is the standard damping constant (60 is the commonly used
default from the original RRF paper) that keeps a single retriever's #1 hit
from completely dominating the fused ranking. `weight_r` defaults to 1.0 for
every channel; the query-decomposition channel gets a higher weight (see
system_adjustments_v3.md §3 / §7 "Weighted RRF theo Judge-R1").

system_adjustments_v3.md §3 — HyDE removed. It used to be a 3rd retrieval
channel here (a generated hypothetical statute passage, embedded and
searched). It has been replaced with two non-generative-content additions:
  1. NER-based entity masking of the query used for LAW retrieval (plaintiff
     / defendant names are noise for statute search).
  2. A decomposition channel: instead of one generated passage, the case
     query is broken into several short legal-aspect questions (no invented
     statute text) and each is run through vector search, fused in with a
     higher RRF weight (Judge-R1-style agentic route).
"""
from __future__ import annotations

from typing import Optional

from backend import config
from backend.indexing import bm25_index, vector_store
from backend.models import RetrievedChunk
from backend.retrieval.ner import extract_entities, mask_person_org_entities
from backend.retrieval.querry_transform import decompose_query, rewrite_query


def _rrf_fuse_weighted(
    channels: list[tuple[list[RetrievedChunk], float]],
    k: int = config.RRF_K,
) -> list[RetrievedChunk]:
    """Fuse several (result_list, weight) channels by weighted reciprocal
    rank. weight=1.0 for every channel reproduces the original unweighted
    RRF used before query decomposition was introduced."""
    scores: dict[str, float] = {}
    best_chunk: dict[str, RetrievedChunk] = {}

    for results, weight in channels:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + weight / (k + rank)
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
    use_decomposition: bool = True,
    top_k: int = 30,
) -> list[RetrievedChunk]:
    """Run BM25 + vector search (optionally over multiple query rewrites and
    NER-decomposed legal-aspect sub-queries), then fuse everything with
    weighted RRF.

    Metadata filtering (law_id / active-only) is pushed down into both the
    BM25 index and the Pinecone query themselves (design doc §3.2), not
    applied after fusion, so it never silently reduces an already-truncated
    top_k.
    """
    bm25 = bm25_index.get_bm25_index()

    # Party/organization names are noise for statute search — mask them out
    # of the query used for the standard BM25+vector channels. (The Case
    # Content API query, built separately in pipeline.py, intentionally
    # keeps names since that API is looking for the matching case segment.)
    entities = extract_entities(query)
    law_query = mask_person_org_entities(query, entities)

    channels: list[tuple[list[RetrievedChunk], float]] = []

    base_queries = rewrite_query(law_query) if use_query_rewriting else [law_query]
    for q in base_queries:
        channels.append((
            bm25.query(q, top_k=config.BM25_TOP_K, law_id=law_id, require_active=require_active),
            config.RRF_WEIGHT_STANDARD,
        ))
        channels.append((
            vector_store.query(q, top_k=config.VECTOR_TOP_K, law_id=law_id, require_active=require_active),
            config.RRF_WEIGHT_STANDARD,
        ))

    if use_decomposition and config.QUERY_DECOMPOSITION_ENABLED:
        for sub_q in decompose_query(
            query, masked_query=law_query, n_subqueries=config.QUERY_DECOMPOSITION_MAX_SUBQUERIES
        ):
            channels.append((
                vector_store.query(sub_q, top_k=config.VECTOR_TOP_K, law_id=law_id, require_active=require_active),
                config.RRF_WEIGHT_AGENT,
            ))

    fused = _rrf_fuse_weighted(channels)
    return fused[:top_k]
