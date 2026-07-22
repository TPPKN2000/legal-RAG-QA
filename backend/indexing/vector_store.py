"""
Pinecone-backed vector store for the law corpus.

Per the README, the project uses Pinecone rather than a self-hosted vector
DB. Metadata filtering (design doc §3.2 — effective/expired provisions,
document type) is pushed down into Pinecone's native metadata filter so
expired law is excluded *before* the ANN search runs, not after.

system_adjustments_v4.md §3.1 (SPEED fix, applied here): `_get_index()` is now
`@lru_cache`d — see its docstring below for why this was the dominant
source of the ~197s/case average observed in test/test_all_backend.log.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from backend import config
from backend.indexing.embed import embed_query, embed_texts
from backend.models import LawChunk, RetrievedChunk


@lru_cache(maxsize=1)
def _get_client():
    from pinecone import Pinecone

    if not config.PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Add it to your .env file (see README §Configuration)."
        )
    return Pinecone(api_key=config.PINECONE_API_KEY)


def ensure_index() -> None:
    """Create the Pinecone index if it doesn't already exist."""
    from pinecone import ServerlessSpec

    pc = _get_client()
    existing = {i["name"] for i in pc.list_indexes()}
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )


@lru_cache(maxsize=1)
def _get_index():
    """Cache the Index client (system_adjustments_v4.md §3.1).

    Constructing `Pinecone(...).Index(...)` triggers a `describe_index()`
    network round-trip inside the SDK, and `ensure_index()` above adds a
    `list_indexes()` round-trip on top of that. Before this fix, `_get_index()`
    had NO caching (unlike `_get_client()`, which already used `lru_cache`),
    so it re-paid both round-trips on *every single call* to `query()` /
    `upsert_chunks()` / `delete_namespace()`.

    With ~14-30 `vector_store.query()` calls per case (BM25+vector run for
    every query-rewrite variant, every decomposed sub-query, and again in
    the retrieval-evaluator's extra round when triggered), that redundant
    network chatter was the dominant cost in the ~197s/case average recorded
    in test/test_all_backend.log — the log shows "Listing indexes" firing
    30+ times per case, once per call, instead of once per process.

    This is a pure caching change: it does not alter retrieval logic,
    filtering, or results. See `reset_index_cache()` below for the one case
    (re-creating/renaming the index mid-process) where the cache needs to be
    invalidated.
    """
    ensure_index()
    return _get_client().Index(config.PINECONE_INDEX_NAME)


def reset_index_cache() -> None:
    """Invalidate the cached Pinecone client/Index.

    Only needed if a single process needs to point at a different index or
    re-create one that was just deleted (e.g. a test suite that calls
    `delete_namespace()` then rebuilds under a different
    `config.PINECONE_INDEX_NAME`) — normal query/upsert usage never needs
    this.
    """
    _get_index.cache_clear()
    _get_client.cache_clear()


def _chunk_metadata(chunk: LawChunk, extra: Optional[dict] = None) -> dict:
    meta = {
        "law_id": chunk.law_id,
        "aid": chunk.aid,
        "level": chunk.level,
        "parent_id": chunk.parent_id or "",
        "breadcrumb": chunk.breadcrumb,
        "text": chunk.text,
    }
    if extra:
        meta.update(extra)
    return meta


def upsert_chunks(
    chunks: list[LawChunk],
    status_by_law: Optional[dict[str, str]] = None,
    batch_size: int = 100,
) -> int:
    """Embed and upsert child chunks only (parents are kept in a local lookup
    table, see `chunker.build_parent_lookup`, and re-attached at generation
    time rather than searched directly, to keep the index focused on the
    granular units queries actually match).
    """
    index = _get_index()
    child_chunks = [c for c in chunks if c.level == "child"]
    status_by_law = status_by_law or {}

    count = 0
    for i in range(0, len(child_chunks), batch_size):
        batch = child_chunks[i : i + batch_size]
        vectors = embed_texts([c.text for c in batch])
        upserts = []
        for chunk, vec in zip(batch, vectors):
            extra = {"status": status_by_law.get(chunk.law_id, "unknown")}
            upserts.append(
                {
                    "id": chunk.chunk_id,
                    "values": vec.tolist(),
                    "metadata": _chunk_metadata(chunk, extra),
                }
            )
        index.upsert(vectors=upserts, namespace=config.PINECONE_NAMESPACE)
        count += len(upserts)
    return count


def query(
    text: str,
    top_k: int = config.VECTOR_TOP_K,
    law_id: Optional[str] = None,
    require_active: bool = True,
) -> list[RetrievedChunk]:
    """Vector search with optional hard metadata filter, applied server-side
    by Pinecone (not a post-hoc filter) so it doesn't eat into top_k."""
    index = _get_index()
    vec = embed_query(text)

    flt: dict = {}
    if law_id:
        flt["law_id"] = {"$eq": law_id}
    if require_active:
        flt["status"] = {"$in": ["active", "unknown"]}

    res = index.query(
        vector=vec,
        top_k=top_k,
        namespace=config.PINECONE_NAMESPACE,
        include_metadata=True,
        filter=flt or None,
    )

    out = []
    for match in res.get("matches", []):
        md = match.get("metadata", {})
        out.append(
            RetrievedChunk(
                chunk_id=match["id"],
                law_id=md.get("law_id", ""),
                aid=int(md.get("aid", -1)),
                text=md.get("text", ""),
                score=float(match.get("score", 0.0)),
                source="vector",
            )
        )
    return out


def delete_namespace() -> None:
    """Wipe the whole namespace — useful when re-ingesting the corpus."""
    _get_index().delete(delete_all=True, namespace=config.PINECONE_NAMESPACE)
