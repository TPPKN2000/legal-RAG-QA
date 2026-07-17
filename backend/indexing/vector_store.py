"""
Pinecone-backed vector store for the law corpus.

Per the README, the project uses Pinecone rather than a self-hosted vector
DB. Metadata filtering (design doc §3.2 — effective/expired provisions,
document type) is pushed down into Pinecone's native metadata filter so
expired law is excluded *before* the ANN search runs, not after.
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


def _get_index():
    ensure_index()
    return _get_client().Index(config.PINECONE_INDEX_NAME)


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
