"""
Hierarchical chunking for the law corpus (design doc §2.2).

Parent chunk  = the whole Điều (article) — used to re-attach full context
                to the LLM once a child chunk has been retrieved.
Child chunk   = an individual Khoản/Điểm inside the article — this is the
                unit that actually gets embedded and searched, since a
                query like "phạt bao nhiêu tiền khi..." usually matches one
                specific clause, not the whole article.

Chunking is entirely rule-based on the Chương > Mục > Điều > Khoản > Điểm
structure (never a fixed token-window split), because a legal clause's
meaning frequently hinges on a preceding qualifier like "trừ trường hợp"
that a hard token cut could separate from its clause.
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.ingestion.parser import RawArticle, split_article_into_khoan_diem
from backend.models import LawChunk

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except ImportError:
    def _count_tokens(text: str) -> int:
        # Rough fallback: ~1 token per 3-4 Vietnamese UTF-8 characters.
        return max(1, len(text) // 4)


def _breadcrumb(law_id: str, chuong: str | None, muc: str | None, aid: int,
                 khoan_no: str | None = None, diem_no: str | None = None) -> str:
    parts = [f"Luật {law_id}"]
    if chuong:
        parts.append(chuong)
    if muc:
        parts.append(muc)
    parts.append(f"Điều {aid}")
    if khoan_no:
        parts.append(f"Khoản {khoan_no}")
    if diem_no:
        parts.append(f"Điểm {diem_no}")
    return " > ".join(parts)


def chunk_article(article: RawArticle) -> list[LawChunk]:
    """Produce one parent chunk + N child chunks for a single article."""
    chunks: list[LawChunk] = []

    parent_id = f"{article.law_id}_a{article.aid}"
    parent_text = (article.title + "\n" + article.body).strip()
    chunks.append(
        LawChunk(
            chunk_id=parent_id,
            law_id=article.law_id,
            aid=article.aid,
            breadcrumb=_breadcrumb(article.law_id, article.chuong, article.muc, article.aid),
            level="parent",
            parent_id=None,
            text=parent_text,
            token_count=_count_tokens(parent_text),
        )
    )

    splits = split_article_into_khoan_diem(article.body)
    for i, split in enumerate(splits):
        if not split.text:
            continue
        suffix = f"_k{split.khoan_no or i}"
        if split.diem_no:
            suffix += f"_d{split.diem_no}"
        child_id = f"{parent_id}{suffix}"
        chunks.append(
            LawChunk(
                chunk_id=child_id,
                law_id=article.law_id,
                aid=article.aid,
                breadcrumb=_breadcrumb(
                    article.law_id, article.chuong, article.muc, article.aid,
                    split.khoan_no, split.diem_no,
                ),
                level="child",
                parent_id=parent_id,
                text=split.text,
                token_count=_count_tokens(split.text),
            )
        )
    return chunks


def chunk_articles(articles: list[RawArticle]) -> list[LawChunk]:
    """Chunk a list of articles, oversized-child fallback included.

    If a single Khoản/Điểm still exceeds `PARENT_CHUNK_MAX_TOKENS` (rare,
    e.g. an unusually long enumerated clause), it is kept intact rather than
    force-split — legal text should never be truncated mid-sentence.
    """
    all_chunks: list[LawChunk] = []
    for art in articles:
        all_chunks.extend(chunk_article(art))
    return all_chunks


def build_parent_lookup(chunks: list[LawChunk]) -> dict[str, LawChunk]:
    """Map parent_id -> parent LawChunk, so retrieval can re-attach full
    article context to a matched child chunk at generation time."""
    return {c.chunk_id: c for c in chunks if c.level == "parent"}
