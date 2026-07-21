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

legalrag_adjustments.md §6b: a Khoản/Điểm that is itself unusually long (a
long enumerated clause) is no longer kept fully intact — it is soft-split on
sentence boundaries (never a hard token/char cut, same rule-based-only
principle as everywhere else in this module) once it exceeds
config.CHILD_MAX_CHARS, matching the threshold ViDRILL found effective for
Vietnamese legal text (~450 words / ~900 chars).

legalrag_adjustments.md §6a: `build_parent_lookup()` used to be dead code —
nothing called it. `scripts/build_index.py` now calls it and persists the
result to `config.PARENT_LOOKUP_PATH`, and `pipeline.py` loads it to give
the cross-encoder reranker full-article context (see pipeline.py's
`collect_law_evidence`), which is then swapped back to the shorter,
citation-precise child text before it reaches the final generation prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from backend import config
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


# Sentence-boundary split used only as a soft-split fallback for oversized
# Khoản/Điểm text — never used to chunk normal-sized clauses. Splits after
# '.' or ';' followed by whitespace, which keeps "trừ trường hợp..."-style
# qualifiers attached to the clause they govern in the common case.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.;])\s+")


def _soft_split_oversized(text: str, max_chars: int = config.CHILD_MAX_CHARS) -> list[str]:
    """Split `text` into pieces no larger than `max_chars`, breaking only at
    sentence boundaries. Returns [text] unchanged if it's already short
    enough, or as a last resort if it has no detectable sentence boundaries
    (never force-splits mid-sentence)."""
    if len(text) <= max_chars:
        return [text]

    sentences = [s for s in _SENTENCE_BOUNDARY_RE.split(text) if s]
    if not sentences:
        return [text]

    parts: list[str] = []
    buf = ""
    for s in sentences:
        if buf and len(buf) + 1 + len(s) > max_chars:
            parts.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}".strip() if buf else s
    if buf:
        parts.append(buf.strip())
    return parts or [text]


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

        # legalrag_adjustments.md §6b: soft-split an oversized Khoản/Điểm
        # into multiple child chunks on sentence boundaries rather than
        # keeping it intact. Each sub-part still traces back to the same
        # (law_id, aid) for citation purposes.
        sub_parts = _soft_split_oversized(split.text)
        for j, part_text in enumerate(sub_parts):
            part_suffix = suffix if len(sub_parts) == 1 else f"{suffix}_p{j}"
            child_id = f"{parent_id}{part_suffix}"
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
                    text=part_text,
                    token_count=_count_tokens(part_text),
                )
            )
    return chunks


def chunk_articles(articles: list[RawArticle]) -> list[LawChunk]:
    """Chunk a list of articles. Oversized Khoản/Điểm are soft-split on
    sentence boundaries (see `_soft_split_oversized`) rather than force-split
    on a fixed token window — legal text should never be truncated or cut
    mid-sentence."""
    all_chunks: list[LawChunk] = []
    for art in articles:
        all_chunks.extend(chunk_article(art))
    return all_chunks


def build_parent_lookup(chunks: list[LawChunk]) -> dict[str, LawChunk]:
    """Map parent_id -> parent LawChunk, so retrieval can re-attach full
    article context to a matched child chunk at generation time.

    legalrag_adjustments.md §6a: this is now actually wired up —
    `scripts/build_index.py` persists the result to
    `config.PARENT_LOOKUP_PATH`, and `pipeline.py.collect_law_evidence`
    loads it to give the cross-encoder reranker full-article context.
    """
    return {c.chunk_id: c for c in chunks if c.level == "parent"}
