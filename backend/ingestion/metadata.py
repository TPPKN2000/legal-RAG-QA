"""
Metadata extraction & enrichment for law documents.

This is the layer that makes "metadata filtering before vector search"
(design doc §3.2) possible: every chunk gets attached a `LawMetadata` record
so retrieval can exclude expired/superseded provisions before scoring.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

from backend.ingestion.parser import RawLawDocument
from backend.models import LawMetadata

# Cross-reference patterns: "Điều 12 Luật ...", "khoản 3 Điều 5 Nghị định .../.../..."
RE_CROSS_REF = re.compile(
    r"(?:Điều\s+(\d+)(?:\s*,\s*khoản\s+(\d+))?)\s*(?:của\s+)?"
    r"([\w\d]+/\d{4}/[A-ZĐ\-]+)?",
    re.IGNORECASE,
)

# Amendment/repeal signal phrases — used to flag a document's status when the
# organizers haven't supplied an explicit `status` field.
RE_REPEALED = re.compile(r"hết hiệu lực|bị bãi bỏ|thay thế bởi", re.IGNORECASE)
RE_AMENDED = re.compile(r"sửa đổi, bổ sung|được sửa đổi bởi", re.IGNORECASE)


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def infer_status(doc: RawLawDocument, as_of: Optional[date] = None) -> str:
    """Infer active/expired/amended status purely from dates when the
    organizer feed doesn't provide an explicit status field.

    NOTE: this is a conservative heuristic, not a substitute for an actual
    legal-status database. Prefer an explicit `status` field if the corpus
    provides one.
    """
    as_of = as_of or date.today()
    expiry = _parse_date(doc.expiry_date)
    effective = _parse_date(doc.effective_date)

    if expiry and expiry <= as_of:
        return "expired"
    if effective and effective > as_of:
        return "unknown"  # not yet in force
    return "active"


def build_metadata(doc: RawLawDocument, superseded_by: Optional[str] = None,
                    supersedes: Optional[str] = None) -> LawMetadata:
    return LawMetadata(
        law_id=doc.law_id,
        doc_type=doc.doc_type,
        issuing_body=doc.issuing_body,
        issue_date=doc.issue_date,
        effective_date=doc.effective_date,
        expiry_date=doc.expiry_date,
        status=infer_status(doc),
        superseded_by=superseded_by,
        supersedes=supersedes,
    )


def extract_cross_references(article_text: str, default_law_id: str) -> list[dict]:
    """Extract simple cross-references such as "Điều 12 Nghị định 145/2020/NĐ-CP".

    Returns a list of {"law_id": ..., "aid": ...} dicts. Cross references that
    don't specify a law number are assumed to point within `default_law_id`.
    This enrichment supports the "reference resolution between articles"
    requirement in design doc §2.2, letting the generation step optionally
    pull in the referenced article as extra context.
    """
    refs = []
    for m in RE_CROSS_REF.finditer(article_text):
        aid_str, _khoan, law_num = m.groups()
        if not aid_str:
            continue
        refs.append({
            "law_id": law_num or default_law_id,
            "aid": int(aid_str),
        })
    return refs


def metadata_filter(
    metadatas: dict[str, LawMetadata],
    law_id: Optional[str] = None,
    require_active: bool = True,
    doc_type: Optional[str] = None,
) -> set[str]:
    """Return the set of law_ids passing a hard metadata filter, to be applied
    BEFORE vector/BM25 search (design doc §3.2) rather than after, to avoid
    wasting retrieval budget scoring irrelevant/expired documents.
    """
    allowed = set()
    for lid, meta in metadatas.items():
        if law_id and lid != law_id:
            continue
        if doc_type and meta.doc_type != doc_type:
            continue
        if require_active and meta.status not in ("active", "unknown"):
            continue
        allowed.add(lid)
    return allowed
