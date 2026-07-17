"""
Pydantic schemas shared across ingestion, indexing, retrieval and generation.
Keeping these in one place avoids circular imports between pipeline stages.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Prediction = Literal["A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"]


# ---------------------------------------------------------------------------
# Ingestion / law corpus structures
# ---------------------------------------------------------------------------
class LawChunk(BaseModel):
    """A single retrievable unit of the law corpus.

    Parent chunks correspond to a full "Điều" (article); child chunks
    correspond to individual "Khoản"/"Điểm" nested inside it. Child chunks
    carry a `parent_id` link so the parent's full context can be re-attached
    at generation time without having to embed the whole article as one
    (overly long / overly diluted) vector.
    """

    chunk_id: str
    law_id: str
    aid: int = Field(..., description="Article id within the law corpus (matches submission `aid`).")
    breadcrumb: str = Field(..., description='e.g. "Chương II > Mục 1 > Điều 12 > Khoản 3"')
    level: Literal["parent", "child"] = "child"
    parent_id: Optional[str] = None
    text: str
    token_count: int = 0


class LawMetadata(BaseModel):
    law_id: str
    doc_type: Optional[str] = None          # Luật / Nghị định / Thông tư / ...
    issuing_body: Optional[str] = None
    issue_date: Optional[str] = None        # ISO 8601
    effective_date: Optional[str] = None    # ISO 8601
    expiry_date: Optional[str] = None       # ISO 8601, null if still in force
    status: Literal["active", "expired", "amended", "unknown"] = "unknown"
    superseded_by: Optional[str] = None
    supersedes: Optional[str] = None


# ---------------------------------------------------------------------------
# Case query / test set
# ---------------------------------------------------------------------------
class CaseQuery(BaseModel):
    case_id: str
    case_query: str
    # Segment count n_i, if provided by the test set; used for API budget math.
    n_segments: Optional[int] = None


# ---------------------------------------------------------------------------
# Retrieval results
# ---------------------------------------------------------------------------
class RetrievedChunk(BaseModel):
    chunk_id: str
    law_id: str
    aid: int
    text: str
    score: float
    source: Literal["bm25", "vector", "fused", "reranked"] = "fused"


class CaseEvidenceHit(BaseModel):
    """One result from the Case Content API `/retrieve` call."""

    chunk_id: str
    text: str
    score: float


# ---------------------------------------------------------------------------
# Submission output
# ---------------------------------------------------------------------------
class LawEvidenceItem(BaseModel):
    law_id: str
    aid: int


class SubmissionRecord(BaseModel):
    case_id: str
    prediction: Prediction
    case_evidence: list[str] = Field(default_factory=list)
    law_evidence: list[LawEvidenceItem] = Field(default_factory=list)

    @field_validator("case_evidence")
    @classmethod
    def _dedup_case_evidence(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for x in v:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    @field_validator("law_evidence")
    @classmethod
    def _dedup_law_evidence(cls, v: list[LawEvidenceItem]) -> list[LawEvidenceItem]:
        seen, out = set(), []
        for item in v:
            key = (item.law_id, item.aid)
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out
