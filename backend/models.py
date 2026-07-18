"""
Single shared module for:
  1. Pydantic schemas used across ingestion, indexing, retrieval and
     generation (keeping these in one place avoids circular imports between
     pipeline stages).
  2. The central LLM loading/generation interface (`generate_text`), used by
     query rewriting, HyDE, and final outcome prediction.

NOTE: an earlier version of this codebase split these into `backend/model.py`
(singular, LLM loader) and `backend/models.py` (plural, schemas) — a naming
accident that made every other import in the repo point at whichever file
didn't have what it needed. Consolidated back into this single file per
request; every `from backend.models import ...` statement across the repo
already points here.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from backend import config

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


# ---------------------------------------------------------------------------
# LLM loading & generation
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_generation_model():
    """Load the causal LM + tokenizer once and cache them for the process
    lifetime. HF `transformers` is loaded in-process per the confirmed
    choice (no vLLM/Ollama server hop)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.GENERATION_MODEL_NAME,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map=device if device.startswith("cuda") else None,
    )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_text(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """Single-turn chat-style generation used by query rewriting, HyDE, and
    final verdict generation. Uses the tokenizer's chat template so it works
    consistently across Qwen3-8B / Qwen3-4B without hand-rolled prompt
    formatting for each swap.

    Raises on failure (deliberately) — callers that must degrade gracefully
    (e.g. HyDE, query rewriting) already wrap this in try/except; callers
    that cannot degrade (outcome prediction) should let it propagate.
    """
    import torch

    tokenizer, model, device = _get_generation_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
