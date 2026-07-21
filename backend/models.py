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
    choice (no vLLM/Ollama server hop).

    legalrag_adjustments.md §2 — safety notes for the <1B generation model
    (previously Qwen3-8B, now config.GENERATION_MODEL_NAME, confirmed
    Qwen3.5-0.8B-class):
      - No `device_map`: that kwarg exists to shard a large model across
        multiple GPUs and is unnecessary (and a source of version-dependent
        accelerate/transformers errors) for a sub-1B model. Load normally
        and `.to(device)`.
      - `dtype` vs `torch_dtype`: transformers is mid-deprecation between the
        two kwarg names across versions; try the new name first and fall
        back to the old one rather than hard-coding either.
      - Explicit `attn_implementation="sdpa"`: avoids an import-time crash if
        transformers auto-selects flash-attention-2 but `flash-attn` isn't
        installed. sdpa ships with torch itself.
      - Loud failure if `tokenizer.chat_template` is missing, which usually
        means the configured repo id points at a base (non-chat) checkpoint
        rather than an "-Instruct"/"-Chat" one — much clearer than the
        AttributeError that would otherwise surface deep inside
        `apply_chat_template`.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = config.DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL_NAME, trust_remote_code=True)
    if tokenizer.chat_template is None:
        raise RuntimeError(
            f"{config.GENERATION_MODEL_NAME} has no chat_template — check that "
            "GENERATION_MODEL_NAME points at an '-Instruct'/'-Chat' checkpoint, "
            "not a base model."
        )

    dtype_val = torch.float16 if device.startswith("cuda") else torch.float32
    load_kwargs = dict(trust_remote_code=True, attn_implementation=config.GENERATION_ATTN_IMPL)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.GENERATION_MODEL_NAME, dtype=dtype_val, **load_kwargs
        )
    except TypeError:
        # Older transformers versions don't accept `dtype=` yet.
        model = AutoModelForCausalLM.from_pretrained(
            config.GENERATION_MODEL_NAME, torch_dtype=dtype_val, **load_kwargs
        )

    model = model.to(device)  # no device_map for a sub-1B model
    model.eval()
    return tokenizer, model, device


def generate_text(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = config.GENERATION_MAX_NEW_TOKENS_DEFAULT,
    temperature: float = 0.3,
    top_p: float = 0.9,
) -> str:
    """Single-turn chat-style generation used by query rewriting, NER-grounded
    query decomposition, the case-fact digest step, and final verdict
    generation. Uses the tokenizer's chat template so it works consistently
    across Qwen3.x checkpoints without hand-rolled prompt formatting.

    Raises on failure (deliberately) — callers that must degrade gracefully
    (query rewriting, query decomposition) already wrap this in try/except;
    callers that cannot degrade (outcome prediction) should let it propagate.
    """
    import torch

    tokenizer, model, device = _get_generation_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    template_kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        # legalrag_adjustments.md §2 point 3: Qwen3-family chat templates
        # default "thinking" mode on, which can burn the whole
        # max_new_tokens budget on a <think>...</think> block before any
        # JSON is emitted. Explicitly disable it (config-controlled).
        encoded = tokenizer.apply_chat_template(
            messages, enable_thinking=config.GENERATION_ENABLE_THINKING, **template_kwargs
        )
    except TypeError:
        # Non-Qwen3 tokenizer/template that doesn't accept enable_thinking.
        encoded = tokenizer.apply_chat_template(messages, **template_kwargs)

  
    # Depending on the tokenizer/transformers version, apply_chat_template can
    # return either a tensor of input IDs or a BatchEncoding/dict containing
    # input_ids plus an attention_mask. Passing a BatchEncoding positionally to
    # model.generate makes transformers treat the whole object as inputs_tensor,
    # which then crashes because BatchEncoding has no .shape. Normalize both
    # forms before generation.
    encoded = encoded.to(device) if hasattr(encoded, "to") else encoded
    if isinstance(encoded, dict):
        model_inputs = dict(encoded)
        input_length = model_inputs["input_ids"].shape[-1]
        generate_args = ()
        generate_kwargs = model_inputs
    else:
        input_length = encoded.shape[-1]
        generate_args = (encoded,)
        generate_kwargs = {}

    do_sample = temperature > 0
    with torch.no_grad():
        output = model.generate(
            *generate_args,
            **generate_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output[0][input_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Defensive cleanup in case a <think> block still leaks through (e.g. the
    # enable_thinking kwarg was silently ignored by a particular checkpoint).
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text
