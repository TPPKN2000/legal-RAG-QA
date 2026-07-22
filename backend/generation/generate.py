"""
Final generation step + grounding verification pass (design doc §7.1).

`predict_outcome()` is the only function `pipeline.py` needs to call: it
builds the prompt, calls the LLM, parses the required JSON schema, and runs
a verification pass that strips any citation the model invented outside the
set of law provisions it was actually shown (hallucination guard).

legalrag_adjustments.md §5: this now takes a pre-built `case_digest` string
(see generation/case_digest.py) instead of raw case-evidence hits — the
caller (pipeline.py) is responsible for building the digest, since the same
case-evidence hits are also needed un-digested for the submission's
`case_evidence` field.

legalrag_adjustments.md §7 ("Neurosymbolic — rẻ, dễ implement"): after
parsing, a small rule-based check downgrades confidence when the model's
*self-reported* confidence isn't backed by any surviving grounded citation.
The system prompt already asks the model to self-report low confidence when
context is thin (rule #3), but nothing previously enforced that — a model
can claim high confidence with zero valid citations. This does not change
the predicted label (that would require case-specific legal judgement this
rule-based layer doesn't have); it only caps the confidence score so
downstream consumers of `confidence` aren't misled.

IMPROVEMENT_PLAN.md §3.4 (ACCURACY fix, applied here):
  1. `OutcomePrediction` now carries `is_fallback` / `fallback_reason` so
     callers (test/test_all_backend.py) can measure how often a case is
     forced to the conservative B_WIN default by a crash/parse failure,
     instead of that being indistinguishable from the model genuinely
     choosing B_WIN — the original diagnosis in IMPROVEMENT_PLAN.md §2.4
     couldn't tell these apart, which is exactly the instrumentation the
     plan's "Bước 1" calls for before attempting any fix to the label
     distribution.
  2. `predict_outcome()` now also reads the `accepted_ratio_estimate` field
     the model is asked for (see prompt_builder.SYSTEM_PROMPT rule #5) and,
     when `config.USE_RATIO_DERIVED_LABEL` is on, derives the label from
     that ratio via fixed thresholds — trusting it over the categorical
     `prediction` field on disagreement, and using it outright when
     `prediction` is missing/invalid. This targets IMPROVEMENT_PLAN.md
     §2.4's "Bước 2b": a quantitative estimate is a different kind of
     judgement call than picking cold from 4 categorical buckets, and is
     less prone to collapsing onto a "safe-looking" default under
     uncertainty (observed: A_WIN/PARTIAL_B_WIN never once appeared across
     50 public-test cases before this fix).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from backend import config
from backend.generation.prompt_builder import allowed_citation_keys, build_prediction_prompt
from backend.models import LawEvidenceItem, Prediction, RetrievedChunk, generate_text

log = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

# legalrag_adjustments.md §7: if zero citations survive the hallucination
# guard, cap self-reported confidence at this ceiling regardless of what the
# model claimed — an ungrounded prediction should never be reported as
# high-confidence.
_UNGROUNDED_CONFIDENCE_CEILING = 0.3


@dataclass
class OutcomePrediction:
    prediction: Prediction
    law_citations: list[LawEvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    dropped_hallucinated_citations: int = 0
    # IMPROVEMENT_PLAN.md §3.4: True iff this result came from _safe_default()
    # (generation/parsing crashed, or the model returned no usable label at
    # all) rather than a genuine model prediction — lets callers separate
    # "the model chose B_WIN" from "the pipeline crashed and we forced
    # B_WIN", which the raw submission schema can't distinguish.
    is_fallback: bool = False
    fallback_reason: str | None = None


def _extract_json(raw: str) -> dict:
    """The model is instructed to return ONLY JSON, but LLMs sometimes wrap
    it in prose or a code fence anyway — extract the first {...} block
    defensively rather than trusting `json.loads(raw)` directly."""
    raw = raw.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw[:200]!r}")
    return json.loads(match.group(0))


def _safe_default(reason: str) -> OutcomePrediction:
    """A parse/generation failure must never crash the whole submission run
    for one case — fall back to the most conservative label (B_WIN, i.e.
    claim not established) with zero confidence, flagged in reasoning and
    via `is_fallback`/`fallback_reason` (IMPROVEMENT_PLAN.md §3.4)."""
    return OutcomePrediction(
        prediction="B_WIN",
        law_citations=[],
        confidence=0.0,
        reasoning=f"[fallback] {reason}",
        is_fallback=True,
        fallback_reason=reason,
    )


def _parse_ratio(raw_ratio) -> float | None:
    """Best-effort parse of `accepted_ratio_estimate` into a float in
    [0.0, 1.0]. Returns None if missing/unparseable/out of range — callers
    should treat that as "no ratio-derived label available", not an error."""
    if raw_ratio is None:
        return None
    try:
        ratio = float(raw_ratio)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= ratio <= 1.0):
        return None
    return ratio


def _label_from_ratio(ratio: float) -> Prediction:
    """Fixed thresholds mirroring prompt_builder.SYSTEM_PROMPT rule #5."""
    if ratio > 0.99:
        return "A_WIN"
    if ratio > 0.5:
        return "PARTIAL_A_WIN"
    if ratio > 0.0:
        return "PARTIAL_B_WIN"
    return "B_WIN"


def predict_outcome(
    case_query: str,
    law_chunks: list[RetrievedChunk],
    case_digest: str,
    max_new_tokens: int = config.GENERATION_MAX_NEW_TOKENS_DEFAULT,
    temperature: float = 0.2,
) -> OutcomePrediction:
    system_prompt, user_prompt = build_prediction_prompt(case_query, law_chunks, case_digest)

    try:
        raw = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        parsed = _extract_json(raw)
    except Exception as e:
        return _safe_default(f"generation/parsing failed: {e}")

    raw_prediction = parsed.get("prediction")
    ratio_estimate = _parse_ratio(parsed.get("accepted_ratio_estimate"))
    ratio_derived_prediction = _label_from_ratio(ratio_estimate) if ratio_estimate is not None else None

    # IMPROVEMENT_PLAN.md §3.4 Bước 2b: prefer the ratio-derived label over
    # the categorical one whenever both are available and disagree, and use
    # the ratio outright when the categorical field is missing/invalid.
    prediction = raw_prediction
    if config.USE_RATIO_DERIVED_LABEL and ratio_derived_prediction is not None:
        if raw_prediction not in config.VALID_PREDICTIONS:
            prediction = ratio_derived_prediction
        elif raw_prediction != ratio_derived_prediction:
            log.info(
                "categorical/ratio label mismatch: model picked %r but "
                "accepted_ratio_estimate=%.2f implies %r -> using the "
                "ratio-derived label (IMPROVEMENT_PLAN.md §3.4)",
                raw_prediction, ratio_estimate, ratio_derived_prediction,
            )
            prediction = ratio_derived_prediction

    if prediction not in config.VALID_PREDICTIONS:
        return _safe_default(f"invalid prediction label from model: {raw_prediction!r}")

    allowed = allowed_citation_keys(law_chunks)
    raw_citations = parsed.get("law_citations") or []
    kept, dropped = [], 0
    for item in raw_citations:
        try:
            key = (str(item["law_id"]), int(item["aid"]))
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        if key in allowed:
            kept.append(LawEvidenceItem(law_id=key[0], aid=key[1]))
        else:
            dropped += 1  # hallucinated / outside retrieved context — drop it

    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    reasoning = str(parsed.get("reasoning", ""))
    if not kept and confidence > _UNGROUNDED_CONFIDENCE_CEILING:
        # Rule-based grounding check (legalrag_adjustments.md §7): the model
        # claimed more confidence than a zero-citation answer should get.
        reasoning = (
            f"[confidence capped: no grounded law citation survived verification] {reasoning}"
        )
        confidence = min(confidence, _UNGROUNDED_CONFIDENCE_CEILING)

    return OutcomePrediction(
        prediction=prediction,
        law_citations=kept,
        confidence=confidence,
        reasoning=reasoning,
        dropped_hallucinated_citations=dropped,
        is_fallback=False,
        fallback_reason=None,
    )
