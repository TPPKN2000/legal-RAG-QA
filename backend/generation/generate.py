"""
Final generation step + grounding verification pass (design doc §7.1).

`predict_outcome()` is the only function `pipeline.py` needs to call: it
builds the prompt, calls the LLM, parses the required JSON schema, and runs
a verification pass that strips any citation the model invented outside the
set of law provisions it was actually shown (hallucination guard).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from backend import config
from backend.generation.compress import compress_case_evidence
from backend.generation.prompt_builder import allowed_citation_keys, build_prediction_prompt
from backend.models import CaseEvidenceHit, LawEvidenceItem, Prediction, RetrievedChunk, generate_text

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class OutcomePrediction:
    prediction: Prediction
    law_citations: list[LawEvidenceItem] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    dropped_hallucinated_citations: int = 0


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
    claim not established) with zero confidence, flagged in reasoning."""
    return OutcomePrediction(
        prediction="B_WIN",
        law_citations=[],
        confidence=0.0,
        reasoning=f"[fallback] {reason}",
    )


def predict_outcome(
    case_query: str,
    law_chunks: list[RetrievedChunk],
    case_evidence_hits: list[CaseEvidenceHit],
    max_new_tokens: int = 700,
    temperature: float = 0.2,
) -> OutcomePrediction:
    compressed_texts = compress_case_evidence([h.text for h in case_evidence_hits])
    system_prompt, user_prompt = build_prediction_prompt(
        case_query, law_chunks, case_evidence_hits, compressed_texts
    )

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

    prediction = parsed.get("prediction")
    if prediction not in config.VALID_PREDICTIONS:
        return _safe_default(f"invalid prediction label from model: {prediction!r}")

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

    return OutcomePrediction(
        prediction=prediction,
        law_citations=kept,
        confidence=float(parsed.get("confidence", 0.0) or 0.0),
        reasoning=str(parsed.get("reasoning", "")),
        dropped_hallucinated_citations=dropped,
    )
