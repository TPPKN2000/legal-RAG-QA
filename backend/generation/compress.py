"""
Prompt compression (design doc §4.2).

CRITICAL legal-domain rule: NEVER compress the verbatim text of a law
provision. Losing a connective like "trừ trường hợp" ("except in the case
of") or "ngoại trừ" ("excluding") can invert the meaning of a clause. Only
the *auxiliary* context — case-evidence padding, retrieved-but-secondary
passages, restated instructions — is eligible for compression.

Implementation uses LLMLingua when available; falls back to a cheap
sentence-count truncation (never token-level truncation, to avoid cutting
mid-clause) if `llmlingua` isn't installed, so the pipeline still runs in
constrained environments.
"""
from __future__ import annotations

from functools import lru_cache

from backend import config


@lru_cache(maxsize=1)
def _get_compressor():
    from llmlingua import PromptCompressor

    return PromptCompressor()


def _sentence_fallback_compress(text: str, target_ratio: float) -> str:
    """Drop trailing sentences until under the target ratio. Never splits a
    sentence mid-way — legal auxiliary text still deserves whole-sentence
    integrity even in the degraded fallback path."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    if not sentences:
        return text
    keep_n = max(1, round(len(sentences) * target_ratio))
    return ". ".join(sentences[:keep_n]) + "."


def compress_auxiliary_text(
    text: str,
    target_ratio: float = config.COMPRESSION_TARGET_RATIO,
) -> str:
    """Compress a block of AUXILIARY context only. Callers must never pass
    verbatim law-provision text here — see `prompt_builder.py`, which keeps
    law text and auxiliary text in separate fields precisely so this
    function is never accidentally applied to statute text."""
    if not config.COMPRESSION_ENABLED or not text.strip():
        return text
    try:
        compressor = _get_compressor()
        result = compressor.compress_prompt(text, rate=target_ratio)
        return result.get("compressed_prompt", text)
    except Exception:
        return _sentence_fallback_compress(text, target_ratio)


def compress_case_evidence(evidence_texts: list[str], target_ratio: float = config.COMPRESSION_TARGET_RATIO) -> list[str]:
    """Compress each case-content evidence segment independently.

    Case-content segments (facts, testimony, procedural history from the
    Case Content API) are auxiliary to the *legal* reasoning even though
    they are central to the *factual* reasoning — they are compressible,
    unlike law-provision text, because paraphrasing a witness statement
    doesn't change which statute applies, only how much detail survives.
    """
    return [compress_auxiliary_text(t, target_ratio) for t in evidence_texts]
