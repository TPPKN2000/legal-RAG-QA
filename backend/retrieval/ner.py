"""
Named-entity recognition helper (design doc / pipeline.txt step 2,
legalrag_adjustments.md §3).

Used for two purposes:
  1. `mask_person_org_entities` — strip plaintiff/defendant names and
     organization names out of the query used for LAW retrieval, since a
     person's name is pure noise when searching for the applicable statute
     (it never appears in law text) and can distract both BM25 and the
     embedding model.
  2. `extract_entities` is also available standalone for callers that want
     the raw entity spans (e.g. to enrich logging or breadcrumbs later).

This intentionally does NOT generate any text (unlike the HyDE step it
replaces) — it only extracts and rearranges spans that are already present
in the input, so it carries none of HyDE's hallucination risk.
"""
from __future__ import annotations

from functools import lru_cache

from backend import config

# Entity groups considered noise for law retrieval. The exact label set
# returned by NlpHUST/ner-vietnamese-electra-base is BIO-tagged; the
# "simple" aggregation strategy collapses it to group names like PERSON/ORG.
_MASKED_ENTITY_GROUPS = ("PERSON", "ORGANIZATION", "ORG")


@lru_cache(maxsize=1)
def _get_ner_pipeline():
    from transformers import pipeline

    return pipeline(
        "token-classification",
        model=config.NER_MODEL_NAME,
        aggregation_strategy="simple",
    )


def extract_entities(text: str) -> list[dict]:
    """Return [{"text":..., "entity_group":..., "start":..., "end":..., "score":...}, ...].

    Fail-safe: a model-loading or inference error must never block
    retrieval — callers should treat an empty list as "no entities found",
    not as an error.
    """
    if not text.strip():
        return []
    try:
        return list(_get_ner_pipeline()(text))
    except Exception:
        return []


def mask_person_org_entities(text: str, entities: list[dict] | None = None) -> str:
    """Replace PERSON/ORGANIZATION spans with a neutral placeholder.

    Used when building the query for LAW retrieval (not for the Case
    Content API, where names and specifics are exactly what a query needs —
    see hybrid_search.py for where this is and isn't applied).
    """
    entities = entities if entities is not None else extract_entities(text)
    spans = sorted(
        [e for e in entities if e.get("entity_group") in _MASKED_ENTITY_GROUPS and "start" in e and "end" in e],
        key=lambda e: e["start"],
        reverse=True,
    )
    for e in spans:
        text = text[: e["start"]] + "[BÊN LIÊN QUAN]" + text[e["end"] :]
    return text
