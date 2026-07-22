"""
End-to-end orchestration for a single test case (design doc pipeline.py role).

process_case() ties together:
  1. Budget-aware Case Content API evidence collection (docs/evaluation.md §2.4 —
     no penalty up to 2*n_i calls, zero credit at 5*n_i).
  2. Law-corpus retrieval: hybrid_search (BM25 + Pinecone + weighted RRF) ->
     rerank (with parent-article context, legalrag_adjustments.md §6a) ->
     optional retrieval-evaluator re-round (legalrag_adjustments.md §7).
  3. A case-fact digest step (legalrag_adjustments.md §5) so the final
     generation call doesn't carry raw case-evidence text.
  4. Outcome generation with grounding verification.
  5. Assembly into a `SubmissionRecord` matching docs/submission_example.json.

`submission.py` is the CLI that loops this over the whole test set.

IMPROVEMENT_PLAN.md §3.4: `process_case()` now delegates to
`process_case_with_debug()`, which returns the `SubmissionRecord` alongside a
small debug dict (`is_fallback`, `fallback_reason`, `confidence`,
`dropped_hallucinated_citations`) sourced from `generate.OutcomePrediction`.
`SubmissionRecord` itself is NOT extended with these fields — it must stay
byte-for-byte compatible with docs/submission_example.json — so callers that
need this instrumentation (test/test_all_backend.py's fallback-rate report)
call `process_case_with_debug()` directly instead.
"""
from __future__ import annotations

import logging
import pickle
from functools import lru_cache

from backend import config
from backend.case_api_client import CaseAPIError, client as case_api_client
from backend.generation.case_digest import build_case_digest
from backend.generation.generate import predict_outcome
from backend.models import CaseEvidenceHit, CaseQuery, LawChunk, RetrievedChunk, SubmissionRecord
from backend.retrieval.hybrid_search import hybrid_search
from backend.retrieval.querry_transform import decompose_query, rewrite_query
from backend.retrieval.rerank import rerank

log = logging.getLogger(__name__)


def _case_api_budget(case: CaseQuery) -> int:
    """No-penalty ceiling for this case (docs/evaluation.md §2.4: B_i = 2*n_i).
    Falls back to a fixed cap when n_i isn't known ahead of time — which,
    per legalrag_adjustments.md §0/§1, is the common case: the private test
    set only exposes case_id + case_query, so this fallback branch is not a
    rare edge case, it's the default path at real scoring time."""
    if case.n_segments:
        return max(1, int(config.API_BUDGET_MULTIPLIER * case.n_segments))
    return config.DEFAULT_MAX_API_CALLS_PER_CASE


def collect_case_evidence(case: CaseQuery, max_queries: int | None = None) -> list[CaseEvidenceHit]:
    """Issue a budget-bounded sequence of queries to the Case Content API and
    return the deduplicated set of evidence segments retrieved.

    Strategy: derive several distinct search queries from the case_query via
    the same legal-register rewriting used for law retrieval (querry_transform
    is retrieval-purpose-agnostic), then query the API once per variant.
    Stops early if two consecutive calls return a chunk already seen — that's
    a signal the API has nothing new left for this line of querying, and
    burning further budget only hurts the efficiency factor E_i for no
    recall gain.
    """
    budget = max_queries or _case_api_budget(case)
    query_variants = rewrite_query(case.case_query, n_variants=budget - 1)[:budget] or [case.case_query]

    seen_chunk_ids: set[str] = set()
    evidence: list[CaseEvidenceHit] = []
    consecutive_repeats = 0

    for q in query_variants:
        if case_api_client.calls_made(case.case_id) >= budget:
            break
        try:
            hit = case_api_client.retrieve(query=q, case_id=case.case_id)
        except CaseAPIError as e:
            log.warning("case_api_client.retrieve failed for case=%s query=%r: %s", case.case_id, q, e)
            continue

        if hit is None:
            consecutive_repeats += 1
        elif hit.chunk_id in seen_chunk_ids:
            consecutive_repeats += 1
        else:
            seen_chunk_ids.add(hit.chunk_id)
            evidence.append(hit)
            consecutive_repeats = 0

        if consecutive_repeats >= 2:
            break

    return evidence


@lru_cache(maxsize=1)
def _get_parent_lookup() -> dict[str, LawChunk]:
    """Lazily load the parent (whole-Điều) chunk lookup persisted by
    scripts/build_index.py (legalrag_adjustments.md §6a). Returns {} if the
    index hasn't been (re)built with the current script yet, so rerank
    degrades gracefully to child-only text instead of crashing."""
    try:
        with open(config.PARENT_LOOKUP_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        log.warning(
            "parent lookup not found at %s — rerank will use child-chunk text only. "
            "Re-run scripts/build_index.py to generate it.",
            config.PARENT_LOOKUP_PATH,
        )
        return {}


def _enrich_with_parent_text(candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Swap in whole-article (parent) text for reranking only
    (legalrag_adjustments.md §6a: a cross-encoder scores a (query, chunk)
    pair more accurately with fuller context). The child chunk_id/aid are
    preserved so citations stay precise; text is swapped back to the child
    text after rerank (see collect_law_evidence) to keep the final
    generation prompt's law section short (legalrag_adjustments.md §5)."""
    parent_lookup = _get_parent_lookup()
    if not parent_lookup:
        return candidates
    enriched = []
    for c in candidates:
        parent = parent_lookup.get(f"{c.law_id}_a{c.aid}")
        enriched.append(c.model_copy(update={"text": parent.text}) if parent else c)
    return enriched


def collect_law_evidence(query_text: str) -> list[RetrievedChunk]:
    """Hybrid search (design doc §3) -> cross-encoder rerank on parent-article
    context (design doc §4.1, legalrag_adjustments.md §6a) -> child text
    restored for citation precision -> optional retrieval-evaluator re-round
    (legalrag_adjustments.md §7).
    """
    candidates = hybrid_search(query_text, top_k=config.RERANK_TOP_K)
    child_text_by_id = {c.chunk_id: c.text for c in candidates}

    reranked = rerank(query_text, _enrich_with_parent_text(candidates), top_k=config.FINAL_LAW_TOP_K)
    reranked = [
        c.model_copy(update={"text": child_text_by_id.get(c.chunk_id, c.text)}) for c in reranked
    ]

    # legalrag_adjustments.md §7 "Retrieval Evaluator -> core cho vòng lặp":
    # reuse the cross-encoder score already computed above as a cheap gate —
    # no extra LLM-judge call. If even the best reranked candidate scores
    # below threshold, retrieval likely missed the mark; try one more round
    # seeded by decomposed legal-aspect sub-queries and merge candidate pools
    # before a final rerank, instead of silently accepting a weak top-5.
    if (
        config.RETRIEVAL_EVALUATOR_ENABLED
        and (not reranked or reranked[0].score < config.RETRIEVAL_EVALUATOR_SCORE_THRESHOLD)
    ):
        sub_queries = decompose_query(query_text, n_subqueries=config.QUERY_DECOMPOSITION_MAX_SUBQUERIES)
        if sub_queries:
            extra_candidates: list[RetrievedChunk] = []
            for sub_q in sub_queries:
                extra_candidates.extend(hybrid_search(sub_q, top_k=config.RERANK_TOP_K, use_decomposition=False))

            merged = {c.chunk_id: c for c in candidates}
            for c in extra_candidates:
                merged.setdefault(c.chunk_id, c)
            merged_candidates = list(merged.values())
            child_text_by_id.update({c.chunk_id: c.text for c in extra_candidates})

            reranked = rerank(query_text, _enrich_with_parent_text(merged_candidates), top_k=config.FINAL_LAW_TOP_K)
            reranked = [
                c.model_copy(update={"text": child_text_by_id.get(c.chunk_id, c.text)}) for c in reranked
            ]

    return reranked


def process_case_with_debug(case: CaseQuery) -> tuple[SubmissionRecord, dict]:
    """Same work as `process_case()`, but also returns a debug dict carrying
    fields that don't belong on `SubmissionRecord` (IMPROVEMENT_PLAN.md
    §3.4): `is_fallback`, `fallback_reason`, `confidence`,
    `dropped_hallucinated_citations`. Intended for local evaluation harnesses
    (test/test_all_backend.py); `backend.submission` should keep using the
    plain `process_case()` below since only the strict submission schema
    matters there.
    """
    case_evidence_hits = collect_case_evidence(case)

    # legalrag_adjustments.md §5: only the top-N (by API relevance score) hits
    # go into the digest LLM call — the *full* retrieved set is still
    # reported in the submission below for Case-Recall scoring purposes,
    # this cap only bounds the digest step's token usage.
    top_hits_for_digest = sorted(case_evidence_hits, key=lambda h: h.score, reverse=True)
    top_hits_for_digest = top_hits_for_digest[: config.TOP_N_EVIDENCE_FOR_DIGEST]
    case_digest = build_case_digest(case.case_query, [h.text for h in top_hits_for_digest])

    law_chunks = collect_law_evidence(case.case_query)
    outcome = predict_outcome(case.case_query, law_chunks, case_digest)

    if outcome.dropped_hallucinated_citations:
        log.info(
            "case=%s: dropped %d citation(s) not present in retrieved law evidence",
            case.case_id, outcome.dropped_hallucinated_citations,
        )
    if outcome.is_fallback:
        log.warning("case=%s: forced fallback prediction (%s)", case.case_id, outcome.fallback_reason)

    record = SubmissionRecord(
        case_id=case.case_id,
        prediction=outcome.prediction,
        case_evidence=[h.chunk_id for h in case_evidence_hits],
        law_evidence=outcome.law_citations,
    )
    debug = {
        "is_fallback": outcome.is_fallback,
        "fallback_reason": outcome.fallback_reason,
        "confidence": outcome.confidence,
        "dropped_hallucinated_citations": outcome.dropped_hallucinated_citations,
    }
    return record, debug


def process_case(case: CaseQuery) -> SubmissionRecord:
    """Thin wrapper over `process_case_with_debug()` for callers that only
    need the `SubmissionRecord` (e.g. `backend.submission`)."""
    record, _debug = process_case_with_debug(case)
    return record
