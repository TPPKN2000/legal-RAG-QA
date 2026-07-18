"""
End-to-end orchestration for a single test case (design doc pipeline.py role).

process_case() ties together:
  1. Budget-aware Case Content API evidence collection (docs/evaluation.md §2.4 —
     no penalty up to 2*n_i calls, zero credit at 5*n_i).
  2. Law-corpus retrieval: hybrid_search (BM25 + Pinecone + RRF) -> rerank.
  3. Outcome generation with grounding verification.
  4. Assembly into a `SubmissionRecord` matching docs/submission_example.json.

`submission.py` is the CLI that loops this over the whole test set.
"""
from __future__ import annotations

import logging

from backend import config
from backend.case_api_client import CaseAPIError, client as case_api_client
from backend.generation.generate import predict_outcome
from backend.retrieval.hybrid_search import hybrid_search
from backend.retrieval.querry_transform import rewrite_query
from backend.retrieval.rerank import rerank
from backend.models import CaseEvidenceHit, CaseQuery, SubmissionRecord

log = logging.getLogger(__name__)


def _case_api_budget(case: CaseQuery) -> int:
    """No-penalty ceiling for this case (docs/evaluation.md §2.4: B_i = 2*n_i).
    Falls back to a fixed cap when n_i isn't known ahead of time."""
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


def collect_law_evidence(query_text: str):
    """Hybrid search (design doc §3) -> cross-encoder rerank (design doc §4.1)."""
    candidates = hybrid_search(query_text, top_k=config.RERANK_TOP_K)
    return rerank(query_text, candidates, top_k=config.FINAL_LAW_TOP_K)


def process_case(case: CaseQuery) -> SubmissionRecord:
    case_evidence_hits = collect_case_evidence(case)
    law_chunks = collect_law_evidence(case.case_query)
    outcome = predict_outcome(case.case_query, law_chunks, case_evidence_hits)

    if outcome.dropped_hallucinated_citations:
        log.info(
            "case=%s: dropped %d citation(s) not present in retrieved law evidence",
            case.case_id, outcome.dropped_hallucinated_citations,
        )

    return SubmissionRecord(
        case_id=case.case_id,
        prediction=outcome.prediction,
        case_evidence=[h.chunk_id for h in case_evidence_hits],
        law_evidence=outcome.law_citations,
    )
