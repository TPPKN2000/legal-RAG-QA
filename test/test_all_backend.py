"""
Test script for legal-RAG-QA (backend/ package) — run the full pipeline over
all cases in ALQAC2026_public_test.json and report OutcomeAccuracy, an
approximate Law F1, and Case Content API usage per case.

Adapted from the src.*-based test/test_all.py used in legalAI_test.ipynb, but
rewired for the backend.* pipeline (process_case / CaseQuery /
SubmissionRecord) instead of src.pipeline.predict_case.

Before running this, make sure the two bugs found in legalAI_test.ipynb are
fixed, or this script will exit early with a clear message instead of
silently producing 50x B_WIN fallback like the notebook run did:
  1. ALQAC_TOKEN must be a real, valid team token (env var or .env).
  2. scripts/build_index.py must call bm25.save() after bm25.build(), and
     you must have re-run it so data/bm25_index.pkl actually exists.

Usage:
    python -m test.test_all_backend                # all cases
    python -m test.test_all_backend -n 5            # smoke test first
    python -m test.test_all_backend -n 10 --seed 1
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

log_file_path = Path(__file__).parent / "test_all_backend.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file_path, encoding="utf-8")],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def parse_gold_law_provisions(related_law_text: str) -> list[dict]:
    """Parse the public test set's 'related_law_provisions' field into
    {law_name, article_num} dicts. The public gold set only gives law NAMES,
    not law_ids, so we can only match on article number (aid) below — same
    caveat that the original src/test_all.py had."""
    provisions = []
    if not related_law_text:
        return provisions
    for line in related_law_text.strip().split("\n"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        law_name, article_part = (p.strip() for p in line.split("|", 1))
        m = re.search(r"Điều\s+(\d+)", article_part)
        if m:
            provisions.append({"law_name": law_name, "article_num": int(m.group(1))})
    return provisions


def compute_law_f1(predicted: list[dict], gold_provisions: list[dict]) -> tuple[float, float, float]:
    """Approximate P/R/F1 matched on article number (aid) only, since
    the public test set doesn't expose law_ids for gold provisions.

    NOTE: this is a PER-CASE (macro-style) score — see main() for the
    micro-averaged aggregate that actually matches docs/evaluation.md §2.6.
    """
    pred_aids = {int(p["aid"]) for p in predicted}
    gold_aids = {g["article_num"] for g in gold_provisions}
    if not pred_aids and not gold_aids:
        return 1.0, 1.0, 1.0
    tp = len(pred_aids & gold_aids)
    precision = tp / len(pred_aids) if pred_aids else 0.0
    recall = tp / len(gold_aids) if gold_aids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Run backend.pipeline over ALQAC2026_public_test.json")
    parser.add_argument("-n", "--num-cases", type=int, default=None, help="Only test N cases (random sample)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Output submission path")
    parser.add_argument("--offline", action="store_true", help="Skip Case Content API entirely (no ALQAC_TOKEN needed); "
                          "only evaluates OutcomeAccuracy + Law F1.")
    args = parser.parse_args()

    from backend import config
    from backend.ingestion.parser import load_test_set
    from backend.models import CaseQuery, SubmissionRecord, generate_text
    from backend.pipeline import process_case
    from backend.case_api_client import client as case_api_client

    # --- pre-flight checks -------------------------------------------------
    # Fail fast and loudly instead of burning ~2h to produce 50x B_WIN
    # fallback, like the legalAI_test.ipynb run did.
    if args.offline:
        log.info("Offline mode: stubbing out Case Content API (no ALQAC_TOKEN required).")

        def _stub_retrieve(query: str, case_id: str):
            # Count the call so budget/logging logic in pipeline.py still works,
            # but never hit the network.
            case_api_client._calls_per_case[case_id] += 1
            return None

        case_api_client.retrieve = _stub_retrieve
    else:
        if not config.ALQAC_TOKEN:
            sys.exit(
                "ALQAC_TOKEN is not set (backend.config.ALQAC_TOKEN is empty).\n"
                "Set it in .env or `export ALQAC_TOKEN=alqac_...` before running."
            )

    from backend.pipeline import process_case  # import after stubbing

    if not config.BM25_INDEX_PATH.exists():
        sys.exit(
            f"{config.BM25_INDEX_PATH} not found.\n"
            "Re-run scripts/build_index.py AFTER adding `bm25.save()` right "
            "after `bm25.build(law_chunks)` — the current script builds the "
            "BM25 index in memory but never persists it to disk."
        )

    # Warm up the generation model ONCE, before the loop. In the notebook
    # run, every single case re-triggered HEAD requests to HF + a ~60s
    # "Loading weights" step even though _get_generation_model() is
    # lru_cache'd — warming up here avoids paying that cost 50 times.
    log.info("Warming up generation model (%s)...", config.GENERATION_MODEL_NAME)
    t0 = time.time()
    generate_text(system_prompt="ping", user_prompt="ping", max_new_tokens=4, temperature=0.0)
    log.info("Model warm-up done in %.1fs", time.time() - t0)

    # --- load test set + gold ----------------------------------------------
    test_path = config.TEST_SET_PATH
    raw_cases = load_test_set(test_path)
    with open(test_path, "r", encoding="utf-8") as f:
        gold_raw = json.load(f)
    gold_by_id = {c["case_id"]: c for c in gold_raw}

    random.seed(args.seed)
    shuffled = raw_cases.copy()
    random.shuffle(shuffled)
    batch = shuffled[: args.num_cases] if args.num_cases else shuffled

    log.info("Loaded %d cases, testing %d (seed=%d)", len(raw_cases), len(batch), args.seed)

    out_path = Path(args.out) if args.out else Path(project_root) / "test" / "test_submission_backend.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submissions: list[dict] = []
    results: list[dict] = []

    # --- main loop -----------------------------------------------------------
    for i, raw_case in enumerate(batch, start=1):
        case_id = raw_case["case_id"]
        gold = gold_by_id[case_id]
        gold_label = gold["verdict_label"]
        gold_provisions = parse_gold_law_provisions(gold.get("related_law_provisions", ""))

        case = CaseQuery(
            case_id=case_id,
            case_query=raw_case.get("case_query") or raw_case.get("query") or "",
            n_segments=raw_case.get("n_segments") or raw_case.get("n_i"),
        )

        log.info("--- [%d/%d] %s ---", i, len(batch), case_id)
        t0 = time.time()
        try:
            record = process_case(case)
        except Exception as e:
            log.error("case %s crashed, emitting conservative fallback: %s", case_id, e)
            record = SubmissionRecord(case_id=case_id, prediction="B_WIN")
        duration = time.time() - t0

        record_dict = json.loads(record.model_dump_json())
        submissions.append(record_dict)
        # Save progressively so a crash mid-run doesn't lose earlier results.
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(submissions, f, ensure_ascii=False, indent=2)

        outcome_correct = int(record.prediction == gold_label)
        precision, recall, f1 = compute_law_f1(record_dict["law_evidence"], gold_provisions)
        api_calls = case_api_client.calls_made(case_id)

        results.append(
            {
                "case_id": case_id,
                "prediction": record.prediction,
                "outcome_correct": outcome_correct,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "api_calls": api_calls,
                "duration": duration,
                "pred_law_evidence": record_dict["law_evidence"],
                "gold_law_provisions": gold_provisions,
                "n_segments": case.n_segments,
            }
        )
        log.info(
            "-> pred=%s gold=%s correct=%s | law F1(approx)=%.3f | api_calls=%d | %.1fs",
            record.prediction, gold_label, bool(outcome_correct), f1, api_calls, duration,
        )

    # --- summary --------------------------------------------------------------
    n = len(results)
    acc = sum(r["outcome_correct"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    total_calls = sum(r["api_calls"] for r in results)

    log.info("=" * 60)
    log.info("BATCH SUMMARY (%d cases)", n)
    log.info("=" * 60)
    log.info("OutcomeAccuracy:        %.3f (%d/%d)", acc, sum(r["outcome_correct"] for r in results), n)
    log.info("Avg Law F1 (macro, approx*):   %.3f", avg_f1)
    log.info("Total API calls:        %d (avg %.1f/case)", total_calls, total_calls / n)
    log.info("Avg time/case:          %.1fs", sum(r["duration"] for r in results) / n)
    log.info("Prediction distribution: %s", Counter(r["prediction"] for r in results))
    log.info("Gold distribution:       %s", Counter(gold_by_id[r["case_id"]]["verdict_label"] for r in results))

    # legalrag_adjustments.md §4: docs/evaluation.md §2.6 defines Law F1 as a
    # MICRO average — pool TP/FP/FN across the whole test set first, THEN
    # divide — not the per-case macro average computed above. The two can
    # diverge a lot when gold-provision counts are uneven across cases, so
    # both are reported: macro above (kept for comparability with earlier
    # runs) and the evaluation-formula-accurate micro F1 below.
    pred_aid_counter: Counter[int] = Counter()
    gold_aid_counter: Counter[int] = Counter()
    for r in results:
        pred_aid_counter.update(int(p["aid"]) for p in r["pred_law_evidence"])
        gold_aid_counter.update(g["article_num"] for g in r["gold_law_provisions"])

    tp = sum(min(pred_aid_counter[a], gold_aid_counter[a]) for a in pred_aid_counter)
    n_pred = sum(pred_aid_counter.values())
    n_gold = sum(gold_aid_counter.values())
    micro_precision = tp / n_pred if n_pred else 0.0
    micro_recall = tp / n_gold if n_gold else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )
    log.info(
        "Micro Law F1 (matches evaluation.md §2.6 formula): P=%.3f R=%.3f F1=%.3f",
        micro_precision, micro_recall, micro_f1,
    )
    log.info(
        "* Both Law F1 numbers are matched on article number (aid) only — the public "
        "gold set gives law NAMES, not law_ids, so these are upper-bound approximations, "
        "not the organizers' real scoring (which also matches on law_id)."
    )

    # legalrag_adjustments.md §4: rough local estimate of the API-efficiency
    # factor E_i (docs/evaluation.md §2.4). The public test set's n_i is
    # usually unknown at real scoring time (§0/§1), so this uses each case's
    # own n_segments when present and otherwise the same
    # DEFAULT_MAX_API_CALLS_PER_CASE fallback pipeline.py itself falls back
    # to, purely so the "effective budget" being measured against matches
    # what the pipeline actually used.
    def _e_i(api_calls: int, budget_n: int) -> float:
        b_i = 2 * budget_n
        ceiling = 5 * budget_n
        if api_calls <= b_i:
            return 1.0
        if api_calls >= ceiling:
            return 0.0
        return 1 - (api_calls - b_i) / (3 * budget_n)

    e_i_values = [
        _e_i(r["api_calls"], r["n_segments"] or config.DEFAULT_MAX_API_CALLS_PER_CASE)
        for r in results
    ]
    avg_e_i = sum(e_i_values) / len(e_i_values)
    log.info(
        "Estimated avg API efficiency E_i: %.3f (n_i unknown for cases without n_segments; "
        "falls back to DEFAULT_MAX_API_CALLS_PER_CASE=%d as the assumed budget for those, "
        "matching pipeline.py's own fallback — NOT the organizers' real n_i)",
        avg_e_i, config.DEFAULT_MAX_API_CALLS_PER_CASE,
    )
    log.info(
        "Approx score (Case Recall excluded — public set has no gold case_evidence): "
        "0.70*%.3f + 0.10*%.3f(micro) = %.3f",
        acc, micro_f1, 0.70 * acc + 0.10 * micro_f1,
    )
    log.info("Submission saved to: %s", out_path)


if __name__ == "__main__":
    main()
