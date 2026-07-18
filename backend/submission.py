"""
CLI entry point: run `process_case` over the whole public/private test set and
write `submission.json` in the exact shape required by
docs/test_design.md / docs/submission_example.json.

Usage:
    python -m backend.submission
    python -m backend.submission --test-set path/to/set.json --out submission.json --limit 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from backend import config
from backend.ingestion.parser import load_test_set
from backend.models import CaseQuery, SubmissionRecord
from backend.pipeline import process_case

log = logging.getLogger(__name__)


def _validate_submission(records: list[SubmissionRecord], test_case_ids: set[str]) -> None:
    """Mirror the organizer's validation rules from docs/test_design.md so
    format violations are caught locally before submission, not after."""
    seen_ids = set()
    for r in records:
        if r.case_id not in test_case_ids:
            raise ValueError(f"case_id {r.case_id!r} is not in the official test set")
        if r.case_id in seen_ids:
            raise ValueError(f"duplicate case_id in submission: {r.case_id!r}")
        seen_ids.add(r.case_id)
        if r.prediction not in config.VALID_PREDICTIONS:
            raise ValueError(f"invalid prediction {r.prediction!r} for case_id {r.case_id!r}")

    missing = test_case_ids - seen_ids
    if missing:
        raise ValueError(f"{len(missing)} test case(s) missing a prediction, e.g. {sorted(missing)[:5]}")


def run(test_set_path=None, out_path=None, limit: int | None = None) -> list[SubmissionRecord]:
    test_set_path = test_set_path or config.TEST_SET_PATH
    out_path = out_path or config.SUBMISSION_OUT_PATH

    raw_cases = load_test_set(test_set_path)
    if limit:
        raw_cases = raw_cases[:limit]

    cases = [
        CaseQuery(
            case_id=c["case_id"],
            case_query=c.get("case_query") or c.get("query") or "",
            n_segments=c.get("n_segments") or c.get("n_i"),
        )
        for c in raw_cases
    ]

    records: list[SubmissionRecord] = []
    for i, case in enumerate(cases, start=1):
        log.info("[%d/%d] processing %s", i, len(cases), case.case_id)
        try:
            records.append(process_case(case))
        except Exception as e:
            log.error("case %s failed, emitting conservative fallback: %s", case.case_id, e)
            # Every test case must have exactly one prediction (docs/test_design.md);
            # a hard failure must still emit *something* rather than drop the case.
            records.append(SubmissionRecord(case_id=case.case_id, prediction="B_WIN"))

    all_case_ids = {c.case_id for c in cases}
    _validate_submission(records, all_case_ids)

    payload = [json.loads(r.model_dump_json()) for r in records]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    log.info("wrote %d records to %s", len(records), out_path)
    return records


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the LegalRAG prediction pipeline over the test set.")
    parser.add_argument("--test-set", default=None, help="Path to test set JSON (default: config.TEST_SET_PATH)")
    parser.add_argument("--out", default=None, help="Output submission path (default: config.SUBMISSION_OUT_PATH)")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N cases (debugging)")
    args = parser.parse_args()

    try:
        run(test_set_path=args.test_set, out_path=args.out, limit=args.limit)
    except Exception:
        log.exception("submission run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
