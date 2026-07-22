"""
Client for the organizer-hosted Case Content API (docs/case_content_api_doc.md).

Key constraints encoded here:
  - 1 request / 5 seconds per team -> a blocking, process-wide rate limiter
    (not per-case) since the limit is per TEAM TOKEN, not per case.
  - Exactly one segment returned per call -> caller must issue multiple
    distinct queries to gather more evidence.
  - 429 -> wait and retry; 503 -> transient, retry with backoff.
  - Every call counts against the case's API-efficiency budget
    (docs/evaluation.md §2.4), so this client also tracks call counts per
    case_id for `pipeline.py`'s budget-aware evidence loop.

IMPROVEMENT_PLAN.md §3.2 (BUDGET INTEGRITY fix, applied here): the call
counter now increments exactly once per logical `retrieve()` invocation,
not once per HTTP attempt inside it — see `retrieve()`'s inline comment.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass

import requests

from backend import config
from backend.models import CaseEvidenceHit


class CaseAPIError(RuntimeError):
    pass


@dataclass
class _RateLimiter:
    min_interval_sec: float
    _lock: threading.Lock = None
    _last_call_ts: float = 0.0

    def __post_init__(self):
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call_ts
            remaining = self.min_interval_sec - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_call_ts = time.monotonic()


class CaseContentAPIClient:
    """Thin, rate-limited wrapper around POST /retrieve.

    One instance should be shared process-wide (the 5s limit is per team
    token, not per case), which is why `pipeline.py` imports the module-level
    singleton `client` below rather than constructing its own instance.
    """

    def __init__(
        self,
        base_url: str = config.ALQAC_API_BASE_URL,
        token: str = config.ALQAC_TOKEN,
        min_interval_sec: float = config.ALQAC_MIN_REQUEST_INTERVAL_SEC,
        max_retries: int = 3,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.max_retries = max_retries
        self.timeout = timeout
        self._limiter = _RateLimiter(min_interval_sec)
        self._calls_per_case: dict[str, int] = defaultdict(int)

    def calls_made(self, case_id: str) -> int:
        return self._calls_per_case[case_id]

    def retrieve(self, query: str, case_id: str) -> CaseEvidenceHit | None:
        """POST /retrieve -> the single top-ranked segment, or None if the
        API returned an empty result set (still counts as a call)."""
        if not self.token:
            raise CaseAPIError(
                "ALQAC_TOKEN is not set. Add it to your .env file (see README §Configuration)."
            )

        url = f"{self.base_url}/retrieve"
        headers = {"X-API-Key": self.token, "Content-Type": "application/json"}
        payload = {"query": query, "case_id": case_id}

        # IMPROVEMENT_PLAN.md §3.2: count exactly ONE call for this logical
        # query, regardless of how many transient-error retries (429 / 503 /
        # network hiccup) it takes underneath. Retries are not additional
        # queries the *caller* issued, and must not inflate the
        # API-efficiency budget c_i that E_i is computed from
        # (docs/evaluation.md §2.4) — previously this increment lived inside
        # the `for attempt` loop below and fired once per HTTP attempt, so a
        # single logical query could silently cost 2-3 units of budget
        # whenever the API returned 429/503 (observed directly in
        # test_all_backend.log: case_3241 reached api_calls=9 despite
        # DEFAULT_MAX_API_CALLS_PER_CASE=8).
        self._calls_per_case[case_id] += 1

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self._limiter.wait()
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_exc = e
                continue

            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if not results:
                    return None
                top = results[0]
                return CaseEvidenceHit(
                    chunk_id=top["chunk_id"], text=top["text"], score=float(top.get("score", 0.0))
                )
            if resp.status_code == 429:
                time.sleep(config.ALQAC_MIN_REQUEST_INTERVAL_SEC)
                continue
            if resp.status_code == 503:
                time.sleep(1.0 * (attempt + 1))
                continue
            if resp.status_code == 403:
                raise CaseAPIError("403 Forbidden — missing or invalid X-API-Key.")
            if resp.status_code == 422:
                raise CaseAPIError(f"422 Malformed request: query={query!r} case_id={case_id!r}")
            resp.raise_for_status()

        raise CaseAPIError(f"/retrieve failed after {self.max_retries} attempts: {last_exc}")


# Process-wide singleton — see class docstring for why this must be shared.
client = CaseContentAPIClient()
