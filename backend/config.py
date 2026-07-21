"""
Central configuration for the LegalRAG system.
All values are overridable via environment variables (.env).
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be set externally


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None else default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None else default


# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(_env("LEGALRAG_DATA_DIR", str(ROOT_DIR / "data")))
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
LAW_CORPUS_PATH = Path(_env("LAW_CORPUS_PATH", str(DATA_DIR / "corpus_law_pub.json")))
TEST_SET_PATH = Path(_env("TEST_SET_PATH", str(DATA_DIR / "ALQAC2026_public_test.json")))
SUBMISSION_OUT_PATH = Path(_env("SUBMISSION_OUT_PATH", str(ROOT_DIR / "submission.json")))

# --- Case Content API ----------------------------------------------------
ALQAC_API_BASE_URL = _env("ALQAC_API_BASE_URL", "https://alqac-api.ngrok.pro")
ALQAC_TOKEN = _env("ALQAC_TOKEN", "")
ALQAC_MIN_REQUEST_INTERVAL_SEC = _env_float("ALQAC_MIN_REQUEST_INTERVAL_SEC", 5.0)
# Budget policy from docs/evaluation.md: no penalty up to 2*n_i calls, zero recall credit at 5*n_i.
# We target a soft ceiling below 2*n_i per case to leave margin for retries.
API_BUDGET_MULTIPLIER = _env_float("API_BUDGET_MULTIPLIER", 2.0)
API_HARD_CEILING_MULTIPLIER = _env_float("API_HARD_CEILING_MULTIPLIER", 5.0)
# If the case's segment count n_i is unknown ahead of time, fall back to a fixed cap.
DEFAULT_MAX_API_CALLS_PER_CASE = _env_int("DEFAULT_MAX_API_CALLS_PER_CASE", 8)

# --- Pinecone (vector store) -------------------------------------------
PINECONE_API_KEY = _env("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = _env("INDEX_NAME", "legalrag-law-corpus")
PINECONE_CLOUD = _env("PINECONE_CLOUD", "aws")
PINECONE_REGION = _env("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = _env("PINECONE_NAMESPACE", "law-corpus")

# --- Models --------------------------------------------------------------
EMBEDDING_MODEL_NAME = _env("EMBEDDING_MODEL_NAME", "AITeamVN/Vietnamese_Embedding")
EMBEDDING_DIM = _env_int("EMBEDDING_DIM", 1024)
RERANKER_MODEL_NAME = _env("RERANKER_MODEL_NAME", "AITeamVN/Vietnamese_Reranker")
# legalrag_adjustments.md §2: Qwen3-8B (fp16, ~16GB weights alone) OOMs on a
# 15GB T4. Qwen3.5-0.8B is the confirmed replacement — verify the exact HF
# repo id (with or without an "-Instruct" suffix) before a real run; see the
# chat_template guard in models.py._get_generation_model for a loud failure
# if this accidentally points at a non-chat base checkpoint.
GENERATION_MODEL_NAME = _env("GENERATION_MODEL_NAME", "Qwen/Qwen3.5-0.8B")
DEVICE = _env("LEGALRAG_DEVICE", "cuda")  # falls back to cpu automatically in model.py
# Qwen3-family chat templates default reasoning ("<think>...</think>") on;
# left un-disabled this can silently eat the whole max_new_tokens budget
# before any JSON is emitted, which _extract_json then fails to parse and
# generate.py falls back to B_WIN — see legalrag_adjustments.md §2 point 3.
GENERATION_ENABLE_THINKING = _env("GENERATION_ENABLE_THINKING", "false").lower() == "true"
# sdpa is built into torch (no extra install); avoids an import-time crash if
# flash-attn isn't present but transformers tries to auto-select it.
GENERATION_ATTN_IMPL = _env("GENERATION_ATTN_IMPL", "sdpa")
GENERATION_MAX_NEW_TOKENS_DEFAULT = _env_int("GENERATION_MAX_NEW_TOKENS_DEFAULT", 500)

# --- Retrieval parameters -------------------------------------------------
BM25_TOP_K = _env_int("BM25_TOP_K", 30)
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", 30)
RRF_K = _env_int("RRF_K", 60)  # standard RRF damping constant
RERANK_TOP_K = _env_int("RERANK_TOP_K", 20)
FINAL_LAW_TOP_K = _env_int("FINAL_LAW_TOP_K", 5)

# --- Query transformation (legalrag_adjustments.md §3) ---------------------
# HyDE was removed: it asked the LLM to draft a hypothetical *statute-styled*
# passage and embedded that for search, which risks feeding a smaller model's
# hallucinated "legal-sounding" text straight into retrieval. Replaced with
# NER-grounded masking (real entity tags, not generated text) + a
# non-generative aspect decomposition, fused with weighted RRF per Judge-R1.
NER_MODEL_NAME = _env("NER_MODEL_NAME", "NlpHUST/ner-vietnamese-electra-base")
QUERY_DECOMPOSITION_ENABLED = _env("QUERY_DECOMPOSITION_ENABLED", "true").lower() == "true"
QUERY_DECOMPOSITION_MAX_SUBQUERIES = _env_int("QUERY_DECOMPOSITION_MAX_SUBQUERIES", 4)
# Judge-R1 uses w_agent=2.0 / w_std=1.0 when fusing the raw-query route with
# the decomposed/planned-query route (agentic route had much higher P@5).
RRF_WEIGHT_STANDARD = _env_float("RRF_WEIGHT_STANDARD", 1.0)
RRF_WEIGHT_AGENT = _env_float("RRF_WEIGHT_AGENT", 2.0)

# --- Retrieval evaluator loop (legalrag_adjustments.md §7) ------------------
# Reuses the existing cross-encoder rerank score as a cheap "is this good
# enough" gate (ViDRILL-style) instead of adding a separate LLM judge. If the
# best reranked score is below threshold, one extra decomposition-based
# retrieval round is triggered before falling back to what was found.
RETRIEVAL_EVALUATOR_ENABLED = _env("RETRIEVAL_EVALUATOR_ENABLED", "true").lower() == "true"
RETRIEVAL_EVALUATOR_SCORE_THRESHOLD = _env_float("RETRIEVAL_EVALUATOR_SCORE_THRESHOLD", 0.75)

# --- Chunking --------------------------------------------------------------
CHILD_CHUNK_MAX_TOKENS = _env_int("CHILD_CHUNK_MAX_TOKENS", 220)
PARENT_CHUNK_MAX_TOKENS = _env_int("PARENT_CHUNK_MAX_TOKENS", 1000)
# legalrag_adjustments.md §6b: soft-split an oversized Khoản/Điểm on sentence
# boundaries (never hard token cuts) once it exceeds this many characters.
# ~900 chars ≈ ViDRILL's 450-word threshold for Vietnamese legal text.
CHILD_MAX_CHARS = _env_int("CHILD_MAX_CHARS", 900)
# legalrag_adjustments.md §6a: parent (whole-Điều) chunks are persisted here
# by scripts/build_index.py and re-attached at rerank time only (never sent
# to the final generation prompt, to keep token budget in check).
PARENT_LOOKUP_PATH = DATA_DIR / "parent_lookup.pkl"

# --- Prompt compression -----------------------------------------------
COMPRESSION_ENABLED = _env("COMPRESSION_ENABLED", "true").lower() == "true"
COMPRESSION_TARGET_RATIO = _env_float("COMPRESSION_TARGET_RATIO", 0.5)

# --- Generation prompt splitting (legalrag_adjustments.md §5) --------------
# The ~10k-token final prompt traced back to stuffing full case-evidence +
# full verbatim law text into ONE generation call. Split into a small
# "case-fact digest" call (auxiliary text only, freely paraphrasable) and a
# leaner "verdict" call that only carries the digest + verbatim law text.
TOP_N_EVIDENCE_FOR_DIGEST = _env_int("TOP_N_EVIDENCE_FOR_DIGEST", 5)
CASE_DIGEST_MAX_NEW_TOKENS = _env_int("CASE_DIGEST_MAX_NEW_TOKENS", 220)

# --- Prediction labels (docs/test_design.md) ------------------------------
VALID_PREDICTIONS = ("A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN")
