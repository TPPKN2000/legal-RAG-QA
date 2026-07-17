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
GENERATION_MODEL_NAME = _env("GENERATION_MODEL_NAME", "Qwen/Qwen3-8B")
HYDE_MODEL_NAME = _env("HYDE_MODEL_NAME", GENERATION_MODEL_NAME)
DEVICE = _env("LEGALRAG_DEVICE", "cuda")  # falls back to cpu automatically in model.py

# --- Retrieval parameters -------------------------------------------------
BM25_TOP_K = _env_int("BM25_TOP_K", 30)
VECTOR_TOP_K = _env_int("VECTOR_TOP_K", 30)
RRF_K = _env_int("RRF_K", 60)  # standard RRF damping constant
RERANK_TOP_K = _env_int("RERANK_TOP_K", 20)
FINAL_LAW_TOP_K = _env_int("FINAL_LAW_TOP_K", 5)

# --- Chunking --------------------------------------------------------------
CHILD_CHUNK_MAX_TOKENS = _env_int("CHILD_CHUNK_MAX_TOKENS", 220)
PARENT_CHUNK_MAX_TOKENS = _env_int("PARENT_CHUNK_MAX_TOKENS", 1000)

# --- Prompt compression -----------------------------------------------
COMPRESSION_ENABLED = _env("COMPRESSION_ENABLED", "true").lower() == "true"
COMPRESSION_TARGET_RATIO = _env_float("COMPRESSION_TARGET_RATIO", 0.5)

# --- Prediction labels (docs/test_design.md) ------------------------------
VALID_PREDICTIONS = ("A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN")
