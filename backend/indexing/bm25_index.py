"""
BM25 keyword index over law chunks (design doc §3.1).

BM25 exists alongside the Pinecone vector index specifically because exact
matches — article numbers, decree numbers like "145/2020/NĐ-CP", defined
legal terms — are things sparse keyword search nails and dense embeddings
sometimes blur. `rank_bm25` is enough for a corpus of this size; swap for
OpenSearch/Elasticsearch only if the corpus grows past what fits in memory.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path

from backend import config
from backend.models import LawChunk, RetrievedChunk

# Simple Vietnamese-aware tokenizer: lowercase, split on non-word chars but
# keep intra-word diacritics and alphanumeric decree numbers like 145/2020/NĐ-CP
# from being shattered.
_TOKEN_RE = re.compile(r"[^\W_]+(?:[/\-][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25IndexData:
    chunk_ids: list[str]
    law_ids: list[str]
    aids: list[int]
    texts: list[str]
    statuses: list[str]


class BM25Index:
    def __init__(self):
        self._bm25 = None
        self._data: BM25IndexData | None = None

    def build(self, chunks: list[LawChunk], status_by_law: dict[str, str] | None = None) -> None:
        from rank_bm25 import BM25Okapi

        status_by_law = status_by_law or {}
        child_chunks = [c for c in chunks if c.level == "child"]
        corpus_tokens = [tokenize(c.text) for c in child_chunks]

        self._bm25 = BM25Okapi(corpus_tokens)
        self._data = BM25IndexData(
            chunk_ids=[c.chunk_id for c in child_chunks],
            law_ids=[c.law_id for c in child_chunks],
            aids=[c.aid for c in child_chunks],
            texts=[c.text for c in child_chunks],
            statuses=[status_by_law.get(c.law_id, "unknown") for c in child_chunks],
        )

    def save(self, path: str | Path = config.BM25_INDEX_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "data": self._data}, f)

    def load(self, path: str | Path = config.BM25_INDEX_PATH) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._bm25 = payload["bm25"]
        self._data = payload["data"]

    def query(
        self,
        text: str,
        top_k: int = config.BM25_TOP_K,
        law_id: str | None = None,
        require_active: bool = True,
    ) -> list[RetrievedChunk]:
        if self._bm25 is None or self._data is None:
            raise RuntimeError("BM25Index not built/loaded. Call .build() or .load() first.")

        scores = self._bm25.get_scores(tokenize(text))
        candidates = list(enumerate(scores))

        # Hard metadata filter BEFORE truncating to top_k (design doc §3.2).
        def _passes(idx: int) -> bool:
            if law_id and self._data.law_ids[idx] != law_id:
                return False
            if require_active and self._data.statuses[idx] not in ("active", "unknown"):
                return False
            return True

        candidates = [(i, s) for i, s in candidates if _passes(i)]
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]

        return [
            RetrievedChunk(
                chunk_id=self._data.chunk_ids[i],
                law_id=self._data.law_ids[i],
                aid=self._data.aids[i],
                text=self._data.texts[i],
                score=float(s),
                source="bm25",
            )
            for i, s in candidates
            if s > 0
        ]


_singleton: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    """Lazily load the singleton BM25 index from disk."""
    global _singleton
    if _singleton is None:
        _singleton = BM25Index()
        _singleton.load()
    return _singleton
