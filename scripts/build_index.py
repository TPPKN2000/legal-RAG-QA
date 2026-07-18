"""
Ingestion CLI: parse the law corpus, chunk it hierarchically, then build/persist
the BM25 index and upsert child chunks into Pinecone.

Usage:
    python -m scripts.build_index
    python -m scripts.build_index --corpus data/corpus_law_pub.json --rebuild-pinecone
"""
from __future__ import annotations

import argparse
import logging

from backend import config
from backend.indexing.bm25_index import BM25Index
from backend.indexing.vector_store import delete_namespace, upsert_chunks
from backend.ingestion.chunker import chunk_articles
from backend.ingestion.metadata import build_metadata
from backend.ingestion.parser import load_law_corpus

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build BM25 + Pinecone indices from the law corpus.")
    parser.add_argument("--corpus", default=str(config.LAW_CORPUS_PATH))
    parser.add_argument("--rebuild-pinecone", action="store_true", help="Wipe the Pinecone namespace before upserting.")
    parser.add_argument("--skip-pinecone", action="store_true", help="Only (re)build the local BM25 index.")
    args = parser.parse_args()

    log.info("loading law corpus from %s", args.corpus)
    docs = load_law_corpus(args.corpus)
    log.info("loaded %d law documents", len(docs))

    status_by_law = {}
    all_chunks = []
    for doc in docs:
        meta = build_metadata(doc)
        status_by_law[doc.law_id] = meta.status
        all_chunks.extend(chunk_articles(doc.articles))

    n_parent = sum(1 for c in all_chunks if c.level == "parent")
    n_child = sum(1 for c in all_chunks if c.level == "child")
    log.info("chunked into %d parent + %d child chunks", n_parent, n_child)

    log.info("building BM25 index")
    bm25 = BM25Index()
    bm25.build(all_chunks, status_by_law=status_by_law)
    bm25.save()
    log.info("saved BM25 index to %s", config.BM25_INDEX_PATH)

    if not args.skip_pinecone:
        if args.rebuild_pinecone:
            log.info("wiping Pinecone namespace %s", config.PINECONE_NAMESPACE)
            delete_namespace()
        log.info("embedding + upserting child chunks to Pinecone")
        n = upsert_chunks(all_chunks, status_by_law=status_by_law)
        log.info("upserted %d vectors", n)


if __name__ == "__main__":
    main()
