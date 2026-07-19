import os
import argparse
import logging
import json
import hashlib
from backend.indexing.bm25_index import BM25Index
from backend.indexing import vector_store
from backend.models import LawChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--rebuild-pinecone", action="store_true")
    args = parser.parse_args()

    logger.info(f"loading law corpus from {args.corpus}")
    with open(args.corpus, 'r', encoding='utf-8') as f:
        corpus = json.load(f)

    logger.info(f"loaded {len(corpus)} law documents")

    law_chunks = []
    n_parent = 0
    n_child = 0

    for doc in corpus:
        law_id = doc.get('law_id', 'unknown')
        articles = doc.get('content', [])
        if isinstance(articles, list):
            for art in articles:
                text = art.get('content_Article', "")
                aid = art.get('aid', n_child)
                if text:
                    # Encode chunk_id to MD5 hex to ensure ASCII compatibility for Pinecone
                    raw_id = f"{law_id}_{aid}"
                    safe_id = hashlib.md5(raw_id.encode()).hexdigest()
                    
                    chunk = LawChunk(
                        chunk_id=safe_id, 
                        law_id=law_id,
                        aid=aid,
                        text=text,
                        level="child",
                        breadcrumb=""
                    )
                    law_chunks.append(chunk)
                    n_child += 1
            n_parent += 1

    logger.info(f"chunked into {n_parent} parent + {n_child} child chunks")

    if n_child == 0:
        logger.error("No chunks created.")
        return

    # 1. Build BM25 Index
    logger.info("Building BM25 index...")
    bm25 = BM25Index()
    bm25.build(law_chunks)

    # 2. Build Pinecone Index
    if args.rebuild_pinecone:
        logger.info("Rebuilding Pinecone index...")
        count = vector_store.upsert_chunks(law_chunks)
        logger.info(f"Successfully upserted {count} chunks to Pinecone.")

if __name__ == "__main__":
    main()
