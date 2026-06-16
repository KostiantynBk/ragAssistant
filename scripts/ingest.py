"""
Ingestion entry point.

Run this before the API. It loads Markdown docs from data/docs, sends them to
src.ingest.loader, src.ingest.chunker, and src.ingest.embedder, then writes the
finished vector collections into db/chroma for the query phase.

Usage:
    python scripts/ingest.py [--docs-dir data/docs] [--db-path db/chroma]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.ingest.chunker import header_aware_chunks, recursive_chunks
from src.ingest.embedder import upsert_chunks
from src.ingest.loader import load_docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", default="data/docs")
    ap.add_argument("--db-path", default="db/chroma")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    args = ap.parse_args()

    docs = load_docs(args.docs_dir)
    if not docs:
        print(f"No .md files found in {args.docs_dir}. Clone or download the LangChain docs there first.")
        print("  git clone --depth 1 https://github.com/langchain-ai/langchain.git tmp_lc")
        print("  xcopy /E /I tmp_lc\\docs\\docs data\\docs")
        sys.exit(1)

    print(f"Loaded {len(docs)} documents.")

    rec_chunks, hdr_chunks = [], []
    for doc in docs:
        rec_chunks.extend(recursive_chunks(doc["text"], doc["path"]))
        hdr_chunks.extend(header_aware_chunks(doc["text"], doc["path"]))

    print(f"Recursive chunks: {len(rec_chunks)}")
    print(f"Header-aware chunks: {len(hdr_chunks)}")

    print("Upserting recursive chunks…")
    n = upsert_chunks(rec_chunks, "langchain_docs_recursive", db_path=args.db_path, embed_model=args.embed_model)
    print(f"  Upserted {n} chunks.")

    print("Upserting header-aware chunks…")
    n = upsert_chunks(hdr_chunks, "langchain_docs_header", db_path=args.db_path, embed_model=args.embed_model)
    print(f"  Upserted {n} chunks.")

    print("Done.")


if __name__ == "__main__":
    main()
