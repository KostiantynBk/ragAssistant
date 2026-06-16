"""
Evaluation entry point for quick console results.

Run this after ingestion. It calls src.eval.harness over the Chroma collections
and prints retrieval metrics for each chunking/reranker configuration.

Usage:
    python scripts/evaluate.py [--qa-path eval_qa.json] [--db-path db/chroma]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.eval.harness import run_harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-path", default="eval_qa.json")
    ap.add_argument("--db-path", default="db/chroma")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--no-faithfulness", action="store_true", help="Skip LLM-as-judge to save API calls")
    args = ap.parse_args()

    configs = [
        ("recursive",    "langchain_docs_recursive", False),
        ("recursive+rr", "langchain_docs_recursive", True),
        ("header",       "langchain_docs_header",    False),
        ("header+rr",    "langchain_docs_header",    True),
    ]

    results = []
    for label, collection, use_rr in configs:
        print(f"Evaluating {label}…")
        m = run_harness(
            qa_path=args.qa_path,
            collection_name=collection,
            config_label=label,
            use_reranker=use_rr,
            db_path=args.db_path,
            embed_model=args.embed_model,
            fake_answers=args.no_faithfulness,
        )
        results.append(m)

    header = f"{'Config':<15} {'Hit@1':>6} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>6} {'Faith':>6} {'Lat(ms)':>8}"
    print("\n" + header)
    print("-" * len(header))
    for m in results:
        print(
            f"{m.config:<15} {m.hit_at_1:>6.2f} {m.hit_at_5:>6.2f} {m.hit_at_10:>7.2f} "
            f"{m.mrr:>6.3f} {m.faithfulness:>6.2f} {m.avg_latency_ms:>8.1f}"
        )


if __name__ == "__main__":
    main()
