"""
Benchmark all 4 configs and write results to SQLite + print a markdown table for the README.

Usage:
    python scripts/compare_configs.py [--qa-path eval_qa.json]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

from src.eval.harness import run_harness
from src.eval.logger import log_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-path", default="eval_qa.json")
    ap.add_argument("--db-path", default="db/chroma")
    ap.add_argument("--sqlite-path", default="db/eval.db")
    ap.add_argument("--embed-model", default="text-embedding-3-small")
    ap.add_argument("--no-faithfulness", action="store_true")
    args = ap.parse_args()

    configs = [
        ("recursive",    "langchain_docs_recursive", False),
        ("recursive+rr", "langchain_docs_recursive", True),
        ("header",       "langchain_docs_header",    False),
        ("header+rr",    "langchain_docs_header",    True),
    ]

    print("Running benchmarks…\n")
    rows = []
    for label, collection, use_rr in configs:
        print(f"  {label}…", end=" ", flush=True)
        m = run_harness(
            qa_path=args.qa_path,
            collection_name=collection,
            config_label=label,
            use_reranker=use_rr,
            db_path=args.db_path,
            embed_model=args.embed_model,
            fake_answers=args.no_faithfulness,
        )
        log_eval(
            config=m.config,
            hit_at_1=bool(m.hit_at_1),
            hit_at_5=bool(m.hit_at_5),
            hit_at_10=bool(m.hit_at_10),
            mrr=m.mrr,
            faithfulness=m.faithfulness,
            avg_latency_ms=m.avg_latency_ms,
            db_path=args.sqlite_path,
        )
        rows.append(m)
        print("done")

    print("\n## Evaluation Results\n")
    print("| Config | Hit@1 | Hit@5 | Hit@10 | MRR | Faithfulness | Avg Latency (ms) |")
    print("|--------|------:|------:|-------:|----:|-------------:|-----------------:|")
    for m in rows:
        print(
            f"| {m.config} | {m.hit_at_1:.2f} | {m.hit_at_5:.2f} | {m.hit_at_10:.2f} "
            f"| {m.mrr:.3f} | {m.faithfulness:.2f} | {m.avg_latency_ms:.1f} |"
        )


if __name__ == "__main__":
    main()
