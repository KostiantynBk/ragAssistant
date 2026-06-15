"""
Evaluation harness: hit@k and MRR over a QA set.

QA format (eval_qa.json):
[
  {
    "question": "...",
    "ground_truth_doc_id": "docs/agents.md",
    "acceptable_answers": ["..."]
  },
  ...
]
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.api.schemas import Source
from src.eval.judge import score_faithfulness
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import rerank


@dataclass
class EvalMetrics:
    config: str
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    mrr: float = 0.0
    faithfulness: float = 0.0
    avg_latency_ms: float = 0.0
    n: int = 0


def _reciprocal_rank(sources: list[Source], ground_truth_doc_id: str) -> float:
    for rank, s in enumerate(sources, start=1):
        if s.doc_id == ground_truth_doc_id:
            return 1.0 / rank
    return 0.0


def run_harness(
    qa_path: str | Path,
    collection_name: str,
    config_label: str,
    use_reranker: bool = False,
    retrieval_top_k: int = 20,
    rerank_top_k: int = 5,
    db_path: str = "db/chroma",
    embed_model: str = "text-embedding-3-small",
    fake_answers: bool = False,
) -> EvalMetrics:
    """Run retrieval-only eval (no generation) over QA pairs."""
    qa_items = json.loads(Path(qa_path).read_text(encoding="utf-8"))
    metrics = EvalMetrics(config=config_label, n=len(qa_items))

    h1 = h5 = h10 = rr_sum = faith_sum = lat_sum = 0.0

    for item in qa_items:
        q = item["question"]
        gt = item["ground_truth_doc_id"]

        t0 = time.perf_counter()
        sources = retrieve(q, collection_name, top_k=retrieval_top_k, db_path=db_path, embed_model=embed_model)
        if use_reranker:
            sources = rerank(q, sources, keep_top_k=rerank_top_k)
        lat_ms = (time.perf_counter() - t0) * 1000

        doc_ids = [s.doc_id for s in sources]
        h1 += int(gt in doc_ids[:1])
        h5 += int(gt in doc_ids[:5])
        h10 += int(gt in doc_ids[:10])
        rr_sum += _reciprocal_rank(sources, gt)
        lat_sum += lat_ms

        if not fake_answers:
            _, faith = score_faithfulness(q, item.get("acceptable_answers", [""])[0], sources)
            faith_sum += faith

    n = len(qa_items)
    metrics.hit_at_1 = h1 / n
    metrics.hit_at_5 = h5 / n
    metrics.hit_at_10 = h10 / n
    metrics.mrr = rr_sum / n
    metrics.faithfulness = faith_sum / n if not fake_answers else 0.0
    metrics.avg_latency_ms = lat_sum / n
    return metrics
