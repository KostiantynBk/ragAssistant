import json
import os
import time

from openai import OpenAI

from src.api.schemas import RAGAnswer, Source
from src.eval.judge import score_faithfulness
from src.eval.logger import log_query
from src.generation.prompts import SYSTEM_PROMPT

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _build_context(sources: list[Source]) -> str:
    parts = []
    for s in sources:
        tag = f"[{s.doc_id}#{s.chunk_index}]"
        parts.append(f"{tag}\n{s.text}")
    return "\n\n---\n\n".join(parts)


def generate(
    question: str,
    sources: list[Source],
    chunking: str = "recursive",
    use_reranker: bool = True,
    model: str | None = None,
) -> RAGAnswer:
    chat_model = model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    context = _build_context(sources)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    t0 = time.perf_counter()
    resp = _get_client().chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    answer_text = resp.choices[0].message.content or ""

    grounded, _ = score_faithfulness(question, answer_text, sources)

    log_query(
        question=question,
        chunking=chunking,
        use_reranker=use_reranker,
        answer=answer_text,
        grounded=grounded,
        latency_ms=latency_ms,
    )

    return RAGAnswer(
        answer=answer_text,
        sources=sources,
        grounded=grounded,
        latency_ms=round(latency_ms, 1),
    )
