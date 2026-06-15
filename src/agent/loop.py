"""OpenAI tool-calling agent loop. The LLM can call retrieve_docs as a tool."""

import json
import os
import time

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.api.schemas import RAGAnswer, Source
from src.eval.judge import score_faithfulness
from src.generation.prompts import SYSTEM_PROMPT
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import rerank

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_docs",
            "description": "Search the documentation corpus and return relevant text chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    }
]

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _call_retrieve(
    query: str,
    top_k: int,
    collection_name: str,
    use_reranker: bool,
    db_path: str,
    embed_model: str,
) -> list[Source]:
    sources = retrieve(query, collection_name, top_k=top_k, db_path=db_path, embed_model=embed_model)
    if use_reranker:
        sources = rerank(query, sources, keep_top_k=5)
    return sources


def agent_answer(
    question: str,
    collection_name: str = "langchain_docs_recursive",
    use_reranker: bool = True,
    db_path: str = "db/chroma",
    embed_model: str = "text-embedding-3-small",
    model: str | None = None,
    max_turns: int = 5,
) -> RAGAnswer:
    chat_model = model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    all_sources: list[Source] = []
    t0 = time.perf_counter()

    for _ in range(max_turns):
        resp = _get_client().chat.completions.create(
            model=chat_model,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message
        messages.append(msg)  # type: ignore[arg-type]

        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                sources = _call_retrieve(
                    query=args["query"],
                    top_k=args.get("top_k", 10),
                    collection_name=collection_name,
                    use_reranker=use_reranker,
                    db_path=db_path,
                    embed_model=embed_model,
                )
                all_sources.extend(sources)
                tool_result = json.dumps(
                    [{"doc_id": s.doc_id, "chunk_index": s.chunk_index, "text": s.text} for s in sources]
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
        else:
            answer_text = msg.content or ""
            latency_ms = (time.perf_counter() - t0) * 1000
            unique_sources = list({(s.doc_id, s.chunk_index): s for s in all_sources}.values())
            grounded, _ = score_faithfulness(question, answer_text, unique_sources)
            return RAGAnswer(
                answer=answer_text,
                sources=unique_sources,
                grounded=grounded,
                latency_ms=round(latency_ms, 1),
            )

    latency_ms = (time.perf_counter() - t0) * 1000
    return RAGAnswer(
        answer="Agent loop exceeded max turns without a final answer.",
        sources=all_sources,
        grounded=False,
        latency_ms=round(latency_ms, 1),
    )
