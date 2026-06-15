import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

load_dotenv()

from src.api.schemas import EvalResult, QueryRequest, RAGAnswer
from src.eval.logger import get_latest_evals
from src.generation.generator import generate
from src.retrieval.retriever import retrieve
from src.retrieval.reranker import rerank

app = FastAPI(
    title="DocsRAG",
    description="Retrieval-Augmented Documentation Assistant",
    version="0.1.0",
)

_COLLECTION = {
    "recursive": os.getenv("COLLECTION_RECURSIVE", "langchain_docs_recursive"),
    "header_aware": os.getenv("COLLECTION_HEADER", "langchain_docs_header"),
}
_DB_PATH = os.getenv("CHROMA_DB_PATH", "db/chroma")
_EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=RAGAnswer)
def ask(req: QueryRequest) -> RAGAnswer:
    collection = _COLLECTION.get(req.chunking)
    if not collection:
        raise HTTPException(status_code=400, detail=f"Unknown chunking: {req.chunking}")

    try:
        sources = retrieve(
            req.question,
            collection,
            top_k=req.top_k * 4 if req.use_reranker else req.top_k,
            db_path=_DB_PATH,
            embed_model=_EMBED_MODEL,
        )
        if req.use_reranker:
            sources = rerank(req.question, sources, keep_top_k=req.top_k)
        else:
            sources = sources[: req.top_k]

        return generate(
            question=req.question,
            sources=sources,
            chunking=req.chunking,
            use_reranker=req.use_reranker,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/eval/latest", response_model=list[EvalResult])
def eval_latest(n: int = Query(default=10, ge=1, le=100)):
    return get_latest_evals(n)
