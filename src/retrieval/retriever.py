import os

import chromadb
from openai import OpenAI

from src.api.schemas import Source

_chroma: chromadb.PersistentClient | None = None
_openai: OpenAI | None = None


def _get_chroma(db_path: str) -> chromadb.PersistentClient:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=db_path)
    return _chroma


def _get_openai() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai


def _embed_query(query: str, model: str) -> list[float]:
    resp = _get_openai().embeddings.create(input=[query], model=model)
    return resp.data[0].embedding


def retrieve(
    query: str,
    collection_name: str,
    top_k: int = 20,
    db_path: str = "db/chroma",
    embed_model: str = "text-embedding-3-small",
) -> list[Source]:
    """Dense retrieval from Chroma. Returns top_k Sources sorted by score desc."""
    client = _get_chroma(db_path)
    col = client.get_collection(collection_name)
    qemb = _embed_query(query, embed_model)
    results = col.query(
        query_embeddings=[qemb],
        n_results=min(top_k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    sources = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]
    for doc, meta, dist in zip(docs, metas, dists):
        sources.append(
            Source(
                doc_id=meta.get("source", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                text=doc,
                score=float(1 - dist),  # cosine distance → similarity
                header_path=meta.get("header_path"),
            )
        )
    return sources
