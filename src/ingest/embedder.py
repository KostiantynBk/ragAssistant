import os
import uuid

import chromadb
from openai import OpenAI

_client: OpenAI | None = None
_chroma: chromadb.PersistentClient | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _chroma_client(db_path: str) -> chromadb.PersistentClient:
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=db_path)
    return _chroma


def _embed_batch(texts: list[str], model: str) -> list[list[float]]:
    resp = _openai().embeddings.create(input=texts, model=model)
    return [r.embedding for r in resp.data]


def upsert_chunks(
    chunks: list[dict],
    collection_name: str,
    db_path: str = "db/chroma",
    embed_model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> int:
    """Embed and upsert chunks into a named Chroma collection. Returns count upserted."""
    client = _chroma_client(db_path)
    col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    upserted = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = _embed_batch(texts, embed_model)
        ids = [str(uuid.uuid4()) for _ in batch]
        metadatas = [c["metadata"] for c in batch]
        col.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        upserted += len(batch)

    return upserted
