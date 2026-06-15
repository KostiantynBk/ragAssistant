from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    doc_id: str
    chunk_index: int
    text: str
    score: float
    header_path: str | None = None


class RAGAnswer(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool = Field(description="True when the answer is fully supported by retrieved context")
    latency_ms: float


class QueryRequest(BaseModel):
    question: str
    chunking: Literal["recursive", "header_aware"] = "recursive"
    use_reranker: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


class EvalResult(BaseModel):
    id: int
    config: str
    hit_at_1: bool
    hit_at_5: bool
    hit_at_10: bool
    mrr: float
    faithfulness: float
    avg_latency_ms: float
    timestamp: str
