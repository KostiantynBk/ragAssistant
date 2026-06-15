from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.api.schemas import Source

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_encoder: CrossEncoder | None = None


def _get_encoder() -> CrossEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CrossEncoder(_MODEL_NAME)
    return _encoder


def rerank(query: str, sources: list[Source], keep_top_k: int = 5) -> list[Source]:
    """Re-score sources with a cross-encoder and return the top keep_top_k."""
    if not sources:
        return sources
    encoder = _get_encoder()
    pairs = [(query, s.text) for s in sources]
    scores = encoder.predict(pairs)
    ranked = sorted(zip(scores, sources), key=lambda x: x[0], reverse=True)
    return [s for _, s in ranked[:keep_top_k]]
