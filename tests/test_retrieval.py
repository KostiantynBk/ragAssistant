from unittest.mock import MagicMock, patch

import pytest

from src.api.schemas import Source


def _make_source(**kwargs) -> Source:
    defaults = dict(doc_id="docs/a.md", chunk_index=0, text="sample text", score=0.9)
    return Source(**(defaults | kwargs))


# ---------------------------------------------------------------------------
# reranker
# ---------------------------------------------------------------------------

def test_rerank_returns_top_k():
    from src.retrieval.reranker import rerank

    sources = [_make_source(chunk_index=i, text=f"text {i}") for i in range(10)]
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = list(range(10))  # scores 0..9

    with patch("src.retrieval.reranker._encoder", mock_encoder):
        result = rerank("query", sources, keep_top_k=3)

    assert len(result) == 3


def test_rerank_empty_input():
    from src.retrieval.reranker import rerank

    assert rerank("query", [], keep_top_k=5) == []


def test_rerank_orders_by_score():
    from src.retrieval.reranker import rerank

    sources = [_make_source(chunk_index=i, text=f"text {i}") for i in range(3)]
    scores = [0.1, 0.9, 0.5]
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = scores

    with patch("src.retrieval.reranker._encoder", mock_encoder):
        result = rerank("query", sources, keep_top_k=3)

    assert result[0].chunk_index == 1  # highest score
    assert result[1].chunk_index == 2
    assert result[2].chunk_index == 0


# ---------------------------------------------------------------------------
# retriever — mock Chroma + OpenAI
# ---------------------------------------------------------------------------

def test_retrieve_returns_sources():
    from src.retrieval.retriever import retrieve

    mock_col = MagicMock()
    mock_col.count.return_value = 5
    mock_col.query.return_value = {
        "documents": [["chunk text"]],
        "metadatas": [[{"source": "docs/a.md", "chunk_index": 0}]],
        "distances": [[0.1]],
    }
    mock_chroma = MagicMock()
    mock_chroma.get_collection.return_value = mock_col

    mock_openai = MagicMock()
    mock_openai.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.0] * 1536)]
    )

    with patch("src.retrieval.retriever._chroma", mock_chroma), \
         patch("src.retrieval.retriever._openai", mock_openai):
        sources = retrieve("test query", "test_collection", top_k=5)

    assert len(sources) == 1
    assert sources[0].doc_id == "docs/a.md"
    assert sources[0].score == pytest.approx(0.9)
