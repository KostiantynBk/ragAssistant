"""
Tests for the FastAPI query entry point.

These mock retrieval, reranking, and generation so src.api.main can be checked
without calling OpenAI or Chroma.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.schemas import RAGAnswer, Source


def _make_answer() -> RAGAnswer:
    return RAGAnswer(
        answer="LangChain is a framework for LLM applications.",
        sources=[Source(doc_id="docs/intro.md", chunk_index=0, text="LangChain…", score=0.95)],
        grounded=True,
        latency_ms=120.0,
    )


@pytest.fixture()
def client():
    with patch("src.api.main.retrieve") as mock_ret, \
         patch("src.api.main.rerank") as mock_rr, \
         patch("src.api.main.generate") as mock_gen:
        mock_ret.return_value = [
            Source(doc_id="docs/intro.md", chunk_index=0, text="LangChain…", score=0.95)
        ]
        mock_rr.return_value = [
            Source(doc_id="docs/intro.md", chunk_index=0, text="LangChain…", score=0.95)
        ]
        mock_gen.return_value = _make_answer()
        from src.api.main import app
        yield TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_returns_rag_answer(client):
    r = client.post("/ask", json={"question": "What is LangChain?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["grounded"], bool)
    assert body["latency_ms"] >= 0


def test_ask_invalid_chunking(client):
    r = client.post("/ask", json={"question": "Q?", "chunking": "unknown"})
    assert r.status_code == 422  # pydantic validation error


def test_ask_top_k_bounds(client):
    r = client.post("/ask", json={"question": "Q?", "top_k": 0})
    assert r.status_code == 422

    r2 = client.post("/ask", json={"question": "Q?", "top_k": 21})
    assert r2.status_code == 422


def test_eval_latest_endpoint():
    from unittest.mock import patch as p2
    with p2("src.api.main.get_latest_evals") as mock_evals:
        mock_evals.return_value = []
        from src.api.main import app
        c = TestClient(app)
        r = c.get("/eval/latest?n=5")
        assert r.status_code == 200
        assert r.json() == []
