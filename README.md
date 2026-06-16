# DocsRAG — Retrieval-Augmented Documentation Assistant

A production-style RAG pipeline that answers questions over a technical-docs corpus (LangChain docs) with **cited, schema-validated responses**.

## Features

| Capability | Detail |
|---|---|
| **Chunking strategies** | Recursive character split vs. markdown-header-aware split |
| **Retrieval** | OpenAI `text-embedding-3-small` embeddings → Chroma vector DB (cosine similarity) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder |
| **Generation** | OpenAI `gpt-4o-mini` with strict context-grounding prompt |
| **Agent loop** | OpenAI function-calling loop exposing `retrieve_docs` as a tool |
| **Output validation** | Pydantic v2 `RAGAnswer` with inline `Source` citations and `grounded` flag |
| **Faithfulness** | LLM-as-judge scores each answer 0–1 |
| **Eval harness** | Hit@1/5/10 + MRR across 4 configs (2 chunking × 2 reranker) |
| **Persistence** | SQLite logs every query and eval run |
| **API** | FastAPI with `POST /ask`, `GET /eval/latest`, `GET /health` |

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/KostiantynBk/docsRAG.git
cd docsRAG
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 3. Download LangChain docs (markdown source)
git clone --depth 1 https://github.com/langchain-ai/langchain.git _tmp_lc
# Windows
xcopy /E /I _tmp_lc\docs\docs data\docs
# macOS / Linux
cp -r _tmp_lc/docs/docs data/docs
rm -rf _tmp_lc

# 4. Ingest (creates two Chroma collections)
python scripts/ingest.py

# 5. Start the API
uvicorn src.api.main:app --reload
```

> Note: the LangChain repository layout may change. If `_tmp_lc\docs\docs` does not exist on Windows,
> copy the available Markdown files instead:
>
> ```powershell
> git clone --depth 1 https://github.com/langchain-ai/langchain.git tmp_lc
> xcopy tmp_lc\*.md data\docs /S /I /Y
> python scripts/ingest.py
> ```

On PowerShell, keep the API server running in one terminal:

```powershell
uvicorn src.api.main:app --reload
```

Then open a second PowerShell terminal and call the API with `Invoke-RestMethod`:

```powershell
$response = Invoke-RestMethod `
  -Uri "http://localhost:8000/ask" `
  -Method Post `
  -ContentType "application/json" `
  -Body (@{
    question = "What is LangChain?"
    chunking = "header_aware"
    use_reranker = $true
    top_k = 5
  } | ConvertTo-Json)
```

Inspect the response:

```powershell
$response.answer
$response.grounded
$response.latency_ms
$response.sources | Format-List
$response | ConvertTo-Json -Depth 10
```

Then query it:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is LCEL?", "chunking": "header_aware", "use_reranker": true}'
```

## API

### `POST /ask`

```json
{
  "question": "string",
  "chunking": "recursive" | "header_aware",
  "use_reranker": true,
  "top_k": 5
}
```

Response (`RAGAnswer`):

```json
{
  "answer": "LCEL is ...",
  "sources": [
    {"doc_id": "expression_language/index.md", "chunk_index": 2, "text": "...", "score": 0.91}
  ],
  "grounded": true,
  "latency_ms": 843.2
}
```

### `GET /eval/latest?n=10`

Returns the last `n` evaluation runs from SQLite.

### `GET /health`

Returns `{"status": "ok"}`.

## Evaluation

Run the harness over all 4 configurations:

```bash
python scripts/evaluate.py
# or with SQLite logging + markdown output:
python scripts/compare_configs.py
```

### Results

> Replace the placeholder rows below with output from `scripts/compare_configs.py` after running.

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | Faithfulness | Avg Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| recursive | — | — | — | — | — | — |
| recursive+rr | — | — | — | — | — | — |
| header | — | — | — | — | — | — |
| header+rr | — | — | — | — | — | — |

## Project Structure

```
src/
  ingest/      loader, chunker (2 strategies), embedder → Chroma
  retrieval/   dense retriever + cross-encoder reranker
  generation/  prompt templates, OpenAI generator
  agent/       tool-calling agent loop
  eval/        harness (hit@k, MRR), LLM-as-judge, SQLite logger
  api/         FastAPI app + Pydantic schemas
scripts/
  ingest.py          ingest both collections
  evaluate.py        print eval table
  compare_configs.py write results to SQLite + print markdown table
tests/
  test_chunker.py    unit tests for both chunking strategies
  test_retrieval.py  mocked retriever + reranker tests
  test_api.py        FastAPI TestClient tests
eval_qa.json         20 hand-crafted QA pairs for evaluation
```

## Running Tests

```bash
pytest tests/ -v
```

## Agent Mode

```python
from src.agent.loop import agent_answer
result = agent_answer("How do I stream responses from LangChain?")
print(result.answer)
print([s.doc_id for s in result.sources])
```

## Tech Stack

Python · OpenAI API · Chroma · Sentence-Transformers · LangChain · FastAPI · Pydantic v2 · SQLite
