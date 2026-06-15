from src.ingest.chunker import header_aware_chunks, recursive_chunks

SAMPLE_MD = """# Introduction

LangChain is a framework for building LLM-powered applications.

## Installation

Install via pip:

```
pip install langchain
```

## Quick Start

Here is a minimal example:

```python
from langchain.llms import OpenAI
llm = OpenAI()
print(llm("Hello"))
```

### Configuration

You can configure the model by passing parameters to the constructor.
"""


def test_recursive_chunks_returns_non_empty():
    chunks = recursive_chunks(SAMPLE_MD, source="test.md")
    assert len(chunks) >= 1
    for c in chunks:
        assert c["text"].strip()
        assert c["metadata"]["source"] == "test.md"
        assert isinstance(c["metadata"]["chunk_index"], int)


def test_recursive_chunks_cover_all_content():
    chunks = recursive_chunks(SAMPLE_MD, source="test.md")
    combined = " ".join(c["text"] for c in chunks)
    assert "LangChain" in combined
    assert "pip install" in combined


def test_header_aware_chunks_returns_non_empty():
    chunks = header_aware_chunks(SAMPLE_MD, source="test.md")
    assert len(chunks) >= 1
    for c in chunks:
        assert c["text"].strip()
        assert c["metadata"]["source"] == "test.md"


def test_header_aware_chunks_have_header_path():
    chunks = header_aware_chunks(SAMPLE_MD, source="test.md")
    paths = [c["metadata"].get("header_path") for c in chunks]
    non_none = [p for p in paths if p]
    assert len(non_none) > 0, "Expected at least some chunks to have a header_path"


def test_chunk_indices_are_sequential():
    for fn in (recursive_chunks, header_aware_chunks):
        chunks = fn(SAMPLE_MD, source="test.md")
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))
