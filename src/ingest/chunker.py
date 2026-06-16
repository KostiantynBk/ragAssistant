"""
Document chunking step in the ingestion phase.

Called by scripts/ingest.py after src.ingest.loader. It creates recursive and
header-aware chunk lists; the next step is src.ingest.embedder.
"""

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
_RECURSIVE = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
_SECTION_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


def recursive_chunks(text: str, source: str) -> list[dict]:
    """Split text with RecursiveCharacterTextSplitter."""
    splits = _RECURSIVE.split_text(text)
    return [
        {"text": t, "metadata": {"source": source, "chunk_index": i}}
        for i, t in enumerate(splits)
        if t.strip()
    ]


def header_aware_chunks(text: str, source: str) -> list[dict]:
    """Split text on markdown headers, then sub-split large sections."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
    header_docs = splitter.split_text(text)

    chunks = []
    idx = 0
    for doc in header_docs:
        header_path = " > ".join(
            v for k in ("h1", "h2", "h3") if (v := doc.metadata.get(k))
        )
        sub_texts = _SECTION_SPLITTER.split_text(doc.page_content)
        for sub in sub_texts:
            if sub.strip():
                chunks.append(
                    {
                        "text": sub,
                        "metadata": {
                            "source": source,
                            "chunk_index": idx,
                            "header_path": header_path or None,
                        },
                    }
                )
                idx += 1
    return chunks
