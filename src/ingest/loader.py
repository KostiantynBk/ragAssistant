from pathlib import Path


def load_docs(docs_dir: str | Path) -> list[dict]:
    """Walk docs_dir and return list of {path, text} for every .md file."""
    root = Path(docs_dir)
    docs = []
    for p in sorted(root.rglob("*.md")):
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if text:
            docs.append({"path": str(p.relative_to(root)), "text": text})
    return docs
