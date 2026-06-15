import json
import os

from openai import OpenAI

from src.api.schemas import Source
from src.generation.prompts import JUDGE_PROMPT

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def score_faithfulness(
    question: str,
    answer: str,
    sources: list[Source],
    model: str | None = None,
) -> tuple[bool, float]:
    """Return (grounded: bool, faithfulness_score: float 0-1)."""
    judge_model = model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    context = "\n\n".join(f"[{s.doc_id}#{s.chunk_index}] {s.text}" for s in sources)
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)

    resp = _get_client().chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
        grounded = bool(data.get("grounded", False))
        score = float(data.get("score", 0.0))
    except (json.JSONDecodeError, ValueError):
        grounded, score = False, 0.0

    return grounded, score
