"""Section C (part 2) — Query rewriting (HyDE-style).

A short OpenAI call that:
  1. Expands financial abbreviations (op margin → operating margin)
  2. Resolves vague time references when possible
  3. Generates 1 hypothetical answer sentence to improve dense recall (HyDE)

The hypothetical sentence is concatenated to the query so the embedding
captures both the question and a plausible answer's vocabulary.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from openai import AzureOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings


_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        _client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
    return _client


REWRITE_SYSTEM = """You rewrite financial questions for retrieval over SEC 10-K/10-Q filings.

Output strict JSON with keys:
  "rewritten": the question with abbreviations expanded and vague terms made precise
  "hypothetical_answer": one short sentence that could plausibly answer it, using likely terminology from a 10-K (revenue, segment, operating margin, MD&A, etc.)

Do not invent specific numbers or company names that were not in the question."""


@lru_cache(maxsize=256)
def rewrite_query(query: str) -> dict:
    """Return {'rewritten': str, 'hypothetical_answer': str, 'combined': str}."""
    if not query.strip():
        return {"rewritten": query, "hypothetical_answer": "", "combined": query}

    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
            max_completion_tokens=400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        import json
        data = json.loads(resp.choices[0].message.content or "{}")
        rewritten = (data.get("rewritten") or query).strip()
        hyde = (data.get("hypothetical_answer") or "").strip()
    except Exception:
        rewritten, hyde = query, ""

    combined = f"{rewritten}\n{hyde}".strip() if hyde else rewritten
    return {"rewritten": rewritten, "hypothetical_answer": hyde, "combined": combined}


if __name__ == "__main__":
    import json, sys as _sys
    if len(_sys.argv) < 2:
        print("usage: python -m src.query_rewriter '<question>'"); raise SystemExit(1)
    print(json.dumps(rewrite_query(_sys.argv[1]), indent=2))
