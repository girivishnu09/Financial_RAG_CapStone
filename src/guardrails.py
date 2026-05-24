"""Section G — Reliability and guardrails.

Three checks:
  1. Out-of-scope detection (lexical fast-path + LLM fallback)
  2. Insufficient-context handling (similarity floor over retrieved chunks)
  3. Citation validation (every [chunk_N] in answer must reference a real chunk)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.retriever import RetrievedChunk


# ---------- 1) Out-of-scope ----------

_OOS_KEYWORDS = (
    "weather", "recipe", "joke", "song", "movie", "horoscope",
    "translate", "code review", "write code", "python code",
    "buy", "sell", "should i invest", "stock pick", "price target",
    "predict the price", "give me advice", "personal advice",
)

_FINANCE_HINTS = (
    "revenue", "profit", "margin", "10-k", "10-q", "filing", "segment",
    "guidance", "risk", "mdna", "md&a", "operating", "income", "expense",
    "cash flow", "balance sheet", "earnings", "fiscal", "year over year",
)


def is_out_of_scope_lexical(query: str) -> bool:
    q = query.lower().strip()
    if not q:
        return True
    if any(k in q for k in _OOS_KEYWORDS):
        # advice/recommendation phrasing is OOS even if the topic is financial
        return True
    return False


def is_in_scope_lexical(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _FINANCE_HINTS)


def is_out_of_scope(query: str) -> bool:
    """Fast lexical check first; assume in-scope otherwise.

    The LLM-as-classifier path is intentionally avoided here because we want
    a deterministic guardrail on the hot path. If the question contains an
    explicit OOS marker (advice, weather, etc.), reject; otherwise let
    insufficient-context handle the rest.
    """
    return is_out_of_scope_lexical(query)


OUT_OF_SCOPE_MESSAGE = (
    "I'm scoped to answer factual questions about the financial filings in this corpus "
    "(10-K / 10-Q SEC reports). I can't provide investment advice, predictions, or "
    "off-topic answers. Try asking about reported revenue, segment performance, risk "
    "factors, or MD&A commentary."
)


# ---------- 2) Insufficient context ----------

def is_insufficient_context(
    chunks: list[RetrievedChunk],
    threshold: float | None = None,
    min_chunks: int = 1,
) -> bool:
    if not chunks or len(chunks) < min_chunks:
        return True
    threshold = settings.INSUFFICIENT_CONTEXT_THRESHOLD if threshold is None else threshold
    # Use the top similarity rather than the mean — one strong hit can be enough.
    top = max(c.score for c in chunks)
    return top < threshold


INSUFFICIENT_CONTEXT_MESSAGE = (
    "I cannot find sufficient evidence in the indexed filings to answer this with confidence. "
    "Either the relevant filing isn't in the corpus, or the question needs to be more specific "
    "(e.g., name a company and year)."
)


# ---------- 3) Citation validation ----------

_CITATION_RE = re.compile(r"\[chunk[_\s-]?(\d+)\]", re.IGNORECASE)


def extract_cited_indices(answer: str) -> list[int]:
    return [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]


def validate_citations(answer: str, n_chunks: int) -> dict:
    cited = extract_cited_indices(answer)
    invalid = [c for c in cited if c < 1 or c > n_chunks]
    return {
        "cited_indices": cited,
        "invalid_indices": invalid,
        "has_any_citation": len(cited) > 0,
        "all_valid": len(invalid) == 0,
    }
