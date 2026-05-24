"""Section C (part 3) — Cross-encoder reranker.

Re-scores candidate chunks against the query using a small cross-encoder
(ms-marco-MiniLM-L-6-v2) and trims to top-K. Toggleable so eval can
ablate with/without rerank.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.retriever import RetrievedChunk


_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    top_k = top_k or settings.TOP_K_RERANK
    model = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = model.predict(pairs, show_progress_bar=False)
    order = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
    out: list[RetrievedChunk] = []
    for i in order[:top_k]:
        c = chunks[i]
        # Replace dense score with rerank score (kept in [0,1] roughly via sigmoid-ish range)
        c.score = float(scores[i])
        out.append(c)
    return out
