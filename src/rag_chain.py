"""Section D — RAG chain with prompt controls.

Orchestrates: query rewrite → dense retrieve (MMR) → optional rerank →
guardrails → grounded generation with [chunk_N] citations.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from openai import AzureOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.retriever import retrieve, RetrievedChunk
from src.reranker import rerank
from src.query_rewriter import rewrite_query
from src.guardrails import (
    is_out_of_scope,
    is_insufficient_context,
    validate_citations,
    OUT_OF_SCOPE_MESSAGE,
    INSUFFICIENT_CONTEXT_MESSAGE,
)


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


SYSTEM_PROMPT = """You are a precise financial-analyst assistant.

Rules:
1. Answer ONLY using the provided context chunks.
2. Cite every claim with bracketed chunk markers like [chunk_1], [chunk_2].
3. If the context does not contain enough information to answer, say:
   "The provided filings do not contain sufficient information to answer this question."
   Do NOT speculate, fill gaps from prior knowledge, or invent figures.
4. Quote exact numbers from the context. Round only if the context rounds.
5. If asked to compare or trend, structure the answer with brief bullets.
6. Be concise — 4-8 sentences unless the question explicitly demands more detail."""


@dataclass
class Citation:
    index: int                 # 1-based as referenced in the answer
    chunk_id: str
    source: str
    page: int | None
    company: str | None
    year: int | None
    section: str | None
    score: float
    text: str


@dataclass
class RagResponse:
    query: str
    rewritten_query: str
    answer: str
    citations: list[Citation]
    used_chunks: list[dict]
    guardrail: str | None       # None | "out_of_scope" | "insufficient_context"
    citation_check: dict
    timings_ms: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            "citations": [asdict(c) for c in self.citations],
            "used_chunks": self.used_chunks,
            "guardrail": self.guardrail,
            "citation_check": self.citation_check,
            "timings_ms": self.timings_ms,
        }


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        m = c.metadata
        header_bits = []
        if m.get("company"):
            header_bits.append(str(m["company"]))
        if m.get("filing_type"):
            header_bits.append(str(m["filing_type"]))
        if m.get("year"):
            header_bits.append(str(m["year"]))
        if m.get("section"):
            header_bits.append(str(m["section"]))
        if m.get("page"):
            header_bits.append(f"p.{m['page']}")
        header = " · ".join(header_bits) if header_bits else (m.get("source") or "filing")
        blocks.append(f"[chunk_{i}] ({header})\n{c.text.strip()}")
    return "\n\n".join(blocks)


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    out = []
    for i, c in enumerate(chunks, start=1):
        m = c.metadata
        out.append(Citation(
            index=i,
            chunk_id=c.chunk_id,
            source=str(m.get("source", "")),
            page=int(m["page"]) if m.get("page") not in (None, "") else None,
            company=m.get("company"),
            year=int(m["year"]) if m.get("year") not in (None, "") else None,
            section=m.get("section"),
            score=round(float(c.score), 4),
            text=c.text,
        ))
    return out


def answer(
    query: str,
    filters: dict | None = None,
    use_rewriter: bool | None = None,
    use_reranker: bool | None = None,
) -> RagResponse:
    use_rewriter = settings.USE_QUERY_REWRITER if use_rewriter is None else use_rewriter
    use_reranker = settings.USE_RERANKER if use_reranker is None else use_reranker

    timings: dict[str, int] = {}
    t0 = time.perf_counter()

    # 1. Out-of-scope guardrail (cheap, deterministic)
    if is_out_of_scope(query):
        return RagResponse(
            query=query, rewritten_query=query, answer=OUT_OF_SCOPE_MESSAGE,
            citations=[], used_chunks=[], guardrail="out_of_scope",
            citation_check={"cited_indices": [], "invalid_indices": [], "has_any_citation": False, "all_valid": True},
            timings_ms={"total": int((time.perf_counter() - t0) * 1000)},
        )

    # 2. Query rewrite (HyDE)
    t = time.perf_counter()
    if use_rewriter:
        rewrite = rewrite_query(query)
        retrieval_query = rewrite["combined"]
        rewritten = rewrite["rewritten"]
    else:
        retrieval_query, rewritten = query, query
    timings["rewrite"] = int((time.perf_counter() - t) * 1000)

    # 3. Dense retrieve (with MMR)
    t = time.perf_counter()
    candidates = retrieve(retrieval_query, k=settings.TOP_K_RETRIEVE, filters=filters, use_mmr=True)
    timings["retrieve"] = int((time.perf_counter() - t) * 1000)

    # 4. Rerank (optional)
    if use_reranker and candidates:
        t = time.perf_counter()
        candidates = rerank(query, candidates, top_k=settings.TOP_K_RERANK)
        timings["rerank"] = int((time.perf_counter() - t) * 1000)
    else:
        candidates = candidates[: settings.TOP_K_RERANK]

    # 5. Insufficient-context guardrail
    if is_insufficient_context(candidates):
        cits = _build_citations(candidates)
        return RagResponse(
            query=query, rewritten_query=rewritten, answer=INSUFFICIENT_CONTEXT_MESSAGE,
            citations=cits,
            used_chunks=[{"chunk_id": c.chunk_id, "score": c.score, "metadata": c.metadata} for c in candidates],
            guardrail="insufficient_context",
            citation_check={"cited_indices": [], "invalid_indices": [], "has_any_citation": False, "all_valid": True},
            timings_ms={**timings, "total": int((time.perf_counter() - t0) * 1000)},
        )

    # 6. Generate
    context_block = _format_context(candidates)
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer (cite chunks):"

    t = time.perf_counter()
    client = _get_client()
    completion = client.chat.completions.create(
        model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT,
        max_completion_tokens=settings.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    timings["generate"] = int((time.perf_counter() - t) * 1000)

    answer_text = (completion.choices[0].message.content or "").strip()
    citations = _build_citations(candidates)
    cit_check = validate_citations(answer_text, n_chunks=len(candidates))

    timings["total"] = int((time.perf_counter() - t0) * 1000)
    return RagResponse(
        query=query,
        rewritten_query=rewritten,
        answer=answer_text,
        citations=citations,
        used_chunks=[{"chunk_id": c.chunk_id, "score": c.score, "metadata": c.metadata} for c in candidates],
        guardrail=None,
        citation_check=cit_check,
        timings_ms=timings,
    )


# ---------- CLI ----------

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--no-rewriter", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args()

    filters = {}
    if args.company:
        filters["company"] = args.company
    if args.year:
        filters["year"] = args.year

    resp = answer(
        args.query,
        filters=filters or None,
        use_rewriter=not args.no_rewriter,
        use_reranker=not args.no_reranker,
    )
    print("=" * 80)
    print(f"Q: {resp.query}")
    if resp.guardrail:
        print(f"[guardrail: {resp.guardrail}]")
    print("-" * 80)
    print(resp.answer)
    print("-" * 80)
    if resp.citations:
        print("Citations:")
        for c in resp.citations:
            cited_marker = "*" if c.index in resp.citation_check["cited_indices"] else " "
            print(f"  {cited_marker} [chunk_{c.index}] {c.company or ''} {c.year or ''} {c.section or ''} (p.{c.page or '-'}, score={c.score})")
    print(f"Timings: {resp.timings_ms}")
