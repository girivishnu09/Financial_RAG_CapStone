"""Section C (part 1) — Dense retriever with MMR + metadata filtering.

Wraps Chroma similarity search and adds:
  - Maximal Marginal Relevance (MMR) for result diversity
  - Metadata pre-filtering (company, year, filing_type, section)
  - Score normalization and a lightweight RetrievedChunk dataclass
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.indexing import get_collection, embed_texts


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict
    score: float  # cosine similarity in [0, 1]

    def short_source(self) -> str:
        m = self.metadata
        bits = [m.get("company") or m.get("source"), str(m.get("year") or ""), m.get("section") or ""]
        return " · ".join(b for b in bits if b)


def _build_chroma_where(filters: dict | None) -> dict | None:
    """Translate flat filters into Chroma's $and form when there are >=2 keys."""
    if not filters:
        return None
    cleaned = {k: v for k, v in filters.items() if v is not None and v != ""}
    if not cleaned:
        return None
    if len(cleaned) == 1:
        k, v = next(iter(cleaned.items()))
        return {k: {"$eq": v}}
    return {"$and": [{k: {"$eq": v}} for k, v in cleaned.items()]}


def _mmr(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    k: int,
    lambda_mult: float,
) -> list[int]:
    """Return indices of `k` chunks selected via MMR."""
    if len(doc_vecs) == 0:
        return []
    selected: list[int] = []
    candidate_idxs = list(range(len(doc_vecs)))
    # Pre-compute similarity to query
    sims_to_query = doc_vecs @ query_vec
    while candidate_idxs and len(selected) < k:
        best_idx = -1
        best_score = -1e9
        for idx in candidate_idxs:
            if not selected:
                score = sims_to_query[idx]
            else:
                max_sim_to_selected = max(float(doc_vecs[idx] @ doc_vecs[s]) for s in selected)
                score = lambda_mult * sims_to_query[idx] - (1 - lambda_mult) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(best_idx)
        candidate_idxs.remove(best_idx)
    return selected


def retrieve(
    query: str,
    k: int | None = None,
    filters: dict | None = None,
    use_mmr: bool = True,
) -> list[RetrievedChunk]:
    """Embed the query, fetch top-N from Chroma, optionally diversify via MMR."""
    k = k or settings.TOP_K_RETRIEVE
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_vec = np.array(embed_texts([query])[0], dtype=np.float32)
    n_fetch = max(k * 3, 20) if use_mmr else k

    where = _build_chroma_where(filters)
    res = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(n_fetch, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    embs = res["embeddings"][0]
    if not docs:
        return []

    # Cosine distance → cosine similarity (Chroma uses 1 - cos)
    sims = [1.0 - float(d) for d in dists]

    if use_mmr and len(docs) > k:
        emb_arr = np.array(embs, dtype=np.float32)
        order = _mmr(query_vec, emb_arr, k, settings.MMR_LAMBDA)
    else:
        order = list(range(min(k, len(docs))))

    out: list[RetrievedChunk] = []
    for i in order:
        meta = dict(metas[i])
        cid = meta.get("chunk_id", f"idx_{i}")
        out.append(RetrievedChunk(chunk_id=cid, text=docs[i], metadata=meta, score=sims[i]))
    return out


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    args = parser.parse_args()

    filters = {}
    if args.company:
        filters["company"] = args.company
    if args.year:
        filters["year"] = args.year

    results = retrieve(args.query, k=args.k, filters=filters or None)
    for r in results:
        print(f"[{r.score:.3f}] {r.short_source()}")
        print(f"  {r.text[:200].replace(chr(10), ' ')}")
        print()
