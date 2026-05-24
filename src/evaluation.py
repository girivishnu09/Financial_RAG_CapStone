"""Section E — RAG triad evaluation (LLM-as-judge).

Three metrics, each scored on a 1-5 Likert scale by an LLM judge with a
strict rubric, then normalized to [0, 1]:

  * context_relevance — are the retrieved chunks relevant to the question?
  * groundedness      — is the answer supported by the retrieved context?
  * answer_relevance  — does the answer address the question asked?

Plus deterministic metrics that don't need a judge:
  * has_citations, citations_valid (from the citation validator)
  * guardrail_triggered (out-of-scope / insufficient-context)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from openai import AzureOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.rag_chain import answer, RagResponse


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


# ---------- judge prompts ----------

CTX_REL_SYSTEM = """You are an evaluation judge. Score how relevant the retrieved context is to the question.

Rubric (1-5):
  5 = every chunk is directly relevant
  4 = most chunks are relevant, 1 is borderline
  3 = mixed; about half are relevant
  2 = mostly off-topic with one weak hit
  1 = no chunks address the question

Return strict JSON: {"score": <int 1-5>, "reason": "<one sentence>"}"""

GROUNDEDNESS_SYSTEM = """You are an evaluation judge. Score how well the answer is supported by the provided context.

Rubric (1-5):
  5 = every claim is directly supported by the context
  4 = all major claims supported; 1 minor unsupported detail
  3 = ~half of claims supported; some plausible filler
  2 = most claims unsupported or contradicted by context
  1 = answer is fabricated relative to the context

If the answer says it cannot answer due to insufficient context, score 5 if context is indeed insufficient, 3 if context was actually sufficient.

Return strict JSON: {"score": <int 1-5>, "reason": "<one sentence>"}"""

ANSWER_REL_SYSTEM = """You are an evaluation judge. Score how well the answer addresses the question.

Rubric (1-5):
  5 = directly and completely answers the question
  4 = answers most of it, missing one minor piece
  3 = partial answer; ignores part of the question
  2 = tangential answer
  1 = does not address the question

A polite refusal due to out-of-scope or insufficient context counts as 5 if the question is genuinely outside the corpus, otherwise 2.

Return strict JSON: {"score": <int 1-5>, "reason": "<one sentence>"}"""


def _judge(system: str, user: str) -> tuple[float, str]:
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=settings.AZURE_OPENAI_JUDGE_DEPLOYMENT,
            max_completion_tokens=400,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        score = int(data.get("score", 0))
        reason = str(data.get("reason", "")).strip()
        score = max(1, min(5, score))
        return (score - 1) / 4.0, reason  # → [0, 1]
    except Exception as e:
        return 0.0, f"judge_error: {e}"


def _format_chunks_for_judge(used_chunks: list[dict], citations) -> str:
    """Render the context used in the answer for the judge to inspect."""
    blocks = []
    for c in citations:
        blocks.append(f"[chunk_{c.index}]\n{c.text[:1200]}")
    return "\n\n".join(blocks) if blocks else "(no chunks retrieved)"


@dataclass
class TriadScores:
    context_relevance: float
    groundedness: float
    answer_relevance: float
    notes: dict


def evaluate_response(question: str, resp: RagResponse) -> TriadScores:
    ctx_str = _format_chunks_for_judge(resp.used_chunks, resp.citations)

    ctx_user = f"Question: {question}\n\nRetrieved context:\n{ctx_str}"
    ctx_score, ctx_reason = _judge(CTX_REL_SYSTEM, ctx_user)

    grnd_user = f"Question: {question}\n\nContext:\n{ctx_str}\n\nAnswer:\n{resp.answer}"
    grnd_score, grnd_reason = _judge(GROUNDEDNESS_SYSTEM, grnd_user)

    ans_user = f"Question: {question}\n\nAnswer:\n{resp.answer}"
    ans_score, ans_reason = _judge(ANSWER_REL_SYSTEM, ans_user)

    return TriadScores(
        context_relevance=ctx_score,
        groundedness=grnd_score,
        answer_relevance=ans_score,
        notes={
            "context_relevance_reason": ctx_reason,
            "groundedness_reason": grnd_reason,
            "answer_relevance_reason": ans_reason,
        },
    )


@dataclass
class EvalRow:
    question: str
    answer: str
    guardrail: str | None
    has_citations: bool
    citations_all_valid: bool
    n_chunks: int
    top_score: float
    context_relevance: float
    groundedness: float
    answer_relevance: float
    rag_triad_mean: float
    notes: dict
    timings_ms: dict


def run_evaluation(questions: Iterable[dict]) -> list[EvalRow]:
    """Run the full RAG chain + triad judging for each question."""
    rows: list[EvalRow] = []
    for q in questions:
        question = q["question"]
        filters = q.get("filters")
        resp = answer(question, filters=filters)
        triad = evaluate_response(question, resp)

        top_score = max((c.score for c in resp.citations), default=0.0)
        triad_mean = (triad.context_relevance + triad.groundedness + triad.answer_relevance) / 3.0

        rows.append(EvalRow(
            question=question,
            answer=resp.answer,
            guardrail=resp.guardrail,
            has_citations=resp.citation_check["has_any_citation"],
            citations_all_valid=resp.citation_check["all_valid"],
            n_chunks=len(resp.citations),
            top_score=round(top_score, 4),
            context_relevance=round(triad.context_relevance, 3),
            groundedness=round(triad.groundedness, 3),
            answer_relevance=round(triad.answer_relevance, 3),
            rag_triad_mean=round(triad_mean, 3),
            notes=triad.notes,
            timings_ms=resp.timings_ms,
        ))
    return rows


def write_report(rows: list[EvalRow], out_path: Path | str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        out_path.write_text("# Evaluation Report\n\nNo rows.\n", encoding="utf-8")
        return

    n = len(rows)
    mean_ctx = sum(r.context_relevance for r in rows) / n
    mean_grnd = sum(r.groundedness for r in rows) / n
    mean_ans = sum(r.answer_relevance for r in rows) / n
    mean_triad = (mean_ctx + mean_grnd + mean_ans) / 3
    pct_cited = sum(1 for r in rows if r.has_citations) / n
    pct_valid = sum(1 for r in rows if r.citations_all_valid) / n
    n_oos = sum(1 for r in rows if r.guardrail == "out_of_scope")
    n_ic = sum(1 for r in rows if r.guardrail == "insufficient_context")

    lines = [
        "# RAG Evaluation Report",
        "",
        f"**Questions evaluated:** {n}",
        f"**LLM (Azure deployment):** `{settings.AZURE_OPENAI_CHAT_DEPLOYMENT}` (judge: `{settings.AZURE_OPENAI_JUDGE_DEPLOYMENT}`)  ",
        f"**Embeddings:** `{settings.EMBED_MODEL}` · **Reranker:** `{settings.RERANKER_MODEL if settings.USE_RERANKER else 'off'}`  ",
        f"**top_k retrieve / rerank:** {settings.TOP_K_RETRIEVE} / {settings.TOP_K_RERANK}  ",
        "",
        "## Aggregate Metrics (RAG Triad)",
        "",
        "| Metric | Mean (0-1) |",
        "|---|---|",
        f"| Context Relevance | {mean_ctx:.3f} |",
        f"| Groundedness | {mean_grnd:.3f} |",
        f"| Answer Relevance | {mean_ans:.3f} |",
        f"| **Triad Mean** | **{mean_triad:.3f}** |",
        "",
        "## Reliability",
        "",
        f"- Answers with at least one citation: **{pct_cited:.0%}**",
        f"- Answers with all citations valid: **{pct_valid:.0%}**",
        f"- Out-of-scope refusals: **{n_oos}**",
        f"- Insufficient-context responses: **{n_ic}**",
        "",
        "## Per-question Detail",
        "",
        "| # | Question | Guardrail | Cited | CtxRel | Ground | AnsRel | Triad | Top score |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows, start=1):
        q_short = (r.question[:80] + "…") if len(r.question) > 80 else r.question
        lines.append(
            f"| {i} | {q_short} | {r.guardrail or '-'} | "
            f"{'yes' if r.has_citations else 'no'} | "
            f"{r.context_relevance:.2f} | {r.groundedness:.2f} | {r.answer_relevance:.2f} | "
            f"{r.rag_triad_mean:.2f} | {r.top_score:.3f} |"
        )

    lines += ["", "## Sample Answers (first 5)", ""]
    for r in rows[:5]:
        lines += [
            f"### Q: {r.question}",
            "",
            f"**Answer:** {r.answer}",
            "",
            f"_Reason (groundedness):_ {r.notes.get('groundedness_reason', '')}",
            "",
            "---",
            "",
        ]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def load_eval_questions(path: Path | str) -> list[dict]:
    path = Path(path)
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions
