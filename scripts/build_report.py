"""Build the capstone PDF report from existing project artifacts.

Combines:
  - Cover page + executive summary (written here)
  - Architecture (architecture.md)
  - Rubric mapping table (written here)
  - Evaluation results (eval/eval_report.md)
  - Reproducibility / setup (extracted from README.md)

Outputs report.pdf at the project root.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


COVER = f"""# RAG-Based Question Answering over Financial Reports

**Capstone Project — Mid-Training Assignment**

**Author:** girivishnu09
**Date:** {date.today().isoformat()}
**Repository:** https://github.com/girivishnu09/Financial_RAG_CapStone

---

## Executive Summary

This capstone implements a production-minded Retrieval-Augmented Generation (RAG)
assistant that answers natural-language financial questions over SEC 10-K filings,
with grounded `[chunk_N]` citations, two-tier guardrails, and an LLM-as-judge
evaluation framework. The corpus comprises 10 recent 10-Ks across five large-cap
companies (Apple, Microsoft, Alphabet, Amazon, NVIDIA), totaling 938 retrieval
chunks. Generation runs through Azure OpenAI's `gpt-5.4-mini`; embeddings use
Azure `text-embedding-3-large` (3072-dim) into a local Chroma index. The
evaluation suite covers 25 test questions and yields a **RAG-triad mean of 0.637**
(Groundedness 0.750, Answer Relevance 0.720, Context Relevance 0.440), with
**100% of cited chunks valid** and correct refusal behavior on out-of-scope and
insufficient-context inputs. The interactive demo is a single-file Streamlit app
exposing metadata filters, retrieval debug, and expandable citation cards.
"""


RUBRIC_MAPPING = """## Implementation by Graded Section

The codebase is organized one-concern-per-file so each rubric requirement maps
to a clearly named module. The table below is a one-glance map from rubric
section to implementation file(s); design decisions and trade-offs are
expanded in the architecture section that follows.

| Rubric Section | Marks | Implementation Highlights | Key Files |
|---|---|---|---|
| **A. Data ingestion + cleaning** | 15 | PDF/HTML/TXT loaders; iXBRL tag stripping (preserves narrative inside `<ix:*>` wrappers, drops `<link>` / `<xbrli>` plumbing); regex-based extraction of `company`, `year`, `filing_type`, `section`, `page` | `src/ingest.py` |
| **B. Chunking + embeddings + indexing** | 15 | Token-aware recursive splitter (target 800 tokens, 120 overlap, tiktoken `cl100k_base`); Azure `text-embedding-3-large` (3072-dim, normalized cosine); Chroma persistent HNSW index | `src/chunking.py`, `src/indexing.py` |
| **C. Retriever design + tuning** | 15 | Dense top-N then MMR (λ = 0.7) for diversity; Chroma `$and` metadata pre-filtering on company/year/filing_type; HyDE-style query rewriter with JSON-mode + lru_cache; optional cross-encoder reranker (toggleable for ablation) | `src/retriever.py`, `src/query_rewriter.py`, `src/reranker.py` |
| **D. RAG chain + prompt controls** | 15 | Strict 6-rule system prompt forbidding speculation, mandating `[chunk_N]` citations, requiring fixed refusal phrasing on insufficient context; deterministic temperature; structured `RagResponse` dataclass for unified UI/eval consumption | `src/rag_chain.py` |
| **E. Evaluation framework** | 15 | RAG-triad LLM-as-judge (context relevance, groundedness, answer relevance); 25-question test set with structured rubric prompts in JSON-mode; aggregate + per-question Markdown report generator | `src/evaluation.py`, `eval/test_questions.jsonl`, `scripts/run_eval.py` |
| **F. UI + citation display** | 15 | Streamlit chat with sidebar metadata filters; pipeline-stage toggles for live ablation; expandable per-citation cards showing source / page / section / score; debug panel exposing rewritten query and timings | `app/streamlit_app.py` |
| **G. Reliability + guardrails** | 10 | Lexical out-of-scope guard on the hot path (deterministic, free); similarity-floor insufficient-context guard with tunable threshold; post-generation citation validator | `src/guardrails.py` |

### Notes on Engineering Quality

- **Centralized config** (`config.py`, pydantic-settings): every tunable knob
  (chunk size, top-k, temperature, model deployments) lives in one file and is
  overridable via `.env`.
- **CLI entrypoints** for every stage (`python -m src.ingest`,
  `python -m src.retriever`, `python -m src.rag_chain`) so each component is
  testable in isolation without booting the UI.
- **Idempotent index build** — `index_chunks` upserts by `chunk_id`, so
  incremental document additions don't require a full rebuild.
"""


SETUP = """## Reproducibility and Setup

The project is fully reproducible from the GitHub repo with three commands.
Raw filings are not committed (large + license-redistributable from SEC EDGAR);
they are fetched on demand by `scripts/fetch_filings.py`.

### Quick start

```powershell
# 1. Clone + create a venv (path matters - long Windows paths can break torch)
git clone https://github.com/girivishnu09/Financial_RAG_CapStone.git
cd Financial_RAG_CapStone
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt

# 2. Configure Azure OpenAI in .env (copy from .env.example)
#    - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (resource base URL only)
#    - AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBED_DEPLOYMENT

# 3. Fetch SEC 10-Ks (≈ 1.5–8 MB each after iXBRL stripping)
python scripts\\fetch_filings.py --tickers AAPL MSFT GOOGL AMZN NVDA --limit 2

# 4. Build the Chroma index (≈ 3 minutes, one-time)
python scripts\\build_index.py --rebuild

# 5. Launch the demo
streamlit run app\\streamlit_app.py
```

### Corporate-firewall note

HuggingFace (`huggingface.co`) is blocked by Zscaler in Ecolab's network, so
the original plan of `BAAI/bge-small-en-v1.5` embeddings + cross-encoder
rerank was replaced by Azure-hosted `text-embedding-3-large`. The cross-encoder
rerank is therefore disabled by default (`USE_RERANKER=False`); the rest of
the retrieval pipeline (MMR + metadata filters + HyDE rewrite) compensates for
this without measurable quality loss on the eval set.
"""


LIMITATIONS_AND_NEXT = """## Limitations and Future Work

### Limitations

1. **Tabular data** in 10-Ks is extracted as flowing text via BeautifulSoup;
   multi-column financial tables can lose alignment, which affects precise
   number lookups (e.g., line-item revenue figures). A layout-aware parser
   (`unstructured`, `Camelot`) would help.
2. **No image / chart understanding** — figures embedded in filings are
   ignored.
3. **Metadata extraction is best-effort regex on filenames + chunk headers.**
   Filenames following the `<TICKER>_10-K_<YEAR>.html` convention give
   correct attribution; non-standard filenames fall back to filename stem.
4. **Cross-company comparisons** depend on dense retrieval surfacing the right
   chunks across companies; the eval set shows comparison questions are the
   weakest category (Triad ≈ 0.33 on Q4).
5. **No streaming output** — Streamlit shows a blocking status spinner during
   generation rather than incremental tokens.
6. **Judge bias** — the same model family (`gpt-5.4-mini`) generates and
   evaluates, which inflates absolute scores. The metrics should be read as
   *comparative* (between configurations, between questions) rather than
   ground truth.

### What we'd build next

- Hybrid retrieval (BM25 + dense, weighted) — financial filings have
  distinctive numeric anchors that BM25 catches well.
- Layout-aware ingestion for tabular data.
- Optional Streamlit caching of `(query, filters) → answer` for snappy
  re-asks during demo recordings.
- Stronger judge model (e.g., `gpt-4o` or Claude) run only at release time
  to reduce same-family bias in the eval.
- Inline streaming output via `client.chat.completions.create(stream=True)`.
"""


def main() -> None:
    arch = read(ROOT / "architecture.md")
    eval_report = read(ROOT / "eval" / "eval_report.md")
    demo_answers = read(ROOT / "demo_answers.md") if (ROOT / "demo_answers.md").exists() else ""

    parts = [
        COVER,
        "\n---\n",
        RUBRIC_MAPPING,
        "\n---\n",
        "## Architecture and Design Choices\n",
        "*The detailed architecture note (design choices, trade-offs, limitations) follows. "
        "It accompanies the rubric-mapping table above and is the long-form companion to it.*\n",
        # Strip the architecture.md top heading since we just added our own
        "\n".join(arch.split("\n")[1:]) if arch.startswith("# ") else arch,
        "\n---\n",
        eval_report,
        "\n---\n",
        SETUP,
        "\n---\n",
        LIMITATIONS_AND_NEXT,
    ]

    if demo_answers:
        parts += [
            "\n---\n",
            "## Demo Answers (Live Run)\n",
            "*The following answers were produced by running the assignment's six "
            "sample questions through the deployed RAG chain. Each answer is grounded "
            "in retrieved chunks and includes inline `[chunk_N]` citations. Source "
            "references and similarity scores follow each answer.*\n",
            "\n".join(demo_answers.split("\n")[1:]) if demo_answers.startswith("# ") else demo_answers,
        ]

    md_text = "\n\n".join(parts)
    md_out = ROOT / "report.md"
    md_out.write_text(md_text, encoding="utf-8")
    print(f"Wrote {md_out}  ({len(md_text):,} chars)")

    # Convert to HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"],
        output_format="xhtml1",
    )

    # Wrap in HTML with print-friendly CSS
    css = """
    @page { size: A4; margin: 1.8cm 2cm; }
    body { font-family: 'Helvetica', 'Arial', sans-serif; font-size: 10pt; line-height: 1.45; color: #222; }
    h1 { font-size: 20pt; color: #1a365d; border-bottom: 2px solid #1a365d; padding-bottom: 4px; margin-top: 18pt; }
    h2 { font-size: 14pt; color: #2c5282; margin-top: 16pt; border-bottom: 1px solid #cbd5e0; padding-bottom: 2px; }
    h3 { font-size: 12pt; color: #2d3748; margin-top: 12pt; }
    h4 { font-size: 11pt; color: #4a5568; margin-top: 10pt; }
    p, li { font-size: 10pt; }
    code { font-family: 'Consolas', 'Courier New', monospace; font-size: 9pt; background: #f7fafc; padding: 1px 4px; border-radius: 2px; color: #c53030; }
    pre { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 8px; font-size: 8.5pt; line-height: 1.3; overflow-x: auto; white-space: pre-wrap; }
    pre code { background: transparent; padding: 0; color: #2d3748; }
    table { border-collapse: collapse; width: 100%; margin: 8pt 0; font-size: 8.5pt; }
    th, td { border: 1px solid #cbd5e0; padding: 4px 6px; text-align: left; vertical-align: top; }
    th { background: #edf2f7; font-weight: bold; color: #1a365d; }
    tr:nth-child(even) td { background: #f7fafc; }
    blockquote { border-left: 3px solid #cbd5e0; padding-left: 10px; margin-left: 0; color: #4a5568; font-style: italic; }
    hr { border: none; border-top: 1px solid #cbd5e0; margin: 16pt 0; }
    """

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RAG Capstone Report</title>
<style>{css}</style></head>
<body>{html_body}</body></html>"""

    pdf_out = ROOT / "Financial_RAG_Capstone_Report.pdf"
    with pdf_out.open("wb") as f:
        result = pisa.CreatePDF(full_html, dest=f, encoding="utf-8")
    if result.err:
        print(f"PDF generation finished with {result.err} errors", file=sys.stderr)
    else:
        print(f"Wrote {pdf_out}  ({pdf_out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
