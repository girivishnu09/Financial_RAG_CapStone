# RAG-Based Q&A over Financial Reports

Production-minded Retrieval-Augmented Generation assistant that answers natural-language questions over SEC 10-K / 10-Q filings, with grounded citations, guardrails, and an evaluation framework.

> Capstone deliverable for the Mid-Training assignment — covers all seven graded sections (A–G).

## Highlights

- **Robust ingestion** — PDF / HTML / TXT loaders with cleaning + automatic metadata sniffing (company, year, filing type, section, page).
- **Token-aware chunking** — recursive splitter with 800-token chunks and 120-token overlap, preserving section context.
- **Local vector index** — Chroma + `BAAI/bge-small-en-v1.5` embeddings (free, no API costs at index time).
- **Hybrid retrieval** — dense similarity + MMR for diversity, optional cross-encoder rerank, optional HyDE query rewriting, metadata pre-filters (company / year / filing).
- **Grounded generation** — strict prompt that forbids speculation, requires `[chunk_N]` citations, refuses politely when context is missing.
- **Two-tier guardrails** — lexical out-of-scope detection + similarity-floor insufficient-context handler + post-hoc citation validator.
- **RAG-triad evaluation** — LLM-as-judge scoring of context relevance, groundedness, and answer relevance over 25+ test questions, with a generated Markdown report.
- **Streamlit demo** — chat UI with sidebar filters, citation cards, pipeline-toggle ablation, debug panel.

## Architecture

```
data/raw/  ──►  ingest  ──►  chunk  ──►  embed (BGE)  ──►  Chroma (persistent)
                                                              │
                          ┌──────────────────────────────────┘
                          ▼
        rewrite (HyDE)  ──►  retrieve (MMR + filters)  ──►  rerank (cross-encoder)
                                                              │
                          ┌──────────────────────────────────┘
                          ▼
                guardrails (OOS, insufficient context)
                          │
                          ▼
        OpenAI gpt-4o-mini  ──►  answer + [chunk_N] citations  ──►  Streamlit
```

Full design discussion in [`architecture.md`](./architecture.md).

## Setup

### 1. Clone + create a virtual environment

```powershell
cd C:\Users\arantar\Downloads\rag-financial-capstone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI

Copy `.env.example` to `.env` and fill in your Azure values:

```powershell
copy .env.example .env
# then edit .env
```

You need:

| Variable | Where to find it |
|---|---|
| `AZURE_OPENAI_API_KEY` | Azure portal → your OpenAI resource → "Keys and Endpoint" |
| `AZURE_OPENAI_ENDPOINT` | Same page, e.g. `https://my-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_VERSION` | Defaults to `2024-10-21`; bump if your deployment requires a newer one |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Deployment name you created (e.g. `gpt-4o-mini`). This is **not** the model name — it's whatever you named the deployment in Azure |
| `AZURE_OPENAI_JUDGE_DEPLOYMENT` | Same idea; usually identical to chat deployment |

### 3. Drop SEC filings into `data/raw/`

Get the [Kaggle SEC Financial Statement Extracts](https://www.kaggle.com/) dataset (10-K / 10-Q) — or any 10-K / 10-Q PDFs / HTML — and place files into `data/raw/`. Subdirectories are walked recursively.

Filename convention helps metadata extraction (optional but recommended):

```
data/raw/AAPL_10-K_2023.pdf
data/raw/MSFT_10-Q_2024_Q1.html
```

The ingestor will auto-detect `company`, `year`, and `filing_type` from filenames where possible, and infer `section` (Item 1A Risk Factors, Item 7 MD&A, etc.) from chunk content.

### 4. Build the vector index

```powershell
python scripts\build_index.py --rebuild --write-chunks
```

This will:
- Walk `data/raw/`, load every PDF/HTML/TXT
- Clean text, extract metadata, chunk with overlap
- Embed all chunks with BGE-small (CPU is fine; first run downloads the model)
- Persist to `chroma_db/`

Re-running without `--rebuild` is a no-op if the collection is already populated.

### 5. Run the demo

```powershell
streamlit run app\streamlit_app.py
```

Open the printed URL (usually `http://localhost:8501`).

### 6. Run the evaluation

```powershell
python scripts\run_eval.py
```

Writes `eval/eval_report.md` (aggregate triad scores + per-question table) and `eval/eval_rows.jsonl` (raw rows).

You can limit the run while iterating:

```powershell
python scripts\run_eval.py --limit 5
```

## Project layout

```
rag-financial-capstone/
├── README.md                     ← you are here
├── architecture.md               ← design choices, trade-offs, limitations
├── requirements.txt
├── .env.example
├── config.py                     ← centralized settings (chunk size, models, k, thresholds)
├── data/
│   ├── raw/                      ← drop SEC filings here
│   └── processed/                ← chunks.jsonl (inspectable)
├── src/
│   ├── ingest.py                 ← Section A
│   ├── chunking.py               ← Section B (chunker)
│   ├── indexing.py               ← Section B (embed + Chroma)
│   ├── retriever.py              ← Section C (dense + MMR + filters)
│   ├── query_rewriter.py         ← Section C (HyDE)
│   ├── reranker.py               ← Section C (cross-encoder)
│   ├── rag_chain.py              ← Section D (orchestration + prompt)
│   ├── guardrails.py             ← Section G
│   └── evaluation.py             ← Section E (RAG triad)
├── scripts/
│   ├── build_index.py
│   └── run_eval.py
├── eval/
│   ├── test_questions.jsonl      ← 25 test questions
│   └── eval_report.md            ← generated
├── app/
│   └── streamlit_app.py          ← Section F
└── chroma_db/                    ← persisted index (gitignored)
```

## Sample questions for the demo

- What was the reported revenue trend in the latest two periods?
- Which factors were cited for profit increase or decrease?
- What risks were highlighted in management discussion?
- Compare operating margin across two selected years.
- What assumptions are stated for forward-looking guidance?
- Which segment contributed most to growth and why?
- Should I buy this company's stock? *(out-of-scope refusal)*
- What was the cryptocurrency holdings disclosed in 1873 fiscal year? *(insufficient-context refusal)*

## CLI quick reference

```powershell
# Inspect what one PDF would ingest
python -m src.ingest data\raw\AAPL_10-K_2023.pdf --limit 2

# Probe the retriever directly
python -m src.retriever "operating margin trend" --k 4 --company Apple

# Run the full chain from CLI
python -m src.rag_chain "What drove revenue growth in fiscal 2023?" --company Apple --year 2023

# Try a query rewrite
python -m src.query_rewriter "op margin yoy"
```

## Design notes (one-liners)

- **Chroma over FAISS** — metadata filtering and persistence ergonomics outweigh raw speed at this corpus size.
- **BGE-small embeddings** — local, free, strong on financial text; OpenAI embeddings are a 1-line swap if needed.
- **Azure OpenAI** for both generation and judging — uses deployment names from `.env` (default `gpt-4o-mini`).
- **MMR + cross-encoder rerank** — diversifies dense hits before a stricter relevance pass; both toggleable for ablation.
- **Two-tier guardrails** — deterministic lexical OOS check on the hot path, similarity-floor insufficient-context check after retrieval, post-gen citation validator.

See `architecture.md` for trade-offs and limitations.
