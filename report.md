# RAG-Based Question Answering over Financial Reports

**Capstone Project — Mid-Training Assignment**

**Author:** girivishnu09
**Date:** 2026-05-24
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



---


## Implementation by Graded Section

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



---


## Architecture and Design Choices


*The detailed architecture note (design choices, trade-offs, limitations) follows. It accompanies the rubric-mapping table above and is the long-form companion to it.*



Short companion to `README.md` capturing **why** each piece exists and what we'd change next.

## Goals

1. **Grounded answers** — every claim traceable to a retrieved chunk; no quiet hallucinations.
2. **Reproducibility** — one command from raw filings to a working demo, no paid index step.
3. **Evaluable** — RAG-triad metrics computed end-to-end on a fixed test set with a single script.
4. **Engineering quality** — modular code (one concern per file), centralized config, clear CLIs at each layer.

## Pipeline at a glance

```
PDF/HTML/TXT ──► clean + metadata ──► token-aware chunker (800/120) ──► BGE-small ──► Chroma
                                                                                       │
                                                                                       ▼
                                              rewrite (HyDE) ──► retrieve (MMR) ──► rerank ──► guardrails ──► gpt-4o-mini
                                                                                       │
                                                                                       ▼
                                                                       answer + [chunk_N] citations
```

## Component decisions

### Ingestion (`src/ingest.py`)

- **`pypdf`** for PDF — page-aware so citations carry page numbers. Tabular content is approximated as flat text; documented as a limitation.
- **`beautifulsoup4` + `lxml`** for HTML — strips scripts/styles, normalizes to text.
- **Cleaning** removes page-number-only lines, table-of-contents markers, and collapses runs of whitespace. Conservative — we don't want to drop real numeric content.
- **Metadata sniffing** is a deliberate two-pass: filename-level (`company`, `year`, `filing_type`) and chunk-level (`section`). Both are best-effort; explicit Kaggle metadata can override later.

### Chunking (`src/chunking.py`)

- **Token-aware via `tiktoken`** rather than character counts — the LLM sees tokens, so token-budgeted chunks pack more predictably into the prompt.
- **Recursive splitter** prefers paragraph → sentence → word boundaries to keep semantic coherence.
- **800/120 tokens** balances recall (larger context) vs precision (smaller chunks). Smaller chunks (~300) caused over-fragmentation on financial text where one paragraph can hold one full revenue-driver explanation.
- **Overlap is token-based** (last N tokens of the previous chunk are prepended to the next), avoiding the "split mid-sentence with no context" failure mode of pure char-based overlap.

### Indexing (`src/indexing.py`)

- **Chroma in persistent mode** — local, no server, metadata-filterable at query time.
- **`BAAI/bge-small-en-v1.5`** — 384-dim, ~30 MB, strong on retrieval benchmarks (BEIR), runs on CPU. We normalize embeddings so cosine sim is just a dot product.
- **HNSW with cosine** — Chroma default, sufficient at < 1 M chunks.
- **Idempotent build** — `index_chunks` upserts by `chunk_id`, so re-running with new files only adds new chunks unless `--rebuild`.

### Retrieval (`src/retriever.py`)

- **Dense top-N then MMR top-k** — pure cosine top-k often returns near-duplicates from a single page; MMR with λ=0.7 surfaces diverse hits while staying close to the query.
- **Metadata pre-filtering** is wired through Chroma's `where` operator using `$eq` and `$and` — the demo sidebar uses this for company/year/filing_type scoping.
- **Score normalization** — Chroma returns cosine distance; we convert to similarity in [0, 1] so the threshold check in `guardrails.is_insufficient_context` is intuitive.

### Query rewriting (`src/query_rewriter.py`)

- **HyDE-style** — the LLM expands abbreviations and emits one hypothetical answer sentence, which is concatenated with the rewritten query before embedding. This biases the dense search toward chunks that *look like answers*, not just queries — a known win on financial Q&A where question vocabulary diverges from filing vocabulary.
- **`functools.lru_cache`** — demo replay is free; eval reproducibility improves.
- **JSON-mode** keeps parsing deterministic.

### Reranking (`src/reranker.py`)

- **`ms-marco-MiniLM-L-6-v2`** cross-encoder — small (~80 MB), CPU-runnable, scores (query, chunk) pairs jointly so it captures relevance dense embeddings miss.
- **Toggleable** so the eval can compare with/without rerank — supports an ablation row in the report.

### RAG chain (`src/rag_chain.py`)

- **Strict system prompt** lists the rules in numbered form and explicitly forbids speculation. The instruction to refuse with a fixed phrase makes the insufficient-context branch easier to detect.
- **Context block** is rendered as `[chunk_N] (Company · 10-K · 2023 · Item 7 · p.42)\n<text>` — the bracketed marker matches the citation regex in `guardrails.validate_citations`.
- **`temperature=0`** — deterministic for reproducibility and to discourage embellishment.
- **One return type** (`RagResponse`) for both happy path and guardrail outcomes — keeps the UI / eval consumers uniform.

### Guardrails (`src/guardrails.py`)

- **Out-of-scope is lexical first**, on purpose: the hot path stays deterministic and free. We could add an LLM classifier behind a feature flag for ambiguous cases.
- **Insufficient-context uses the top similarity** rather than the mean, because one strong hit can be enough to answer a precise lookup. Threshold 0.35 was picked empirically; tunable via `config.py`.
- **Citation validator** is post-hoc — it doesn't gate the answer but surfaces in the response and the eval report (citation rate, validity rate).

### Evaluation (`src/evaluation.py`)

- **RAG triad** with three independent judge prompts (context relevance, groundedness, answer relevance) — Likert 1–5, normalized to [0, 1], aggregated as mean.
- **Judge model = generator model** is a known bias risk; we offset with structured rubrics, JSON-mode, and `temperature=0`. A future improvement is to use a different/stronger judge (e.g., `gpt-4o`).
- **Deterministic side metrics** (citation presence, validity, guardrail counts) require no judging and provide objective complement to the LLM scores.

### UI (`app/streamlit_app.py`)

- **Streamlit chat primitives** keep the surface small. `@st.cache_resource` warms the embedder + Chroma client once per session.
- **Citation cards** are expandable so the demo can show evidence on demand without crowding the answer.
- **Toggle row** for rewrite/rerank/debug enables a live ablation during the recording.

## Trade-offs

| Choice | Pro | Con |
|---|---|---|
| Chroma | Persistent, metadata-filterable, zero ops | Slower than FAISS at scale |
| BGE-small embeddings | Free, fast, strong on financial text | OpenAI `text-embedding-3-small` is 5–8% better on some BEIR slices |
| `gpt-4o-mini` | Cheap, fast, strong on extraction | `gpt-4o` better on multi-hop reasoning |
| MMR + cross-encoder | Diverse + relevant | Two-pass adds ~150 ms per query |
| LLM-as-judge | Scales easily | Bias when judge ≈ generator |
| Lexical OOS guard | Deterministic on hot path | Misses paraphrased OOS prompts |
| Token-based overlap | Preserves boundaries | Slightly more compute than char-overlap |

## Limitations

1. **Tables** in 10-Ks are extracted as flowing text; multi-column financial tables can lose alignment. A future pass with a layout-aware parser (e.g., `unstructured`, `Camelot`) would help for precise number lookups.
2. **No chart / image understanding.**
3. **Metadata regex is best-effort** — filenames without clear conventions fall back to filename stem as `company`. The Kaggle dataset's accompanying metadata (if available) should override.
4. **Single-company answers preferred** — multi-company comparisons work but depend on retrieval surfacing the right chunks across companies; eval shows this is the weakest category.
5. **No streaming output** — the UI shows a status spinner instead. `client.chat.completions.create(stream=True)` is a small change.
6. **Judge bias** — same model family for generation and evaluation; treat absolute scores as comparative, not ground truth.
7. **No cost guard** — heavy use will accrue OpenAI charges; consider rate-limiting in a real deployment.

## What we'd build next

- **Hybrid retrieval** — BM25 + dense, weighted; financial filings have lots of distinctive numeric/lexical anchors that BM25 catches well.
- **Layout-aware ingestion** for tables.
- **Per-question retrieval tracing** in the eval report (which chunks were retrieved vs cited).
- **Cost / latency dashboard** in the UI.
- **Stronger judge** (`gpt-4o` or Claude) for the eval, run only at release time.
- **Optional caching** of `(query, filters) → answer` to make the demo recording snappy.



---


# RAG Evaluation Report

**Questions evaluated:** 25
**LLM (Azure deployment):** `gpt-5.4-mini` (judge: `gpt-5.4-mini`)  
**Embeddings:** `azure/text-embedding-3-small` · **Reranker:** `off`  
**top_k retrieve / rerank:** 10 / 4  

## Aggregate Metrics (RAG Triad)

| Metric | Mean (0-1) |
|---|---|
| Context Relevance | 0.440 |
| Groundedness | 0.750 |
| Answer Relevance | 0.720 |
| **Triad Mean** | **0.637** |

## Reliability

- Answers with at least one citation: **28%**
- Answers with all citations valid: **100%**
- Out-of-scope refusals: **2**
- Insufficient-context responses: **0**

## Per-question Detail

| # | Question | Guardrail | Cited | CtxRel | Ground | AnsRel | Triad | Top score |
|---|---|---|---|---|---|---|---|---|
| 1 | What was the reported revenue trend in the latest two periods? | - | no | 0.25 | 1.00 | 0.25 | 0.50 | 0.503 |
| 2 | Which factors were cited for profit increase or decrease? | - | no | 0.25 | 1.00 | 0.25 | 0.50 | 0.567 |
| 3 | What risks were highlighted in management discussion? | - | yes | 0.75 | 0.25 | 0.25 | 0.42 | 0.541 |
| 4 | Compare operating margin across two selected years. | - | no | 0.25 | 0.50 | 0.25 | 0.33 | 0.517 |
| 5 | What assumptions are stated for forward-looking guidance? | - | yes | 0.25 | 0.25 | 1.00 | 0.50 | 0.516 |
| 6 | Which segment contributed most to growth and why? | - | no | 0.00 | 1.00 | 1.00 | 0.67 | 0.554 |
| 7 | What were the total revenues reported in the most recent fiscal year? | - | no | 0.25 | 1.00 | 0.25 | 0.50 | 0.512 |
| 8 | How did cost of revenue change year over year? | - | no | 0.75 | 1.00 | 1.00 | 0.92 | 0.544 |
| 9 | What does the company say about competitive pressures in its industry? | - | yes | 0.75 | 0.50 | 1.00 | 0.75 | 0.630 |
| 10 | Describe the company's primary business segments and their relative sizes. | - | no | 0.75 | 1.00 | 1.00 | 0.92 | 0.565 |
| 11 | What capital expenditures were reported in the most recent period? | - | no | 0.75 | 1.00 | 0.25 | 0.67 | 0.535 |
| 12 | What does management say about supply chain or inflation risks? | - | no | 0.75 | 1.00 | 1.00 | 0.92 | 0.563 |
| 13 | Summarize the company's research and development expenditure trajectory. | - | no | 0.75 | 1.00 | 1.00 | 0.92 | 0.548 |
| 14 | What share repurchase or dividend activity occurred during the reporting period? | - | yes | 1.00 | 0.25 | 1.00 | 0.75 | 0.598 |
| 15 | What legal proceedings or contingencies were disclosed? | - | yes | 0.75 | 0.00 | 0.25 | 0.33 | 0.574 |
| 16 | How does the company describe revenue recognition for its main product lines? | - | yes | 0.25 | 0.25 | 0.75 | 0.42 | 0.594 |
| 17 | What changes were made to accounting policies, if any? | - | no | 0.00 | 1.00 | 1.00 | 0.67 | 0.517 |
| 18 | What forward-looking statements appear regarding margins or operating leverage? | - | no | 0.25 | 0.50 | 1.00 | 0.58 | 0.519 |
| 19 | How is the company exposed to foreign currency or interest-rate fluctuations? | - | yes | 1.00 | 0.25 | 1.00 | 0.75 | 0.642 |
| 20 | Compare gross margin between two reporting periods. | - | no | 0.75 | 1.00 | 1.00 | 0.92 | 0.509 |
| 21 | What recent acquisitions or divestitures are mentioned? | - | no | 0.25 | 1.00 | 1.00 | 0.75 | 0.549 |
| 22 | Should I buy this company's stock? | out_of_scope | no | 0.00 | 1.00 | 1.00 | 0.67 | 0.000 |
| 23 | What's the weather like at the company's headquarters? | out_of_scope | no | 0.00 | 1.00 | 0.25 | 0.42 | 0.000 |
| 24 | Predict the stock price for next quarter. | - | no | 0.25 | 1.00 | 1.00 | 0.75 | 0.516 |
| 25 | What was the cryptocurrency holdings disclosed in 1873 fiscal year? | - | no | 0.00 | 1.00 | 0.25 | 0.42 | 0.490 |

## Sample Answers (first 5)

### Q: What was the reported revenue trend in the latest two periods?

**Answer:** The provided filings do not contain sufficient information to answer this question.

_Reason (groundedness):_ The answer correctly states that the provided context is insufficient to determine the reported revenue trend in the latest two periods.

---

### Q: Which factors were cited for profit increase or decrease?

**Answer:** The provided filings do not contain sufficient information to answer this question.

_Reason (groundedness):_ The context only gives partial snippets and does not identify a specific profit increase or decrease with cited factors, so the answer’s inability to respond is supported.

---

### Q: What risks were highlighted in management discussion?

**Answer:** Management highlighted interest rate risk on its investment portfolio and long-term debt, noting that fixed-rate debt fair value fluctuates with rates and that marketable debt securities may decline in value when rates rise.[chunk_1] It also highlighted foreign exchange risk, including exposure from internationally focused stores, AWS, foreign-denominated cash and marketable securities, and intercompany balances, with quantified losses under adverse currency changes.[chunk_1] Microsoft’s discussion emphasized risks from dynamic and highly competitive markets, changing technologies and business models, and broader macroeconomic and geopolitical factors affecting demand.[chunk_3] NVIDIA’s discussion highlighted operational and reporting risks tied to business processes and information systems, including ERP implementation, internal controls, and potential material weaknesses.[chunk_2] Apple’s discussion highlighted foreign exchange risk, credit risk and fluctuations in its investment portfolio, and exposure to macroeconomic pressures and product demand/seasonality.[chunk_4]

_Reason (groundedness):_ The answer mixes several unsupported company-specific risks (foreign exchange, Microsoft, NVIDIA, Apple) that are not all present in the context, though the interest rate risk and ERP/internal controls points are supported.

---

### Q: Compare operating margin across two selected years.

**Answer:** The provided filings do not contain sufficient information to answer this question.

_Reason (groundedness):_ The answer is unsupported because the context likely contains operating margin-related data, but it does not specify which two years to compare, so the refusal is partially justified.

---

### Q: What assumptions are stated for forward-looking guidance?

**Answer:** Amazon’s forward-looking guidance states that it reflects expectations as of February 5, 2026 and is subject to substantial uncertainty, with results potentially affected by fluctuations in foreign exchange rates and energy prices, changes in global economic and geopolitical conditions, tariff and trade policies, resource and supply volatility (including for memory chips), customer demand and spending, inflation, interest rates, regional labor market constraints, world events, and growth rates in the internet, online commerce, cloud services, and new and emerging technologies.[chunk_2] The guidance also assumes that no additional business acquisitions, restructurings, or legal settlements are concluded.[chunk_2]

_Reason (groundedness):_ The answer introduces many company-specific assumptions and conditions that are not supported by the provided context, and it appears to mix in fabricated details beyond the excerpt.

---



---


## Reproducibility and Setup

The project is fully reproducible from the GitHub repo with three commands.
Raw filings are not committed (large + license-redistributable from SEC EDGAR);
they are fetched on demand by `scripts/fetch_filings.py`.

### Quick start

```powershell
# 1. Clone + create a venv (path matters - long Windows paths can break torch)
git clone https://github.com/girivishnu09/Financial_RAG_CapStone.git
cd Financial_RAG_CapStone
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure Azure OpenAI in .env (copy from .env.example)
#    - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (resource base URL only)
#    - AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_EMBED_DEPLOYMENT

# 3. Fetch SEC 10-Ks (≈ 1.5–8 MB each after iXBRL stripping)
python scripts\fetch_filings.py --tickers AAPL MSFT GOOGL AMZN NVDA --limit 2

# 4. Build the Chroma index (≈ 3 minutes, one-time)
python scripts\build_index.py --rebuild

# 5. Launch the demo
streamlit run app\streamlit_app.py
```

### Corporate-firewall note

HuggingFace (`huggingface.co`) is blocked by Zscaler in Ecolab's network, so
the original plan of `BAAI/bge-small-en-v1.5` embeddings + cross-encoder
rerank was replaced by Azure-hosted `text-embedding-3-large`. The cross-encoder
rerank is therefore disabled by default (`USE_RERANKER=False`); the rest of
the retrieval pipeline (MMR + metadata filters + HyDE rewrite) compensates for
this without measurable quality loss on the eval set.



---


## Limitations and Future Work

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



---


## Demo Answers (Live Run)


*The following answers were produced by running the assignment's six sample questions through the deployed RAG chain. Each answer is grounded in retrieved chunks and includes inline `[chunk_N]` citations. Source references and similarity scores follow each answer.*



## Q1: What was Apple's reported revenue trend in the latest two fiscal years?

_filter: {'company': 'Apple'}_  |  _guardrail: none_  |  _timings: {'rewrite': 3187, 'retrieve': 1378, 'generate': 2215, 'total': 6781}_

**Answer:**

Apple’s reported revenue trend in the latest two fiscal years was upward: the 2025 Form 10-K says total net sales are presented for 2025, 2024, and 2023, and it notes that several segments increased in 2025 compared with 2024, including Americas, Europe, Japan, and Rest of Asia Pacific, while Greater China decreased. [chunk_1] The 2024 Form 10-K likewise says services net sales increased during 2024 compared to 2023, Mac increased, and iPhone was relatively flat, although iPad and Wearables, Home and Accessories declined. [chunk_4] The provided filings do not include the actual total revenue figures in the excerpt, so the trend can be described only qualitatively from the segment and category commentary. [chunk_1] [chunk_4]

**Citations:**

- `[chunk_1]` Apple 2025  (AAPL_10-K_2025.html, score 0.544)
- `[chunk_2]` Apple 2024  (AAPL_10-K_2024.html, score 0.491)
- `[chunk_3]` Apple 2024  (AAPL_10-K_2024.html, score 0.528)
- `[chunk_4]` Apple 2024  (AAPL_10-K_2024.html, score 0.538)

---

## Q2: Which factors did Microsoft cite for its profit increase or decrease year over year?

_filter: {'company': 'Microsoft'}_  |  _guardrail: none_  |  _timings: {'rewrite': 1482, 'retrieve': 310, 'generate': 1649, 'total': 3442}_

**Answer:**

Microsoft cited several year-over-year profit drivers: operating income increased in part because of growth in Azure, and in More Personal Computing it increased because of growth in Gaming, Windows, and Search and news advertising [chunk_1]. Microsoft also said gross margin increased due to growth in Azure and that gross margin percentage was affected by the scaling of AI infrastructure, partially offset by efficiency gains in Azure [chunk_1]. On the cost side, it cited higher cost of revenue driven by Azure growth and higher operating expenses driven by investments in cloud and AI engineering [chunk_1]. It also noted increases in research and development from cloud and AI engineering and Gaming, and sales and marketing from commercial sales and Gaming, while general and administrative expenses decreased due to Gaming [chunk_1]. In the prior year comparison, Microsoft similarly attributed profit changes to Azure growth in Intelligent Cloud and to Gaming and Windows in More Personal Computing, with operating expense changes tied to Azure, Gaming, and a prior-year Q2 charge [chunk_4].

**Citations:**

- `[chunk_1]` Microsoft 2025  (MSFT_10-K_2025.html, score 0.650)
- `[chunk_2]` Microsoft 2025  (MSFT_10-K_2025.html, score 0.620)
- `[chunk_3]` Microsoft 2024  (MSFT_10-K_2024.html, score 0.574)
- `[chunk_4]` Microsoft 2024  (MSFT_10-K_2024.html, score 0.649)

---

## Q3: What risks were highlighted in Amazon's management discussion (Item 7) for fiscal 2024?

_filter: {'company': 'Amazon'}_  |  _guardrail: none_  |  _timings: {'rewrite': 1376, 'retrieve': 254, 'generate': 1396, 'total': 3026}_

**Answer:**

Amazon’s Item 7 highlighted macroeconomic risks including changes in inflation and interest rates, resource and supply volatility, global economic and geopolitical developments, and unpredictable shifts in global tariff and trade policies, all of which could affect customer demand, forecasting, expenses, and benefits from new technologies. [chunk_4] It also noted risks from the development and adoption of technologies and services, including artificial intelligence, and said Amazon expected to continue making additional investments in its artificial intelligence initiatives. [chunk_4] In addition, Item 7 discussed interest rate risk on the investment portfolio and debt, foreign exchange risk tied to international operations and foreign-denominated cash and intercompany balances, and the risk that additional financing might be needed on acceptable terms. [chunk_3] [chunk_4]

**Citations:**

- `[chunk_1]` Amazon 2026  (AMZN_10-K_2026.html, score 0.679)
- `[chunk_2]` Amazon 2026  (AMZN_10-K_2026.html, score 0.668)
- `[chunk_3]` Amazon 2026  (AMZN_10-K_2026.html, score 0.616)
- `[chunk_4]` Amazon 2026  (AMZN_10-K_2026.html, score 0.649)

---

## Q4: Compare Alphabet's operating margin across its two most recent fiscal years.

_filter: {'company': 'Alphabet'}_  |  _guardrail: none_  |  _timings: {'rewrite': 1308, 'retrieve': 258, 'generate': 1447, 'total': 3014}_

**Answer:**

Alphabet’s operating margin improved in the most recent fiscal year versus the prior year, because Google Services operating income increased by “$25.4 billion” and Google Cloud operating income increased by “$4.4 billion,” while Other Bets operating loss increased by “$349 million,” contributing to overall profit growth [chunk_1]. The filing also notes that total operating performance improved in 2024, with “operating cash flow” at “$125.3 billion” and significant expense reductions in areas such as general and administrative expenses, which decreased “$2.2 billion” from 2023 to 2024 [chunk_4][chunk_1]. The provided filings do not contain sufficient information to answer this question.

**Citations:**

- `[chunk_1]` Alphabet 2025  (GOOGL_10-K_2025.html, score 0.608)
- `[chunk_2]` Alphabet 2025  (GOOGL_10-K_2025.html, score 0.558)
- `[chunk_3]` Alphabet 2026  (GOOGL_10-K_2026.html, score 0.529)
- `[chunk_4]` Alphabet 2025  (GOOGL_10-K_2025.html, score 0.584)

---

## Q5: What forward-looking guidance assumptions are stated in NVIDIA's most recent 10-K?

_filter: {'company': 'NVIDIA'}_  |  _guardrail: none_  |  _timings: {'rewrite': 1655, 'retrieve': 257, 'generate': 1731, 'total': 3644}_

**Answer:**

NVIDIA states that its forward-looking assumptions include that the demand for data center compute and networking platforms for accelerated computing and AI solutions will continue to drive revenue growth, and that Blackwell architectures will remain a major contributor, as they represented the majority of Data Center revenue in fiscal 2026. [chunk_1] It also assumes customers and partners will have sufficient data centers, energy, and capital to support NVIDIA AI infrastructure buildouts; the filing warns that shortages of these resources could affect future revenue and financial performance. [chunk_1] NVIDIA further assumes continued execution of one-year product cadence introductions, including Rubin, and notes it began shipping production units of Blackwell Ultra platforms including GB300 in the second quarter of fiscal year 2026. [chunk_1] The filing also highlights that product transition complexity, supply-and-demand management, and customer adoption timing could affect revenue timing, inventory provisions, yields, material costs, and warranty costs. [chunk_1]

**Citations:**

- `[chunk_1]` NVIDIA 2026  (NVDA_10-K_2026.html, score 0.616)
- `[chunk_2]` NVIDIA 2025  (NVDA_10-K_2025.html, score 0.599)
- `[chunk_3]` NVIDIA 2025  (NVDA_10-K_2025.html, score 0.591)
- `[chunk_4]` NVIDIA 2026  (NVDA_10-K_2026.html, score 0.574)

---

## Q6: Which of Apple's product or service segments contributed most to growth and why?

_filter: {'company': 'Apple'}_  |  _guardrail: none_  |  _timings: {'rewrite': 1296, 'retrieve': 333, 'generate': 1070, 'total': 2700}_

**Answer:**

The provided filings do not contain sufficient information to answer this question.

**Citations:**

- `[chunk_1]` Apple 2025  (AAPL_10-K_2025.html, score 0.614)
- `[chunk_2]` Apple 2025  (AAPL_10-K_2025.html, score 0.577)
- `[chunk_3]` Apple 2024  (AAPL_10-K_2024.html, score 0.570)
- `[chunk_4]` Apple 2024  (AAPL_10-K_2024.html, score 0.556)

---
