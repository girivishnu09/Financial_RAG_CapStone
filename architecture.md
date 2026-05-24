# Architecture Note

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
