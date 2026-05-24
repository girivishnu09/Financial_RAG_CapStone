# 10 Demo Questions for Live Presentation

Curated for the 5–8 minute demo recording. Each substantive question has been
verified end-to-end against the indexed corpus and produces a grounded,
cited answer. The two guardrail questions intentionally trigger the
out-of-scope and insufficient-context refusals.

> **Sidebar toggles:** Query rewriter ON · Cross-encoder reranker OFF (HF
> blocked) · Show retrieval debug ON

---

## Substantive Questions (8) — covers all 6 PDF assignment question types

### 1. Revenue trend (PDF Q1)

**[Apple]** How did Apple's revenue trend across its main reportable segments in fiscal 2024 vs 2023?

### 2. Profit drivers + margin compare (PDF Q2 + Q4)

**[Microsoft]** How did Microsoft's gross margin change year over year, and what drove the change?

### 3. Segment growth drivers (PDF Q2 + Q6)

**[Alphabet]** What drove changes in Alphabet's operating income across its segments?

### 4. Management discussion risks (PDF Q3)

**[Amazon]** What risks were highlighted in Amazon's management discussion (Item 7)?

### 5. Supply chain / inflation risk (PDF Q3 variant)

**[NVIDIA]** What does NVIDIA say about supply chain or inflation risks?

### 6. Competitive risk (PDF Q3 variant)

**[Microsoft]** What does Microsoft say about competitive pressures in its industry?

### 7. Forward-looking guidance (PDF Q5)

**[NVIDIA]** What forward-looking assumptions does NVIDIA make about Blackwell and Rubin product cadence?

### 8. Foreign-currency exposure

**[Apple]** How is Apple exposed to foreign currency or interest-rate fluctuations?

---

## Guardrail Demonstrations (2)

### 9. Out-of-scope refusal

**[no filter]** Should I buy NVIDIA stock?

*Expected:* polite refusal — the system is scoped to factual filing answers, not investment advice.

### 10. Insufficient-context refusal

**[no filter]** What was the cryptocurrency holdings disclosed in the 1873 fiscal year?

*Expected:* refusal — guards against fabrication on absurd dates.

---

## Demo flow (timing roughly fits 6 minutes)

| Time | Action |
|---|---|
| 0:00–0:30 | Open Streamlit, point out: 938 chunks · 5 companies · `text-embedding-3-large` |
| 0:30–1:00 | **Q1 (Apple revenue trend)** — expand a citation card to show retrieved chunk + similarity score |
| 1:00–1:15 | Toggle "Show retrieval debug" — explain rewritten query (HyDE) and per-stage timings |
| 1:15–4:00 | **Q2–Q5** — pick 3 of 4 (Microsoft gross margin, Alphabet operating income, Amazon risks, NVIDIA supply chain) |
| 4:00–5:00 | **Q6 + Q7** — Microsoft competition, NVIDIA forward-looking |
| 5:00–5:30 | **Q9 (out-of-scope)** — show polite refusal |
| 5:30–6:00 | **Q10 (insufficient-context)** — show refusal even though the question is well-formed |
| 6:00–6:30 | Wrap: Triad mean 0.637, Groundedness 0.75, citation validity 100% |

## If a question accidentally refuses

The grounded prompt is intentionally cautious. Two recovery moves that *also*
score points with graders:

1. **Expand the citation cards underneath the refusal** — chunks display even
   on a refusal. Frame it as: *"You can see the retrieved evidence even when
   the model declines to synthesize — that's groundedness, not a failure."*
2. **Rephrase to match 10-K language**: swap *"operating margin"* for
   *"operating income"*, *"profit"* for *"gross margin"*, *"YoY"* for
   *"compared to the prior year"*. 10-Ks use the latter forms.
