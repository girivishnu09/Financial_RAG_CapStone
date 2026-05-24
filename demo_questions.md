# Demo Questions for Live Presentation

Twenty questions for the 5–8 min Streamlit demo. Each one has been verified
against the indexed corpus — either through the live `demo_answers.md` run or
through the 25-question eval (Triad scores 0.67+).

Use the **Company filter** in the left sidebar where indicated by `[Brackets]`.

> **Toggles before you start:**
> - Query rewriter (HyDE): **ON**
> - Cross-encoder reranker: **OFF** (HuggingFace is firewall-blocked; will fail)
> - Show retrieval debug: **ON** (after Q1, to expose timings + rewritten query)

---

## Section 1 — Lookup / Extraction (warm-up, all proven)

1. **[Apple]** How did Apple's services net sales trend in fiscal 2024 compared to 2023?
2. **[Microsoft]** What capital expenditures did Microsoft report in the most recent fiscal year?
3. **[Apple]** What share repurchases or dividends did Apple disclose in the latest 10-K?
4. **[NVIDIA]** Summarize NVIDIA's research and development expenditure trajectory.

## Section 2 — Trend / Comparison (assignment Q1 + Q4 — all proven)

5. **[Apple]** How did Apple's revenue trend across its main reportable segments in fiscal 2024 vs 2023?
6. **[Microsoft]** How did Microsoft's gross margin change year over year, and what drove the change?
7. **[Amazon]** How did Amazon's cost of revenue change year over year?
8. **[NVIDIA]** Compare NVIDIA's R&D expenses across the latest two fiscal years.

## Section 3 — Drivers / "Why" (assignment Q2 + Q6 — all proven)

9. **[Microsoft]** Which factors did Microsoft cite for its profit growth — break it down by segment.
10. **[Alphabet]** What drove changes in Alphabet's operating income across its segments?
11. **[NVIDIA]** What forward-looking assumptions does NVIDIA make about Blackwell and Rubin product cadence?

## Section 4 — Risks / MD&A (assignment Q3 — all proven)

12. **[Amazon]** What risks were highlighted in Amazon's management discussion (Item 7)?
13. **[NVIDIA]** What does NVIDIA say about supply chain or inflation risks?
14. **[Microsoft]** What does Microsoft say about competitive pressures in its industry?

## Section 5 — Forward-Looking Guidance (assignment Q5 — all proven)

15. **[Amazon]** What forward-looking statements does Amazon make about AI investments and infrastructure?
16. **[Apple]** How is Apple exposed to foreign currency or interest-rate fluctuations?
17. **[Alphabet]** Describe Alphabet's primary business segments and their relative sizes.

## Section 6 — Cross-Document (still uses a company filter for retrieval quality)

18. **[Microsoft]** What recent acquisitions or divestitures does Microsoft mention in its filings?

## Section 7 — Guardrail Demonstrations (close strong)

19. **[no filter]** Should I buy NVIDIA stock?
    *Expected:* polite **out-of-scope** refusal — never financial advice.

20. **[no filter]** What was the cryptocurrency holdings disclosed in the 1873 fiscal year?
    *Expected:* **insufficient-context** refusal — guards against fabrication on absurd dates.

---

## If a question accidentally refuses

The grounded prompt is intentionally cautious. Two quick recoveries:

1. **Expand the citation cards under the refusal** — the chunks still display.
   This is a feature, not a failure: "you can see the retrieved evidence even
   when the model declines to synthesize."
2. **Rephrase to match 10-K language**, e.g. swap "operating margin" for
   "operating income" or "gross margin" — 10-Ks use the latter more often.

## Demo flow tip

After Q1, toggle **"Show retrieval debug"** so the audience sees, for the rest
of the demo:

- Rewritten query (HyDE expansion)
- Per-stage timings (rewrite / retrieve / generate)
- Citation validation result
- Each cited chunk's similarity score

These signal "engineering quality" — one of the four graded dimensions.
