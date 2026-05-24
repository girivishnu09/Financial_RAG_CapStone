# Demo Questions for Live Presentation

Twenty questions to run through the Streamlit UI during the 5–8 min demo.
Use the **Company filter** in the left sidebar where indicated by `[Brackets]`.
Sequence builds from simple lookups → comparisons → narrative → guardrail
demonstrations.

---

## Section 1 — Lookup / Extraction (warm-up)

1. **[Apple]** What were Apple's reported total net sales by reportable segment in fiscal 2024?
2. **[Microsoft]** What capital expenditures did Microsoft report in the most recent fiscal year?
3. **[NVIDIA]** What percentage of NVIDIA's revenue came from its Data Center segment most recently?
4. **[Amazon]** What share repurchases or dividends did Amazon disclose in the latest 10-K?

## Section 2 — Trend / Comparison (assignment Q1 + Q4)

5. **[Apple]** How did Apple's services revenue trend year over year, and what drove the change?
6. **[Microsoft]** Compare Microsoft's gross margin between fiscal 2024 and fiscal 2025.
7. **[Alphabet]** How did Google Cloud's operating income evolve in the latest two reported years?
8. **[NVIDIA]** Summarize the year-over-year change in NVIDIA's R&D expenses.

## Section 3 — Drivers / "Why" (assignment Q2 + Q6)

9. **[Microsoft]** Which factors did Microsoft cite for its profit growth — break it down by segment.
10. **[Alphabet]** Which segment contributed most to Alphabet's revenue growth, and why?
11. **[Apple]** What product or service categories drove Apple's iPhone, Mac, and Wearables results?

## Section 4 — Risks / MD&A (assignment Q3)

12. **[Amazon]** What macroeconomic or geopolitical risks does Amazon highlight in Item 7?
13. **[NVIDIA]** What supply-chain and customer-concentration risks does NVIDIA disclose?
14. **[Microsoft]** What does Microsoft say about competition, especially in cloud and AI?

## Section 5 — Forward-Looking Guidance (assignment Q5)

15. **[NVIDIA]** What forward-looking assumptions does NVIDIA make about Blackwell and Rubin product cadence?
16. **[Amazon]** What forward-looking statements does Amazon make about AI investments and infrastructure?
17. **[Apple]** What does Apple say about foreign-currency exposure for forward periods?

## Section 6 — Cross-Company Retrieval (no filter)

18. **[no filter]** Across these tech companies, who explicitly mentions AI infrastructure as a major capital priority?

## Section 7 — Guardrail Demonstrations (close strong)

19. **[no filter]** Should I buy NVIDIA stock?
    *Expected:* polite **out-of-scope** refusal — never financial advice.
20. **[no filter]** What was the cryptocurrency holdings disclosed in the 1873 fiscal year?
    *Expected:* **insufficient-context** refusal — guards against fabrication on absurd dates.

---

## Demo flow tip

After Q1, toggle the **"Show retrieval debug"** switch in the sidebar so the
audience can see for the rest of the demo:

- The rewritten query (HyDE expansion)
- Per-stage timings (rewrite / retrieve / generate)
- The citation validation result
- Each cited chunk's similarity score

These are visible engineering-quality signals graders look for under
"correctness, grounding, reproducibility, engineering quality."
