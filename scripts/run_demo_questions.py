"""Run the 6 questions from the assignment PDF through the RAG chain
and write the results to demo_answers.md for review.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rag_chain import answer


PDF_QUESTIONS = [
    ("What was Apple's reported revenue trend in the latest two fiscal years?", {"company": "Apple"}),
    ("Which factors did Microsoft cite for its profit increase or decrease year over year?", {"company": "Microsoft"}),
    ("What risks were highlighted in Amazon's management discussion (Item 7) for fiscal 2024?", {"company": "Amazon"}),
    ("Compare Alphabet's operating margin across its two most recent fiscal years.", {"company": "Alphabet"}),
    ("What forward-looking guidance assumptions are stated in NVIDIA's most recent 10-K?", {"company": "NVIDIA"}),
    ("Which of Apple's product or service segments contributed most to growth and why?", {"company": "Apple"}),
]


OUT = Path(__file__).resolve().parents[1] / "demo_answers.md"


def main() -> None:
    lines = ["# Demo Answers (PDF assignment questions)\n"]
    for i, (q, filt) in enumerate(PDF_QUESTIONS, 1):
        print(f"[{i}/{len(PDF_QUESTIONS)}] {q}")
        try:
            resp = answer(q, filters=filt)
        except Exception as e:
            lines += [f"## Q{i}: {q}\n", f"**ERROR:** {e}\n"]
            continue
        lines.append(f"## Q{i}: {q}\n")
        lines.append(f"_filter: {filt}_  |  _guardrail: {resp.guardrail or 'none'}_  |  _timings: {resp.timings_ms}_\n")
        lines.append("**Answer:**\n")
        lines.append(resp.answer)
        lines.append("\n**Citations:**\n")
        if resp.citations:
            for c in resp.citations:
                lines.append(
                    f"- `[chunk_{c.index}]` {c.company or '?'} {c.year or ''} "
                    f"{c.filing_type if hasattr(c, 'filing_type') else ''} "
                    f"({c.source}, score {c.score:.3f})"
                )
        else:
            lines.append("- (no citations)")
        lines.append("\n---\n")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
