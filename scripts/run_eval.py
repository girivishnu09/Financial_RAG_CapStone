"""Run the full evaluation suite and write eval/eval_report.md."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.evaluation import run_evaluation, write_report, load_eval_questions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(settings.EVAL_DIR / "test_questions.jsonl"))
    parser.add_argument("--out", default=str(settings.EVAL_DIR / "eval_report.md"))
    parser.add_argument("--rows-out", default=str(settings.EVAL_DIR / "eval_rows.jsonl"),
                        help="Optional JSONL of per-question rows for further analysis")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N questions (0 = all)")
    args = parser.parse_args()

    questions = load_eval_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]
    print(f"Evaluating {len(questions)} questions ...")

    rows = run_evaluation(questions)
    write_report(rows, args.out)
    print(f"Wrote {args.out}")

    if args.rows_out:
        with open(args.rows_out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
        print(f"Wrote {args.rows_out}")


if __name__ == "__main__":
    main()
