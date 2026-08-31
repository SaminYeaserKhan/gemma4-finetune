"""Print the judge's reasoning on answers it rejected but that were correct.

A false-reject rate is a number; it does not tell you whether the judge is being
appropriately strict about reasoning or simply making things up. This prints the
cases so you can read them, and it is how the defective first judging prompt was
diagnosed (`docs/THESIS_DOSSIER.md` §5.6) -- the objections turned out to be
things like "the four apples cost $1.50 x 4 = $6.00, not $6", and pointers that
quoted the model's own correct line back as the error.

Run it whenever the false-reject rate moves, before trusting the number.

    python experiments/inspect_rejections.py --limit 30 --show 4

`--mode false-reject` (default) shows correct answers the judge rejected.
`--mode false-accept` shows wrong answers it waved through -- the other failure,
useful when recall drops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running `python experiments/foo.py` puts experiments/ on sys.path, not the
# repo root, so the package would not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.gsm8k import answers_match, extract_final_answer
from thesis_pipeline.io_utils import read_jsonl
from thesis_pipeline.supervisor_client import SupervisorClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read the judge's actual objections.")
    parser.add_argument("--predictions", default=None, help="Defaults to the attempt-1 cache.")
    parser.add_argument("--limit", type=int, default=30, help="Questions to judge.")
    parser.add_argument("--show", type=int, default=4, help="Cases to print.")
    parser.add_argument(
        "--mode", default="false-reject", choices=["false-reject", "false-accept"]
    )
    parser.add_argument("--provider", default="llamacpp")
    parser.add_argument("--supervisor-model", default=None)
    parser.add_argument("--supervisor-url", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ThesisConfig()
    client = SupervisorClient(
        cfg, args.provider, model=args.supervisor_model, base_url=args.supervisor_url
    )
    path = args.predictions or cfg.attempt1_cache
    rows = [r for _, r in zip(range(args.limit), read_jsonl(path))]

    want_correct = args.mode == "false-reject"
    shown = 0
    for row in rows:
        correct = answers_match(
            extract_final_answer(row["prediction"]), row["gold_final_answer"]
        )
        if correct != want_correct:
            continue
        decision = client.judge(row["question"], row["prediction"], row["gold_final_answer"])
        # false-reject = was correct and rejected; false-accept = wrong and accepted
        if decision.accepted != (not want_correct):
            continue

        shown += 1
        verdict = "NO" if want_correct else "YES"
        state = "CORRECT" if correct else "WRONG"
        print("=" * 78)
        print(f"id {row['id']}  gold={row['gold_final_answer']}  "
              f"(final answer was {state}, judge said {verdict})")
        print(f"Q: {row['question'][:220]}")
        print(f"MODEL: {row['prediction'][:400]}")
        print(f"POINTER:    {decision.pointer}")
        print(f"CORRECTION: {decision.correction}")
        if shown >= args.show:
            break

    if not shown:
        print(f"No {args.mode} cases in the first {len(rows)} questions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
