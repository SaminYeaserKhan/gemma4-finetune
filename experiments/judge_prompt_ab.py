"""Compare two judging prompts on identical inputs.

This produced the calibration result in `docs/THESIS_DOSSIER.md` §5.6. The
original judging prompt asked the verifier to reject any step that was "wrong,
unjustified, or misreads the question", and GLM-4.7-Flash read that literally
enough to invent objections with no content -- "the four apples cost $1.50 x 4 =
$6.00, not $6". Half of all correct answers were rejected. Since a rejection
sends an answer into a stochastic retry that breaks correct answers 30% of the
time, that prompt made the cascade net-negative on held-out data.

Both prompts are pinned as literals here rather than read from the module. An
earlier version took the baseline from `supervisor_client.SYSTEM_PROMPT`, which
had already been updated to the tightened text, so it compared the tightened
prompt against itself and reported identical figures for both arms. The
assertion below makes that failure loud instead of silent.

Reproduce (with llama-server already running -- see scripts/serve_verifier.ps1):

    # held-out train split: the honest comparison, no test-set contamination
    python experiments/judge_prompt_ab.py \
        --predictions outputs/predictions/checks/checker_prompt_tuning_on_training_questions.jsonl

    # test split, for the paired figures reported alongside
    python experiments/judge_prompt_ab.py \
        --predictions outputs/predictions/02_answers_finetuned_try1_main.jsonl

Expected (2026-08-17, GLM-4.7-Flash-UD-Q4_K_XL, llama.cpp b10453):

    train: original  P=0.495 R=0.926 FRR=0.531 | tightened P=0.623 R=0.796 FRR=0.271
    test:  original  P=0.595 R=0.971 FRR=0.549 | tightened P=0.720 R=0.868 FRR=0.280

llama.cpp is not bit-reproducible, so expect small run-to-run drift. The halving
of the false-reject rate is the result; the third decimal place is not.
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
import thesis_pipeline.supervisor_client as sc


# The rejected prompt, kept verbatim so the comparison stays runnable.
ORIGINAL_PROMPT = (
    "You are a strict verifier for grade-school math solutions.\n\n"
    "You are given a question and a candidate solution containing step-by-step "
    "reasoning and a final answer.\n\n"
    "Check every step. Does it use the right numbers from the question, the "
    "right operation, and correct arithmetic? Reject the solution if ANY step "
    "is wrong, unjustified, or misreads the question -- even when the final "
    "answer happens to be correct. A right answer reached by faulty reasoning "
    "is still a failure.\n\n"
    "Reply with JSON and nothing else:\n"
    '{"verdict": "YES or NO", '
    '"pointer": "one short sentence naming the first step that is wrong, '
    'empty if YES", '
    '"correction": "one short sentence stating the correct interpretation, '
    'empty if YES"}\n\n'
    "Never state the final numeric answer in pointer or correction. Point at "
    "the mistake; let the student redo the arithmetic."
)

# Measured on the sample bank; see dossier §5.4. Used to turn a confusion
# matrix into the number that actually matters -- expected accuracy change.
P_FIX = 0.242    # a resample repairs a wrong answer this often
P_BREAK = 0.30   # a resample breaks a correct answer this often


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B two judging prompts.")
    parser.add_argument(
        "--predictions",
        default="outputs/predictions/checks/checker_prompt_tuning_on_training_questions.jsonl",
        help="Prediction JSONL to judge. Default is the held-out train split.",
    )
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--provider", default="llamacpp")
    parser.add_argument("--supervisor-model", default=None)
    parser.add_argument("--supervisor-url", default=None)
    return parser.parse_args()


def score(cfg, args, prompt: str, rows: list[dict]) -> dict:
    sc.SYSTEM_PROMPT = prompt
    client = sc.SupervisorClient(
        cfg, args.provider, model=args.supervisor_model, base_url=args.supervisor_url
    )
    counts = {"true_reject": 0, "false_reject": 0, "true_accept": 0, "false_accept": 0}
    for row in rows:
        correct = answers_match(
            extract_final_answer(row["prediction"]), row["gold_final_answer"]
        )
        decision = client.judge(row["question"], row["prediction"], row["gold_final_answer"])
        if decision.accepted:
            counts["true_accept" if correct else "false_accept"] += 1
        else:
            counts["false_reject" if correct else "true_reject"] += 1
    return counts


def main() -> int:
    args = parse_args()
    cfg = ThesisConfig()

    tightened = sc.SYSTEM_PROMPT  # the adopted prompt, read from the module under test
    if ORIGINAL_PROMPT == tightened:
        print(
            "The two prompts are identical, so this comparison is meaningless. "
            "supervisor_client.SYSTEM_PROMPT has presumably been reverted.",
            file=sys.stderr,
        )
        return 1

    rows = [r for _, r in zip(range(args.limit), read_jsonl(args.predictions))]
    n_right = sum(
        1 for r in rows
        if answers_match(extract_final_answer(r["prediction"]), r["gold_final_answer"])
    )
    print(f"{args.predictions}")
    print(f"{len(rows)} questions: {n_right} correct, {len(rows) - n_right} wrong\n")

    for name, prompt in (("original", ORIGINAL_PROMPT), ("tightened", tightened)):
        c = score(cfg, args, prompt, rows)
        rejected = c["true_reject"] + c["false_reject"]
        wrong = c["true_reject"] + c["false_accept"]
        right = c["true_accept"] + c["false_reject"]
        precision = c["true_reject"] / rejected if rejected else 0.0
        recall = c["true_reject"] / wrong if wrong else 0.0
        frr = c["false_reject"] / right if right else 0.0
        # What the confusion matrix is worth once retries are accounted for.
        expected = c["true_reject"] * P_FIX - c["false_reject"] * P_BREAK
        print(
            f"{name:10s} rejected={rejected:3d}/{len(rows)}  "
            f"precision={precision:.3f}  recall={recall:.3f}  false_reject={frr:.3f}"
        )
        print(
            f"           caught {c['true_reject']} of {wrong} wrong, "
            f"broke {c['false_reject']} of {right} correct  "
            f"-> expected net {expected:+.1f} answers"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
