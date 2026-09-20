"""Pure metrics for the per-system benchmark table.

Everything here is counted from result files except one column, wall-clock time.
No full run recorded its own duration -- the timing fields reached `supervise.py`
after the last full arm had finished -- so time is composed from component speeds
that *were* measured (`reports/latency_benchmark.json`, and the llama-server log for
GLM). That makes it an estimate, and the benchmark labels it as one everywhere.
"""

from __future__ import annotations


def estimated_seconds(
    generated_tokens: float,
    judge_calls: float,
    solver_tokens_per_second: float,
    judge_seconds: float,
) -> float:
    """Local writing time plus checker waiting time.

    The solver speed was measured as total generated tokens over total wall clock
    for complete answers, so prompt reading for a short zero-shot prompt is already
    folded into it. A long prompt -- the 8-example baseline -- is *not* covered, and
    its estimate is a floor.
    """
    if solver_tokens_per_second <= 0:
        raise ValueError(
            f"solver speed must be positive, got {solver_tokens_per_second}; "
            "an unmeasured speed must not be turned into a time"
        )
    return generated_tokens / solver_tokens_per_second + judge_calls * judge_seconds


def flips(before: list[bool], after: list[bool]) -> tuple[int, int]:
    """(fixed, broken): answers turned right, and answers turned wrong."""
    if len(before) != len(after):
        raise ValueError(f"runs differ in length: {len(before)} vs {len(after)}")
    fixed = sum(1 for b, a in zip(before, after) if not b and a)
    broken = sum(1 for b, a in zip(before, after) if b and not a)
    return fixed, broken


def judge_quality(accepted: list[bool | None], correct: list[bool]) -> dict:
    """Recall, false-reject rate and precision of a checker's verdicts.

    A rejection is the checker calling an answer wrong. `None` means the checker
    was never asked about that answer, and is excluded rather than counted as
    either verdict.
    """
    if len(accepted) != len(correct):
        raise ValueError(f"lengths differ: {len(accepted)} vs {len(correct)}")
    pairs = [(a, c) for a, c in zip(accepted, correct) if a is not None]
    wrong = [a for a, c in pairs if not c]
    right = [a for a, c in pairs if c]
    rejected_wrong = sum(1 for a in wrong if not a)
    rejected_right = sum(1 for a in right if not a)
    rejected = rejected_wrong + rejected_right
    return {
        "judged": len(pairs),
        "recall": rejected_wrong / len(wrong) if wrong else 0.0,
        "false_reject_rate": rejected_right / len(right) if right else 0.0,
        "precision": rejected_wrong / rejected if rejected else None,
    }
