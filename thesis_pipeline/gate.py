"""On-device escalation gates.

The cascade only sends a question to the supervisor when the local model is
probably wrong about it. Everything in this module is a pure function over
signals the phone already has after generating an answer -- no network, no
gold answers, no extra model. That constraint is the point: a gate that needed
the correct answer to decide would be useless in deployment.

Higher score = less trustworthy = escalate sooner.

Two gates are implemented:

- `length_score` is free. Longer answers correlate with wrong answers
  (AUC 0.676 on the FYDP 2 fine-tuned predictions), because the model rambles
  when it is unsure. Costs nothing beyond the token count already recorded.
- `disagreement_score` is stronger but costs k local generations. Sample the
  model k times and measure how much the answers disagree with each other.
  On the target hardware -- old laptops, office desktops, phones -- generation
  is the dominant cost, so this multiplies the wall-clock time per question by
  k. That is the trade this module exists to quantify.

Thresholds are deliberately not baked in here. `gate_curve` sweeps every
escalation rate so the accuracy/cost trade-off is chosen during analysis.
"""

from __future__ import annotations

from collections.abc import Sequence

from .gsm8k import answers_match


def length_score(tokens_generated: int | None) -> float:
    """Free gate: output length as a proxy for uncertainty."""
    return float(tokens_generated or 0)


def confidence_score(logprob: float | None) -> float:
    """Free gate: the model's own log-probability for the answer it wrote.

    Generation already computes a probability for every token it emits; we
    simply stopped throwing it away. Scoring a stored answer needs one forward
    pass rather than a generation loop, so this gate costs the device almost
    nothing beyond what it already spent -- unlike `disagreement_score`, which
    needs k full generations and so takes k times as long to answer.

    Log-probability runs the opposite way to this module's convention (closer
    to 0 = more certain), so the transformation is a sign flip and nothing
    else. Rescaling would not change AUC or the escalation ranking, both of
    which are rank-based, and would hide the raw measurement.

    `None` means the row was never scored, which is escalated first. Reading a
    missing value as maximal confidence would quietly exclude it from the gate.
    """
    if logprob is None:
        return float("inf")
    return -float(logprob)


def cluster_answers(answers: Sequence[str | None]) -> list[list[int]]:
    """Group answer indices by numeric equivalence.

    Uses `answers_match` rather than string equality so that `72`, `72.0` and
    `$72` count as agreement -- the same normalisation the thesis metric uses.
    """
    clusters: list[list[int]] = []
    for idx, answer in enumerate(answers):
        for cluster in clusters:
            if answers_match(answer, answers[cluster[0]]):
                cluster.append(idx)
                break
        else:
            clusters.append([idx])
    return clusters


def majority_answer(answers: Sequence[str | None]) -> str | None:
    """Self-consistency vote: the answer from the largest agreeing cluster.

    Ties break toward the earliest sample, which is attempt 1 (greedy decoding)
    when callers pass the sample bank in order.
    """
    if not answers:
        return None
    clusters = cluster_answers(answers)
    largest = max(clusters, key=len)
    return answers[largest[0]]


def disagreement_score(answers: Sequence[str | None]) -> float:
    """Stronger gate: fraction of samples outside the modal answer.

    0.0 means all k samples agree, approaching 1.0 means they all differ.
    """
    if not answers:
        return 1.0
    clusters = cluster_answers(answers)
    largest = max(len(cluster) for cluster in clusters)
    return 1.0 - largest / len(answers)


def tie_broken_score(
    primary: Sequence[float], secondary: Sequence[float]
) -> list[float]:
    """Rank by `primary`, break ties with `secondary`. Higher still escalates first.

    `disagreement_score` over k samples takes only k+1 distinct values -- with
    k=3, exactly three -- so it cannot rank inside its own groups. On the test
    split those groups hold 427 / 400 / 492 questions, which is why a 30%
    escalation rate has to cut one of them arbitrarily. Confidence is a
    continuous signal that is weaker on its own (AUC 0.715 against 0.840) but
    still informative inside a group (0.663 among the all-differ questions),
    so it is exactly what those ties want.

    Measured: 0.840 -> 0.869 AUC, and 105 -> 125 errors caught at a 10%
    escalation rate. Letting confidence *outweigh* agreement instead scores
    worse than agreement alone, so the combination is strictly lexicographic:
    the secondary is compressed to fit inside the smallest gap between
    distinct primary values and can never reorder them. There is no weight to
    tune, which also means there is none to have tuned on the test set.

    The secondary is used by rank, not by value, so its scale is irrelevant
    and infinities are ordered rather than propagated -- an unscored row
    (`confidence_score(None)`) sorts to the front of its own group instead of
    poisoning the arithmetic.
    """
    if len(primary) != len(secondary):
        raise ValueError(
            f"primary and secondary must be the same length, got "
            f"{len(primary)} and {len(secondary)}"
        )
    if not primary:
        return []

    # Equal secondary values share one averaged rank. Handing them
    # consecutive ranks instead would invent an ordering out of a signal that
    # does not distinguish them -- which matters most for the rows that share
    # `inf` because they were never scored.
    order = sorted(range(len(secondary)), key=lambda i: secondary[i])
    rank = [0.0] * len(secondary)
    position = 0
    while position < len(order):
        end = position
        while end < len(order) and secondary[order[end]] == secondary[order[position]]:
            end += 1
        shared = (position + end - 1) / 2 / len(secondary)  # in [0, 1)
        for index in order[position:end]:
            rank[index] = shared
        position = end

    distinct = sorted(set(primary))
    gaps = [b - a for a, b in zip(distinct, distinct[1:])]
    # No gap to respect when every primary value is identical: the secondary
    # is then the only signal there is, and should decide outright.
    room = min(gaps) if gaps else 1.0
    return [p + room * r for p, r in zip(primary, rank)]


def auc(scores: Sequence[float], is_wrong: Sequence[bool]) -> float:
    """Probability the gate ranks a wrong answer above a correct one.

    0.5 is a coin flip, 1.0 is a perfect gate. Rank-based (Mann-Whitney) with
    ties averaged, so the many answers sharing a token count are handled.
    """
    n_wrong = sum(1 for flag in is_wrong if flag)
    n_right = len(scores) - n_wrong
    if not n_wrong or not n_right:
        return float("nan")

    ordered = sorted(zip(scores, is_wrong), key=lambda pair: pair[0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = (i + j - 1) / 2 + 1  # ranks are 1-indexed
        rank_sum += sum(average_rank for k in range(i, j) if ordered[k][1])
        i = j
    return (rank_sum - n_wrong * (n_wrong + 1) / 2) / (n_wrong * n_right)


def escalation_order(scores: Sequence[float]) -> list[int]:
    """Indices from least to most trustworthy -- escalate from the front."""
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def select_for_escalation(scores: Sequence[float], escalation_rate: float) -> set[int]:
    """The worst-scoring `escalation_rate` fraction of examples."""
    if not 0.0 <= escalation_rate <= 1.0:
        raise ValueError(f"escalation_rate must be in [0, 1], got {escalation_rate}")
    count = round(len(scores) * escalation_rate)
    return set(escalation_order(scores)[:count])


def gate_curve(
    scores: Sequence[float],
    is_wrong: Sequence[bool],
    rates: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0),
) -> list[dict]:
    """Sweep escalation rates.

    `ceiling_accuracy` is the accuracy this gate would reach if every escalated
    error were fixed -- an upper bound on the whole cascade at that rate, which
    no supervisor or retry strategy can beat.
    """
    if len(scores) != len(is_wrong):
        raise ValueError("scores and is_wrong must be the same length")
    total = len(scores)
    total_wrong = sum(1 for flag in is_wrong if flag)
    rows = []
    for rate in rates:
        escalated = select_for_escalation(scores, rate)
        caught = sum(1 for i in escalated if is_wrong[i])
        retained = [i for i in range(total) if i not in escalated]
        retained_right = sum(1 for i in retained if not is_wrong[i])
        rows.append(
            {
                "escalation_rate": rate,
                "escalated": len(escalated),
                "errors_caught": caught,
                "recall": caught / total_wrong if total_wrong else float("nan"),
                "precision": caught / len(escalated) if escalated else float("nan"),
                "retained_accuracy": retained_right / len(retained) if retained else float("nan"),
                "ceiling_accuracy": (total - total_wrong + caught) / total if total else float("nan"),
            }
        )
    return rows
