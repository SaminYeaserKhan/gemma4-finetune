"""Turn cascade runs into the FYDP 3 tables.

Everything reported here is derived offline from stored artifacts, so the
expensive GPU and API passes happen once and every threshold, gate and
feedback level is re-analysed for free.

    python analyze_supervision.py \
        --baseline outputs/predictions/02_answers_finetuned_try1_main.jsonl \
        --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl \
        --sample-bank outputs/predictions/04_answers_finetuned_try3.jsonl \
        --cascade L0=outputs/predictions/cascade_gemini_disagreement_L0_test.jsonl \
        --cascade L1=outputs/predictions/cascade_gemini_disagreement_L1_test.jsonl \
        --cascade L2=outputs/predictions/cascade_gemini_disagreement_L2_test.jsonl

Sections written:
  1. gate quality        -- can the phone tell when it is wrong, for free?
  2. sample-bank arms    -- pass@3 ceiling, self-consistency, blind retry
  3. cascade arms        -- the L0/L1/L2 comparison, with cost
  4. verifier quality    -- confusion matrix against GSM8K ground truth
  5. flip analysis       -- did retrying break answers that were already right?
  6. significance        -- McNemar against the un-supervised baseline
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from thesis_pipeline.experiment_log import append_experiment_event, command_string
from thesis_pipeline.gate import (
    auc,
    confidence_score,
    disagreement_score,
    gate_curve,
    length_score,
    majority_answer,
    tie_broken_score,
)
from thesis_pipeline.gsm8k import answers_match, extract_final_answer
from thesis_pipeline.io_utils import read_jsonl, write_csv


def approx_tokens(text: str | None) -> int:
    """~4 characters per token, the standard BPE approximation for English.

    Used only for the counterfactual per-level costs: the supervisor is asked
    once for pointer and correction together, so the L0/L1/L2 output costs are
    reconstructed by measuring each field. `--validate-levels` on supervise.py
    cross-checks this against genuinely separate per-level calls.
    """
    return math.ceil(len(text) / 4) if text else 0


def load_rows(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))


def final_answer_of(row: dict) -> str | None:
    return (
        row.get("accepted_final_answer")
        or row.get("pred_final_answer")
        or extract_final_answer(row.get("accepted_answer") or row.get("prediction"))
    )


def gold_of(row: dict) -> str | None:
    return row.get("gold_final_answer") or extract_final_answer(row.get("gold_answer"))


def is_correct(row: dict) -> bool:
    return answers_match(final_answer_of(row), gold_of(row))


def load_sample_bank(paths: list[str]) -> dict[int, list[str | None]]:
    bank: dict[int, list[str | None]] = {}
    for path in paths:
        for row in read_jsonl(path):
            bank.setdefault(int(row["id"]), []).append(final_answer_of(row))
    return bank


# -- section 1: gate quality ------------------------------------------------


def load_confidence(path: str | Path | None) -> dict[int, dict]:
    if not path or not Path(path).exists():
        return {}
    return {int(row["id"]): row for row in read_jsonl(path)}


def gate_report(
    baseline: list[dict],
    bank: dict[int, list[str | None]],
    confidence: dict[int, dict] | None = None,
) -> list[dict]:
    """How well each on-device signal predicts that the local model is wrong.

    Ordered by what the signal costs the device, because that is the axis the
    thesis argues on: length is already recorded, confidence is one extra
    forward pass over text the model just wrote, disagreement is k more full
    generations. A weaker gate that costs nothing can still be the right choice
    on hardware where generation is the slow step.
    """
    wrong = [not is_correct(row) for row in baseline]
    reports = []

    lengths = [length_score(row.get("tokens_generated")) for row in baseline]
    reports.append(("length (free)", auc(lengths, wrong), gate_curve(lengths, wrong)))

    for field, label in (
        ("mean_logprob", "confidence: mean logprob (1 forward pass)"),
        ("min_logprob", "confidence: min logprob (1 forward pass)"),
        ("final_logprob", "confidence: final-answer logprob (1 forward pass)"),
    ):
        scores = [
            confidence_score((confidence or {}).get(int(row["id"]), {}).get(field))
            for row in baseline
        ]
        # All-infinite means the measure was never populated -- no score file,
        # or a column that is None throughout. An AUC over identical values is
        # 0.5, which would read as a real and uninformative measurement rather
        # than as missing data.
        if all(math.isinf(score) for score in scores):
            continue
        reports.append((label, auc(scores, wrong), gate_curve(scores, wrong)))

    if bank:
        scores = [
            disagreement_score([final_answer_of(row), *bank.get(int(row["id"]), [])])
            for row in baseline
        ]
        reports.append(
            ("disagreement (k samples)", auc(scores, wrong), gate_curve(scores, wrong))
        )

        if confidence:
            # Disagreement over k samples takes only k+1 distinct values, so it
            # cannot rank inside its own groups; confidence is continuous and
            # still informative there. Ranking by the first and breaking ties
            # with the second beats either alone, and costs nothing new -- both
            # signals were already being computed.
            certainty = [
                confidence_score(confidence.get(int(row["id"]), {}).get("final_logprob"))
                for row in baseline
            ]
            merged = tie_broken_score(scores, certainty)
            reports.append(
                (
                    "combined: disagreement, ties broken by confidence",
                    auc(merged, wrong),
                    gate_curve(merged, wrong),
                )
            )

    rows = []
    for name, gate_auc, curve in reports:
        for point in curve:
            rows.append({"gate": name, "auc": f"{gate_auc:.4f}", **{
                key: (f"{value:.4f}" if isinstance(value, float) else value)
                for key, value in point.items()
            }})
    return rows


# -- section 2: what the sample bank alone can do ---------------------------


def sample_bank_report(
    baseline: list[dict], bank: dict[int, list[str | None]]
) -> list[dict]:
    """Controls that need no supervisor at all.

    `pass@k` is the hard ceiling for the whole study: if the local model never
    produces the right answer in k samples, no verifier can find it.
    """
    if not bank:
        return []
    total = len(baseline)
    pass_at_k = consistency = blind = 0
    for row in baseline:
        gold = gold_of(row)
        samples = [final_answer_of(row), *bank.get(int(row["id"]), [])]
        if any(answers_match(sample, gold) for sample in samples):
            pass_at_k += 1
        if answers_match(majority_answer(samples), gold):
            consistency += 1
        if answers_match(samples[-1], gold):
            blind += 1
    k = 1 + max(len(v) for v in bank.values())
    return [
        _accuracy_row(f"local only (1 sample)", sum(is_correct(r) for r in baseline), total),
        _accuracy_row(f"blind retry, take last", blind, total),
        _accuracy_row(f"self-consistency@{k} (majority)", consistency, total),
        _accuracy_row(f"pass@{k} CEILING", pass_at_k, total),
    ]


def _accuracy_row(name: str, correct: int, total: int) -> dict:
    return {
        "condition": name,
        "correct": correct,
        "total": total,
        "accuracy": f"{correct / total:.4f}" if total else "",
        "cloud_tokens_per_q": "0.0",
    }


# -- section 3: cascade arms ------------------------------------------------


def cascade_report(name: str, rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if is_correct(row))
    escalated = [row for row in rows if row.get("escalated")]
    calls = sum(int(row.get("supervisor_calls") or 0) for row in rows)
    cloud_in = sum(int(row.get("supervisor_input_tokens") or 0) for row in rows)
    cloud_out = sum(int(row.get("supervisor_output_tokens") or 0) for row in rows)
    local_tokens = sum(
        int(row.get("total_model_tokens") or 0) + int(row.get("total_prompt_tokens") or 0)
        for row in rows
    )
    errors = sum(1 for row in rows if (row.get("attempts") or [{}])[0].get("supervisor_error"))
    return {
        "condition": name,
        "correct": correct,
        "total": total,
        "accuracy": f"{correct / total:.4f}" if total else "",
        "escalation_rate": f"{len(escalated) / total:.4f}" if total else "",
        "supervisor_calls": calls,
        "cloud_tokens_per_q": f"{(cloud_in + cloud_out) / total:.1f}" if total else "",
        "local_tokens_per_q": f"{local_tokens / total:.1f}" if total else "",
        "avg_attempts": f"{sum(int(r.get('attempt_count') or 1) for r in rows) / total:.2f}" if total else "",
        "supervisor_errors": errors,
    }


# -- section 3b: the stacked arm --------------------------------------------


def stack_voting(
    rows: list[dict], bank: dict[int, list[str | None]]
) -> list[dict]:
    """Re-score a cascade arm as "vote first, escalate only the split votes".

    The two repairs are aimed at different questions. Majority voting fixes
    answers where the model already knew better and greedy decoding picked the
    wrong branch; the supervisor fixes answers the model gets wrong every time
    it tries. Running each where it works means taking the vote on the
    questions the gate kept on-device and the cascade's answer on the ones it
    sent up.

    This costs the device nothing extra. The disagreement gate has *already*
    generated the k samples in order to decide what to escalate, so their
    majority is a number the device is holding either way -- the un-stacked arm
    simply throws it away. Supervisor calls are unchanged: the escalated set,
    and therefore every judgment, is exactly the arm's own.

    Returns new rows; the caller still reports the un-stacked arm from the
    originals.
    """
    if not bank:
        return rows
    stacked = []
    for row in rows:
        if row.get("escalated"):
            stacked.append(row)
            continue
        first = (row.get("attempts") or [{}])[0]
        samples = [
            first.get("pred_final_answer") or extract_final_answer(first.get("prediction")),
            *bank.get(int(row["id"]), []),
        ]
        voted = majority_answer(samples)
        stacked.append({
            **row,
            "accepted_final_answer": voted,
            "correct": answers_match(voted, gold_of(row)),
        })
    return stacked


def counterfactual_level_costs(rows: list[dict]) -> list[dict]:
    """What each feedback level would have cost, from the recorded hint text."""
    out = []
    for level, field in ((0, None), (1, "supervisor_pointer"), (2, "supervisor_correction")):
        tokens = 0
        for row in rows:
            for attempt in row.get("attempts") or []:
                if attempt.get("supervisor_accepted") is None:
                    continue
                verdict_tokens = 8  # {"verdict":"NO"} plus JSON scaffolding
                if level == 0:
                    tokens += verdict_tokens
                elif level == 1:
                    tokens += verdict_tokens + approx_tokens(attempt.get("supervisor_pointer"))
                else:
                    tokens += (
                        verdict_tokens
                        + approx_tokens(attempt.get("supervisor_pointer"))
                        + approx_tokens(attempt.get("supervisor_correction"))
                    )
        total = len(rows)
        out.append(
            {
                "feedback_level": f"L{level}",
                "cloud_output_tokens": tokens,
                "per_question": f"{tokens / total:.1f}" if total else "",
            }
        )
    return out


# -- section 4: is the verifier any good? -----------------------------------


def verifier_confusion(name: str, rows: list[dict]) -> dict:
    """Supervisor verdict on attempt 1 vs. GSM8K ground truth.

    A false rejection is the expensive kind of mistake: it sends an answer that
    was already correct into a stochastic retry that may break it.
    """
    tp = fp = tn = fn = 0  # positive = "supervisor rejects"
    for row in rows:
        attempts = row.get("attempts") or []
        if not attempts:
            continue
        first = attempts[0]
        accepted = first.get("supervisor_accepted")
        if accepted is None:  # never escalated, so never judged
            continue
        wrong = not first.get("exact_correct")
        if wrong and not accepted:
            tp += 1
        elif wrong and accepted:
            fn += 1
        elif not wrong and not accepted:
            fp += 1
        else:
            tn += 1
    judged = tp + fp + tn + fn
    return {
        "condition": name,
        "judged": judged,
        "true_reject": tp,
        "false_reject": fp,
        "true_accept": tn,
        "false_accept": fn,
        "precision": f"{tp / (tp + fp):.4f}" if (tp + fp) else "",
        "recall": f"{tp / (tp + fn):.4f}" if (tp + fn) else "",
        "false_reject_rate": f"{fp / (fp + tn):.4f}" if (fp + tn) else "",
        "accuracy": f"{(tp + tn) / judged:.4f}" if judged else "",
    }


# -- section 5: did retrying help or hurt? ----------------------------------


def flip_report(name: str, rows: list[dict]) -> dict:
    fixed = broken = held = still_wrong = 0
    for row in rows:
        attempts = row.get("attempts") or []
        if not attempts:
            continue
        before = bool(attempts[0].get("exact_correct"))
        after = is_correct(row)
        if not before and after:
            fixed += 1
        elif before and not after:
            broken += 1
        elif before:
            held += 1
        else:
            still_wrong += 1
    return {
        "condition": name,
        "wrong_to_right": fixed,
        "right_to_wrong": broken,
        "net": fixed - broken,
        "stayed_right": held,
        "stayed_wrong": still_wrong,
    }


# -- section 6: significance ------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs.

    b = baseline right, cascade wrong. c = baseline wrong, cascade right.
    Exact rather than chi-square because b + c is often small.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def significance_report(name: str, baseline: list[dict], rows: list[dict]) -> dict:
    baseline_by_id = {int(row["id"]): is_correct(row) for row in baseline}
    b = c = 0
    for row in rows:
        example_id = int(row["id"])
        if example_id not in baseline_by_id:
            continue
        before, after = baseline_by_id[example_id], is_correct(row)
        if before and not after:
            b += 1
        elif not before and after:
            c += 1
    p = mcnemar_exact(b, c)
    return {
        "condition": name,
        "baseline_only_right": b,
        "cascade_only_right": c,
        "p_value": f"{p:.5f}",
        "significant_at_0.05": "yes" if p < 0.05 else "no",
    }


# -- output -----------------------------------------------------------------


def normalize_rows(rows: list[dict]) -> list[dict]:
    """Pad rows to a common set of columns, in first-seen order.

    The headline table mixes shapes: sample-bank conditions have no supervisor
    columns, cascade arms do. Taking the columns from the first row alone would
    silently drop the cost data.
    """
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    return [{key: row.get(key, "") for key in headers} for row in rows]


def render_table(rows: list[dict]) -> list[str]:
    if not rows:
        return ["_no data_", ""]
    rows = normalize_rows(rows)
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    lines.append("")
    return lines


def parse_named(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path)
    path = Path(value)
    return path.stem, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze FYDP 3 cascade runs.")
    parser.add_argument("--baseline", default="outputs/predictions/02_answers_finetuned_try1_main.jsonl")
    parser.add_argument("--sample-bank", action="append", default=[])
    parser.add_argument(
        "--confidence",
        default="outputs/predictions/05_confidence_for_try1.jsonl",
        help="Log-probability scores from experiments/score_confidence.py. Ignored if absent.",
    )
    parser.add_argument("--cascade", action="append", default=[], help="name=path. Repeatable.")
    parser.add_argument("--csv-output", default="reports/fydp3_summary.csv")
    parser.add_argument("--md-output", default="reports/fydp3_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_rows(args.baseline)
    bank = load_sample_bank(args.sample_bank)
    confidence = load_confidence(args.confidence)
    cascades = [(name, load_rows(path)) for name, path in map(parse_named, args.cascade)]

    headline = sample_bank_report(baseline, bank)
    if not headline:
        headline = [
            _accuracy_row("local only (1 sample)", sum(is_correct(r) for r in baseline), len(baseline))
        ]
    # The stacked arm reuses the samples the gate already generated, so it
    # appears next to its parent arm at identical supervisor cost.
    stacked = (
        [(f"{name} + voting", stack_voting(rows, bank)) for name, rows in cascades]
        if bank
        else []
    )
    for index, (name, rows) in enumerate(cascades):
        headline.append(cascade_report(name, rows))
        if stacked:
            headline.append(cascade_report(*stacked[index]))

    lines = ["# FYDP 3 — Supervised Cascade Results", ""]
    lines += [
        f"Baseline: `{args.baseline}` — "
        f"{sum(is_correct(r) for r in baseline)}/{len(baseline)} correct.",
        "",
        "## 1. Headline: accuracy vs. cloud cost",
        "",
    ]
    lines += render_table(headline)

    lines += ["## 2. Gate quality (can the device tell when it is wrong?)", ""]
    lines += render_table(gate_report(baseline, bank, confidence))

    if cascades:
        lines += ["## 3. Verifier quality vs. GSM8K ground truth", ""]
        lines += render_table([verifier_confusion(n, r) for n, r in cascades])

        lines += ["## 4. Flip analysis (did retrying break correct answers?)", ""]
        lines += render_table([flip_report(n, r) for n, r in cascades])

        lines += ["## 5. Significance vs. the un-supervised baseline (exact McNemar)", ""]
        lines += render_table(
            [significance_report(n, baseline, r) for n, r in cascades + stacked]
        )

        if stacked:
            # The honest control is not the greedy baseline -- it is the free
            # majority vote, which needs no supervisor at all. An arm that
            # cannot beat it has not earned its cloud calls.
            vote_rows = stack_voting(
                [{**r, "escalated": False} for r in cascades[0][1]], bank
            )
            lines += [
                "## 5b. Significance vs. free self-consistency (the control that matters)",
                "",
            ]
            lines += render_table(
                [
                    significance_report(n, vote_rows, r)
                    for n, r in cascades + stacked
                ]
            )

        lines += ["## 6. Counterfactual cloud output cost per feedback level", ""]
        for name, rows in cascades:
            lines += [f"**{name}**", ""]
            lines += render_table(counterfactual_level_costs(rows))

    md_path = Path(args.md_output)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_csv(args.csv_output, normalize_rows(headline))

    append_experiment_event(
        "analysis",
        "Analyzed FYDP 3 cascade runs",
        {
            "command": command_string([sys.executable, *sys.argv]),
            "baseline": args.baseline,
            "sample_bank": args.sample_bank,
            "confidence": args.confidence if confidence else None,
            "cascades": [name for name, _ in cascades],
            "md_output": args.md_output,
            "csv_output": args.csv_output,
        },
    )
    for row in headline:
        print(f"{row['condition']}: {row.get('correct')}/{row.get('total')} acc={row.get('accuracy')}")
    print(f"Saved reports to {args.csv_output} and {args.md_output}")


if __name__ == "__main__":
    main()
