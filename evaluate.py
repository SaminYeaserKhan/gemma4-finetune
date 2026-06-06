from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from thesis_pipeline.experiment_log import append_experiment_event, command_string
from thesis_pipeline.gsm8k import answers_match, extract_final_answer
from thesis_pipeline.io_utils import read_jsonl, write_csv


def parse_prediction_arg(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path)
    path = Path(value)
    return path.stem, path


def summarize_predictions(name: str, path: Path) -> dict:
    rows = list(read_jsonl(path))
    total = len(rows)
    correct = 0
    accepted = 0
    has_supervisor = False
    attempt_counts: list[int] = []
    gen_token_counts: list[int] = []
    prompt_token_counts: list[int] = []

    for row in rows:
        gold = row.get("gold_final_answer") or extract_final_answer(row.get("gold_answer"))
        pred = (
            row.get("accepted_final_answer")
            or row.get("pred_final_answer")
            or extract_final_answer(row.get("accepted_answer") or row.get("prediction"))
        )
        if answers_match(pred, gold):
            correct += 1
        if "accepted_by_supervisor" in row:
            has_supervisor = True
            if row["accepted_by_supervisor"] is True:
                accepted += 1
        if row.get("attempt_count") is not None:
            attempt_counts.append(int(row["attempt_count"]))
        gt = row.get("total_model_tokens") or row.get("tokens_generated")
        if gt is not None:
            gen_token_counts.append(int(gt))
        pt = row.get("total_prompt_tokens") or row.get("prompt_tokens")
        if pt is not None:
            prompt_token_counts.append(int(pt))

    accuracy = correct / total if total else 0.0
    avg_attempts = sum(attempt_counts) / len(attempt_counts) if attempt_counts else ""
    avg_gen = sum(gen_token_counts) / len(gen_token_counts) if gen_token_counts else ""
    avg_prompt = sum(prompt_token_counts) / len(prompt_token_counts) if prompt_token_counts else ""
    avg_total = (
        avg_prompt + avg_gen if avg_prompt != "" and avg_gen != "" else ""
    )
    supervisor_accept_rate = accepted / total if (has_supervisor and total) else ""
    return {
        "name": name,
        "file": str(path),
        "total": total,
        "correct": correct,
        "accuracy": f"{accuracy:.4f}",
        "avg_prompt_tokens": f"{avg_prompt:.1f}" if avg_prompt != "" else "",
        "avg_gen_tokens": f"{avg_gen:.1f}" if avg_gen != "" else "",
        "avg_total_tokens": f"{avg_total:.1f}" if avg_total != "" else "",
        "supervisor_accept_rate": (
            f"{supervisor_accept_rate:.4f}" if supervisor_accept_rate != "" else ""
        ),
        "avg_attempts": f"{avg_attempts:.2f}" if avg_attempts != "" else "",
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.\-/ ]", " ", text.lower())).strip()


def contains_any_answer(prediction: str, answers: list[str]) -> bool:
    pred = normalize_text(prediction)
    return any(normalize_text(answer) in pred for answer in answers if answer)


def summarize_factuality(name: str, path: Path) -> dict:
    rows = list(read_jsonl(path))
    total = len(rows)
    lexical_pass = 0
    contains_correct = 0
    contains_incorrect = 0
    for row in rows:
        correct_answers = row.get("correct_answers") or [row.get("best_answer", "")]
        incorrect_answers = row.get("incorrect_answers") or []
        prediction = row.get("prediction", "")
        has_correct = contains_any_answer(prediction, correct_answers)
        has_incorrect = contains_any_answer(prediction, incorrect_answers)
        contains_correct += int(has_correct)
        contains_incorrect += int(has_incorrect)
        lexical_pass += int(has_correct and not has_incorrect)
    proxy_accuracy = lexical_pass / total if total else 0.0
    return {
        "name": name,
        "file": str(path),
        "total": total,
        "lexical_pass": lexical_pass,
        "lexical_proxy_accuracy": f"{proxy_accuracy:.4f}",
        "contains_correct": contains_correct,
        "contains_incorrect": contains_incorrect,
    }


def write_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction JSONL files.")
    parser.add_argument("--task", default="gsm8k", choices=["gsm8k", "factuality"])
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        help="Prediction path or name=path. Can be repeated.",
    )
    parser.add_argument("--csv-output", default="reports/gsm8k_summary.csv")
    parser.add_argument("--md-output", default="reports/gsm8k_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarizer = summarize_factuality if args.task == "factuality" else summarize_predictions
    summaries = [summarizer(name, path) for name, path in map(parse_prediction_arg, args.predictions)]
    write_csv(args.csv_output, summaries)
    write_markdown(Path(args.md_output), summaries)
    append_experiment_event(
        "evaluation",
        f"Evaluated {args.task} predictions",
        {
            "command": command_string([sys.executable, *sys.argv]),
            "task": args.task,
            "summaries": summaries,
            "csv_output": args.csv_output,
            "md_output": args.md_output,
        },
    )
    for summary in summaries:
        if args.task == "factuality":
            print(
                f"{summary['name']}: {summary['lexical_pass']}/{summary['total']} "
                f"lexical_proxy_accuracy={summary['lexical_proxy_accuracy']}"
            )
        else:
            print(
                f"{summary['name']}: {summary['correct']}/{summary['total']} "
                f"accuracy={summary['accuracy']}"
            )
    print(f"Saved reports to {args.csv_output} and {args.md_output}")


if __name__ == "__main__":
    main()
