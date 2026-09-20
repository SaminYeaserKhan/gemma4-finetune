from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.experiment_log import append_experiment_event, command_string
from thesis_pipeline.gsm8k import (
    answers_match,
    build_fewshot_prompt,
    build_prompt,
    family_for,
    extract_final_answer,
)
from thesis_pipeline.io_utils import append_jsonl, read_jsonl
from thesis_pipeline.model_utils import generate_answer, load_inference_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GSM8K predictions.")
    parser.add_argument("--task", default="gsm8k", choices=["gsm8k", "factuality"])
    parser.add_argument("--input-jsonl", default=None, help="Input JSONL for factuality task")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--few-shot",
        type=int,
        default=0,
        help="In-context examples from the train split (GSM8K task). "
        "Use >0 for base-model evaluation; keep 0 for the fine-tuned model.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ThesisConfig()
    if args.model_name:
        cfg = ThesisConfig(**{**cfg.__dict__, "model_name": args.model_name})
    # Chat tags and stop token follow the base model, not the pipeline. A
    # mismatch here does not crash -- it produces a fluent, parseable,
    # meaningless run -- so it is resolved once, up front, and threaded down.
    family = family_for(cfg.model_name)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else cfg.dataset_dir
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else None
    run_name = args.run_name or ("fine_tuned" if adapter_dir else "baseline")
    output = Path(args.output) if args.output else (
        cfg.outputs_dir / "predictions" / f"{run_name}_{args.task}_{args.split}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.offset == 0:
        output.write_text("", encoding="utf-8")

    tokenizer = load_tokenizer(cfg)
    model = load_inference_model(cfg, adapter_dir)
    max_new_tokens = args.max_new_tokens or cfg.max_new_tokens

    if args.task == "factuality":
        if not args.input_jsonl:
            raise ValueError("--input-jsonl is required for --task factuality")
        rows = list(read_jsonl(args.input_jsonl))
        end = len(rows) if args.limit is None else min(len(rows), args.offset + args.limit)
        selected_rows = rows[args.offset:end]
        for local_idx, row in enumerate(tqdm(selected_rows, desc=f"Generating {run_name}")):
            example_id = row.get("id", args.offset + local_idx)
            question = row["question"]
            result = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=build_prompt(question, family),
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                family=family,
            )
            append_jsonl(
                output,
                {
                    "id": example_id,
                    "run_name": run_name,
                    "task": "factuality",
                    "question": question,
                    "best_answer": row.get("best_answer", ""),
                    "correct_answers": row.get("correct_answers", []),
                    "incorrect_answers": row.get("incorrect_answers", []),
                    "prediction": result.text,
                    "tokens_generated": result.tokens_generated,
                    "prompt_tokens": result.prompt_tokens,
                },
            )
        append_experiment_event(
            "generation",
            f"Generated factuality predictions: {run_name}",
            {
                "command": command_string([sys.executable, *sys.argv]),
                "run_name": run_name,
                "task": args.task,
                "examples": len(selected_rows),
                "adapter_dir": str(adapter_dir) if adapter_dir else "",
                "output": str(output),
            },
            cfg.reports_dir,
        )
        print(f"Saved predictions to {output}")
        return

    dataset_dict = load_from_disk(str(dataset_dir))
    dataset = dataset_dict[args.split]
    shots: list[tuple[str, str]] = []
    if args.few_shot > 0:
        train_split = dataset_dict["train"]
        shots = [
            (train_split[i]["question"], train_split[i]["answer"])
            for i in range(args.few_shot)
        ]
    end = len(dataset) if args.limit is None else min(len(dataset), args.offset + args.limit)
    rows = dataset.select(range(args.offset, end))

    for local_idx, row in enumerate(tqdm(rows, desc=f"Generating {run_name}")):
        example_id = args.offset + local_idx
        question = row["question"]
        prompt = (
            build_fewshot_prompt(question, shots, family)
            if shots
            else build_prompt(question, family)
        )
        result = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            family=family,
        )
        gold_final = row.get("final_answer") or extract_final_answer(row["answer"])
        pred_final = extract_final_answer(result.text)
        append_jsonl(
            output,
            {
                "id": example_id,
                "run_name": run_name,
                "split": args.split,
                "question": question,
                "gold_answer": row["answer"],
                "gold_final_answer": gold_final,
                "prediction": result.text,
                "pred_final_answer": pred_final,
                "tokens_generated": result.tokens_generated,
                "prompt_tokens": result.prompt_tokens,
                "correct": answers_match(pred_final, gold_final),
            },
        )

    append_experiment_event(
        "generation",
        f"Generated GSM8K predictions: {run_name}",
        {
            "command": command_string([sys.executable, *sys.argv]),
            "run_name": run_name,
            "task": args.task,
            "split": args.split,
            "examples": len(rows),
            "adapter_dir": str(adapter_dir) if adapter_dir else "",
            "output": str(output),
            "temperature": args.temperature,
            "max_new_tokens": max_new_tokens,
            "few_shot": args.few_shot,
        },
        cfg.reports_dir,
    )
    print(f"Saved predictions to {output}")


if __name__ == "__main__":
    main()
