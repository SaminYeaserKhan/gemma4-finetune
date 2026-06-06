from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.gsm8k import format_gsm8k_example
from thesis_pipeline.io_utils import write_jsonl


def prepare_gsm8k(cfg: ThesisConfig, output_dir: Path) -> None:
    dataset = load_dataset(cfg.dataset_name, cfg.dataset_config)
    formatted = dataset.map(format_gsm8k_example)
    formatted.save_to_disk(str(output_dir))

    print(f"Saved formatted GSM8K dataset to {output_dir}")
    print(f"Train examples: {len(formatted['train'])}")
    print(f"Test examples:  {len(formatted['test'])}")
    print("\nPreview:")
    print(formatted["train"][0]["text"])


def prepare_truthfulqa_sample(output_path: Path, limit: int) -> None:
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    rows = []
    for idx, example in enumerate(dataset.select(range(min(limit, len(dataset))))):
        rows.append(
            {
                "id": idx,
                "question": example["question"],
                "best_answer": example.get("best_answer", ""),
                "correct_answers": example.get("correct_answers", []),
                "incorrect_answers": example.get("incorrect_answers", []),
            }
        )
    write_jsonl(output_path, rows)
    print(f"Saved factuality sample to {output_path} ({len(rows)} examples)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare thesis datasets.")
    parser.add_argument("--output-dir", default="gsm8k_formatted")
    parser.add_argument("--with-factuality", action="store_true")
    parser.add_argument("--factuality-output", default="outputs/factuality_sample.jsonl")
    parser.add_argument("--factuality-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ThesisConfig()
    cfg.ensure_output_dirs()
    prepare_gsm8k(cfg, Path(args.output_dir))
    if args.with_factuality:
        prepare_truthfulqa_sample(Path(args.factuality_output), args.factuality_limit)


if __name__ == "__main__":
    main()

