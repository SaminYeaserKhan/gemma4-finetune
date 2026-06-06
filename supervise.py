from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.gsm8k import answers_match, build_prompt, extract_final_answer
from thesis_pipeline.io_utils import append_jsonl
from thesis_pipeline.model_utils import generate_answer, load_inference_model, load_tokenizer
from thesis_pipeline.supervisor_client import SupervisorClient


def retry_question(question: str, previous_answer: str) -> str:
    return (
        f"{question.strip()}\n\n"
        "A verifier rejected your previous solution. Re-solve the problem from "
        "scratch, check the arithmetic carefully, and finish with exactly "
        "`#### <final answer>`.\n\n"
        f"Previous rejected answer:\n{previous_answer.strip()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised retry evaluation.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--provider", default=None, choices=["openai", "anthropic", "exact", "none"])
    parser.add_argument("--retry-limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ThesisConfig()
    if args.model_name:
        cfg = ThesisConfig(**{**cfg.__dict__, "model_name": args.model_name})
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else cfg.dataset_dir
    adapter_dir = Path(args.adapter_dir) if args.adapter_dir else cfg.final_adapter_dir
    limit = args.limit if args.limit is not None else cfg.supervisor_eval_size
    retry_limit = args.retry_limit if args.retry_limit is not None else cfg.retry_limit
    max_attempts = retry_limit + 1
    max_new_tokens = args.max_new_tokens or cfg.max_new_tokens
    output = Path(args.output) if args.output else (
        cfg.outputs_dir / "predictions" / f"supervised_{args.split}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")

    dataset = load_from_disk(str(dataset_dir))[args.split]
    end = min(len(dataset), args.offset + limit)
    rows = dataset.select(range(args.offset, end))

    tokenizer = load_tokenizer(cfg)
    model = load_inference_model(cfg, adapter_dir)
    supervisor = SupervisorClient(cfg, args.provider)

    for local_idx, row in enumerate(tqdm(rows, desc="Supervised eval")):
        example_id = args.offset + local_idx
        question = row["question"]
        gold_final = row.get("final_answer") or extract_final_answer(row["answer"])
        attempts = []
        total_model_tokens = 0
        total_prompt_tokens = 0
        current_question = question
        accepted_answer = ""
        accepted_final = None
        accepted_by_supervisor = False

        for attempt_idx in range(max_attempts):
            result = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=build_prompt(current_question),
                max_new_tokens=max_new_tokens,
                temperature=args.temperature if attempt_idx > 0 else 0.0,
            )
            candidate = result.text
            total_model_tokens += result.tokens_generated
            total_prompt_tokens += result.prompt_tokens
            pred_final = extract_final_answer(candidate)
            exact_correct = answers_match(pred_final, gold_final)
            decision = supervisor.judge(question, candidate, gold_final)
            attempts.append(
                {
                    "attempt": attempt_idx + 1,
                    "prediction": candidate,
                    "pred_final_answer": pred_final,
                    "tokens_generated": result.tokens_generated,
                    "prompt_tokens": result.prompt_tokens,
                    "exact_correct": exact_correct,
                    "supervisor_accepted": decision.accepted,
                    "supervisor_raw": decision.raw_response,
                }
            )
            if decision.accepted or attempt_idx == max_attempts - 1:
                accepted_answer = candidate
                accepted_final = pred_final
                accepted_by_supervisor = decision.accepted
                break
            current_question = retry_question(question, candidate)

        append_jsonl(
            output,
            {
                "id": example_id,
                "run_name": "supervised",
                "split": args.split,
                "question": question,
                "gold_answer": row["answer"],
                "gold_final_answer": gold_final,
                "accepted_answer": accepted_answer,
                "accepted_final_answer": accepted_final,
                "accepted_by_supervisor": accepted_by_supervisor,
                "attempt_count": len(attempts),
                "total_model_tokens": total_model_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "correct": answers_match(accepted_final, gold_final),
                "attempts": attempts,
            },
        )

    print(f"Saved supervised predictions to {output}")


if __name__ == "__main__":
    main()

