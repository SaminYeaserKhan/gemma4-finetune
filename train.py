from __future__ import annotations

import argparse
import re
import os
from pathlib import Path

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.gsm8k import build_completion, build_prompt


class CausalLMAnswerOnlyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [
            torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
        ]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def tokenize_answer_only(example: dict, tokenizer, max_length: int) -> dict:
    prompt = example.get("prompt") or build_prompt(example["question"])
    completion = example.get("completion") or build_completion(example["answer"])
    full_text = prompt + completion

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    labels = list(encoded["input_ids"])
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    labels = [
        token if mask == 1 else -100
        for token, mask in zip(labels, encoded["attention_mask"])
    ]
    encoded["labels"] = labels
    return encoded


def latest_checkpoint(checkpoint_dir: Path) -> str | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = [
        item
        for item in checkpoint_dir.iterdir()
        if item.is_dir() and item.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    def checkpoint_step(path: Path) -> int:
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        return int(match.group(1)) if match else -1

    return str(max(checkpoints, key=checkpoint_step))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 on GSM8K.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--final-adapter-dir", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Override checkpoint save frequency. Use 50 or 100 for frequent outages.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=None,
        help="Override number of recent checkpoints to keep.",
    )
    parser.add_argument(
        "--eval-strategy",
        choices=["no", "epoch", "steps"],
        default="no",
        help=(
            "Use 'no' for RTX 4080 training stability. Full text-loss eval can "
            "OOM; run generate.py/evaluate.py after training instead."
        ),
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Only used when --eval-strategy steps is selected.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Limit eval examples if in-training eval is enabled.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--resume-mode",
        choices=["full", "adapter"],
        default="full",
        help=(
            "full resumes optimizer/scheduler/trainer state; adapter loads only LoRA "
            "weights from the latest checkpoint and starts a fresh optimizer."
        ),
    )
    return parser.parse_args()


def patch_torch_load_for_trusted_local_checkpoints(torch_module) -> None:
    """Restore pre-2.6 torch.load behavior for local Trainer checkpoint files.

    PyTorch 2.6 defaults torch.load(..., weights_only=True). Older Hugging Face
    Trainer checkpoints may contain RNG/training objects that need the previous
    behavior when resuming from a checkpoint you created locally.
    """

    original_load = torch_module.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch_module.load = compatible_load


def main() -> None:
    args = parse_args()
    cfg = ThesisConfig()
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else cfg.dataset_dir
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else cfg.checkpoint_dir
    final_adapter_dir = (
        Path(args.final_adapter_dir) if args.final_adapter_dir else cfg.final_adapter_dir
    )
    model_name = args.model_name or cfg.model_name
    max_seq_length = args.max_seq_length or cfg.max_seq_length
    save_steps = args.save_steps or cfg.save_steps
    save_total_limit = args.save_total_limit or cfg.save_total_limit
    cfg = ThesisConfig(
        **{
            **cfg.__dict__,
            "model_name": model_name,
            "dataset_dir": dataset_dir,
            "checkpoint_dir": checkpoint_dir,
            "final_adapter_dir": final_adapter_dir,
            "max_seq_length": max_seq_length,
            "save_steps": save_steps,
            "save_total_limit": save_total_limit,
        }
    )

    # Import these only after parsing args. On this Windows setup, importing the
    # native ML stack in the wrong order can trigger a Python.exe memory dialog.
    from datasets import load_from_disk

    dataset = load_from_disk(str(cfg.dataset_dir))
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Test examples:  {len(dataset['test'])}")

    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    from thesis_pipeline.model_utils import load_base_model, load_tokenizer

    patch_torch_load_for_trusted_local_checkpoints(torch)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training this model needs the RTX GPU.")

    tokenizer = load_tokenizer(cfg)
    print(f"Tokenizing with max_seq_length={cfg.max_seq_length} and answer-only labels...")
    tokenized_train = dataset["train"].map(
        lambda row: tokenize_answer_only(row, tokenizer, cfg.max_seq_length),
        remove_columns=dataset["train"].column_names,
    )
    tokenized_test = None
    if args.eval_strategy != "no":
        eval_source = dataset["test"]
        if args.eval_limit is not None:
            eval_source = eval_source.select(range(min(args.eval_limit, len(eval_source))))
        tokenized_test = eval_source.map(
            lambda row: tokenize_answer_only(row, tokenizer, cfg.max_seq_length),
            remove_columns=eval_source.column_names,
        )

    model = load_base_model(cfg, for_training=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    resume_checkpoint = None if args.no_resume else latest_checkpoint(cfg.checkpoint_dir)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )

    peft_config = lora_config
    if args.resume_mode == "adapter" and resume_checkpoint:
        print(f"Loading adapter weights from checkpoint: {resume_checkpoint}")
        model = PeftModel.from_pretrained(model, resume_checkpoint, is_trainable=True)
        peft_config = None

    training_args = SFTConfig(
        output_dir=str(cfg.checkpoint_dir),
        num_train_epochs=cfg.train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=CausalLMAnswerOnlyCollator(tokenizer),
        args=training_args,
        peft_config=peft_config,
    )

    if resume_checkpoint:
        if args.resume_mode == "full":
            print(f"Resuming full trainer state from checkpoint: {resume_checkpoint}")
        else:
            print("Continuing from adapter weights with a fresh optimizer/scheduler.")
    else:
        print("Starting fresh training.")

    trainer.train(
        resume_from_checkpoint=resume_checkpoint if args.resume_mode == "full" else None
    )
    trainer.save_model(str(cfg.final_adapter_dir))
    tokenizer.save_pretrained(str(cfg.final_adapter_dir))
    print(f"Training complete. Adapter saved to {cfg.final_adapter_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
