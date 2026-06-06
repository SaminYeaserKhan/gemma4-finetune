from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


@dataclass(frozen=True)
class ThesisConfig:
    model_name: str = os.getenv("BASE_MODEL_NAME", "google/gemma-4-E2B-it")
    dataset_name: str = os.getenv("DATASET_NAME", "openai/gsm8k")
    dataset_config: str = os.getenv("DATASET_CONFIG", "main")
    dataset_dir: Path = Path(os.getenv("DATASET_DIR", "gsm8k_formatted"))
    checkpoint_dir: Path = Path(os.getenv("CHECKPOINT_DIR", "checkpoints"))
    final_adapter_dir: Path = Path(os.getenv("FINAL_ADAPTER_DIR", "gemma4-gsm8k-final"))
    outputs_dir: Path = Path(os.getenv("OUTPUTS_DIR", "outputs"))
    reports_dir: Path = Path(os.getenv("REPORTS_DIR", "reports"))

    max_seq_length: int = env_int("MAX_SEQ_LENGTH", 1024)
    max_new_tokens: int = env_int("MAX_NEW_TOKENS", 512)
    train_epochs: int = env_int("TRAIN_EPOCHS", 3)
    per_device_train_batch_size: int = env_int("PER_DEVICE_TRAIN_BATCH_SIZE", 1)
    gradient_accumulation_steps: int = env_int("GRADIENT_ACCUMULATION_STEPS", 16)
    learning_rate: float = env_float("LEARNING_RATE", 2e-4)
    warmup_steps: int = env_int("WARMUP_STEPS", 50)
    save_steps: int = env_int("SAVE_STEPS", 200)
    save_total_limit: int = env_int("SAVE_TOTAL_LIMIT", 2)

    lora_r: int = env_int("LORA_R", 16)
    lora_alpha: int = env_int("LORA_ALPHA", 32)
    lora_dropout: float = env_float("LORA_DROPOUT", 0.05)
    lora_target_modules: str = os.getenv(
        "LORA_TARGET_MODULES",
        r"model\.language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
    )

    supervisor_provider: str = os.getenv("SUPERVISOR_PROVIDER", "openai")
    supervisor_eval_size: int = env_int("SUPERVISOR_EVAL_SIZE", 200)
    retry_limit: int = env_int("RETRY_LIMIT", 2)
    openai_model: str = os.getenv("OPENAI_SUPERVISOR_MODEL", "gpt-5.2")
    anthropic_model: str = os.getenv(
        "ANTHROPIC_SUPERVISOR_MODEL", "claude-sonnet-4-5"
    )

    def ensure_output_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "predictions").mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-dir", default=None, help="Path to formatted dataset")
    parser.add_argument("--model-name", default=None, help="Base Hugging Face model id")
    parser.add_argument("--adapter-dir", default=None, help="Optional PEFT adapter path")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--split", default="test", choices=["train", "test"])


def apply_common_overrides(args: argparse.Namespace, cfg: ThesisConfig) -> ThesisConfig:
    values = cfg.__dict__.copy()
    if getattr(args, "dataset_dir", None):
        values["dataset_dir"] = Path(args.dataset_dir)
    if getattr(args, "model_name", None):
        values["model_name"] = args.model_name
    if getattr(args, "adapter_dir", None):
        values["final_adapter_dir"] = Path(args.adapter_dir)
    return ThesisConfig(**values)

