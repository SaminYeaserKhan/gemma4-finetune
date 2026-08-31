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

    supervisor_provider: str = os.getenv("SUPERVISOR_PROVIDER", "gemini")
    supervisor_eval_size: int = env_int("SUPERVISOR_EVAL_SIZE", 200)
    retry_limit: int = env_int("RETRY_LIMIT", 2)
    openai_model: str = os.getenv("OPENAI_SUPERVISOR_MODEL", "gpt-5.2")
    anthropic_model: str = os.getenv(
        "ANTHROPIC_SUPERVISOR_MODEL", "claude-sonnet-4-5"
    )
    # Pinned to 2.5: it is the newest Gemini Flash that accepts
    # thinkingBudget=0, and thinking tokens billed against the output budget
    # return an empty reply that would be read as approval.
    gemini_model: str = os.getenv("GEMINI_SUPERVISOR_MODEL", "gemini-2.5-flash")
    local_supervisor_model: str = os.getenv(
        "LOCAL_SUPERVISOR_MODEL", "Qwen/Qwen3.5-9B"
    )
    # The primary supervisor: GLM-4.7-Flash, 30B total but only 3B active per
    # token, so the CPU offload needed to fit 17.5 GB of weights beside Gemma on
    # a 16 GB card costs far less than it would for a dense model of that size.
    llamacpp_base_url: str = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8080")
    llamacpp_model: str = os.getenv("LLAMACPP_MODEL", "GLM-4.7-Flash-UD-Q4_K_XL")
    supervisor_max_output_tokens: int = env_int("SUPERVISOR_MAX_OUTPUT_TOKENS", 200)
    supervisor_max_retries: int = env_int("SUPERVISOR_MAX_RETRIES", 5)
    supervisor_backoff_seconds: float = env_float("SUPERVISOR_BACKOFF_SECONDS", 2.0)
    supervisor_timeout: int = env_int("SUPERVISOR_TIMEOUT", 60)
    # Consecutive failures before a run gives up. Errors become accepts, so
    # without a ceiling a dead endpoint approves every remaining question.
    supervisor_error_abort: int = env_int("SUPERVISOR_ERROR_ABORT", 10)
    supervisor_rpm: float = env_float("SUPERVISOR_RPM", 0.0)
    verdict_cache: Path = Path(
        os.getenv("VERDICT_CACHE", "outputs/verdict_cache.jsonl")
    )

    # FYDP 3 cascade. Attempt 1 is greedy and deterministic, so it is read from
    # the FYDP 2 prediction file instead of being regenerated for every arm.
    attempt1_cache: Path = Path(
        os.getenv("ATTEMPT1_CACHE", "outputs/predictions/02_answers_finetuned_try1_main.jsonl")
    )
    sample_bank_k: int = env_int("SAMPLE_BANK_K", 2)
    gate_kind: str = os.getenv("GATE_KIND", "disagreement")
    escalation_rate: float = env_float("ESCALATION_RATE", 0.3)
    feedback_level: int = env_int("FEEDBACK_LEVEL", 1)
    retry_temperature: float = env_float("RETRY_TEMPERATURE", 0.7)

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

