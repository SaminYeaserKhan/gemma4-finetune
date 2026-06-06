from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

from .config import ThesisConfig


class GenerationResult(NamedTuple):
    text: str
    tokens_generated: int
    prompt_tokens: int


class _MultiTokenStop(StoppingCriteria):
    """Stops generation when the tail of generated tokens matches stop_seq."""
    def __init__(self, stop_seq: list[int]):
        self.stop_seq = stop_seq
        self.n = len(stop_seq)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[1] < self.n:
            return False
        return input_ids[0, -self.n:].tolist() == self.stop_seq


def load_tokenizer(cfg: ThesisConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(cfg: ThesisConfig, for_training: bool = False):
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_name,
        quantization_config=quantization_config(),
        device_map="auto",
    )
    model.config.use_cache = not for_training
    return model


def load_inference_model(cfg: ThesisConfig, adapter_dir: str | Path | None = None):
    model = load_base_model(cfg, for_training=False)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    repetition_penalty: float = 1.15,
) -> GenerationResult:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    do_sample = temperature > 0
    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    end_turn_tokens = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList([_MultiTokenStop(end_turn_tokens)]) if end_turn_tokens else None
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_ids or None,
        "repetition_penalty": repetition_penalty,
        "stopping_criteria": stopping_criteria,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
    if "<end_of_turn>" in decoded:
        decoded = decoded.split("<end_of_turn>", 1)[0].strip()
    tokens_generated = len(tokenizer.encode(decoded, add_special_tokens=False))
    return GenerationResult(
        text=decoded,
        tokens_generated=tokens_generated,
        prompt_tokens=prompt_tokens,
    )
