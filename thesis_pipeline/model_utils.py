from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList

from .config import ThesisConfig
from .gsm8k import GEMMA, ModelFamily, family_for


class GenerationResult(NamedTuple):
    text: str
    tokens_generated: int
    prompt_tokens: int


class ConfidenceResult(NamedTuple):
    """Three ways of reading the same forward pass.

    Which one predicts errors best is an empirical question, so all three are
    recorded -- they cost nothing extra once the pass has been run.

    mean_logprob   average over every answer token. Length-normalised, so it
                   partly re-measures the length gate.
    min_logprob    the single least-likely token: the weakest link. Not
                   length-normalised, and insensitive to a long confident
                   preamble around one bad step.
    final_logprob  average over the tokens of the final answer alone. The
                   thesis metric only grades that number, so this is the most
                   directly relevant span -- and the narrowest.
    """

    mean_logprob: float
    min_logprob: float
    final_logprob: float | None
    scored_tokens: int


def token_logprobs(logits, input_ids, start: int) -> list[float]:
    """Log-probability the model assigned to each token from `start` onward.

    A causal LM's logits at position i are its prediction for position i+1, so
    the score for `input_ids[i]` is read from `logits[i-1]`. That shift is the
    only subtle thing here and it fails silently if dropped -- you still get a
    full set of negative numbers, just for the wrong tokens.

    Scoring an already-written answer this way takes a single forward pass
    instead of a token-by-token generation loop, which is what makes the
    confidence gate cheap enough to add after the fact.
    """
    if start < 1:
        raise ValueError(f"start must be >= 1; nothing precedes token 0 (got {start})")
    # log_softmax over the vocabulary, in float32: the bf16 compute dtype has
    # ~3 decimal digits of mantissa, which is coarse for values summed over
    # hundreds of tokens.
    log_probs = torch.log_softmax(logits[0, start - 1 : -1].float(), dim=-1)
    targets = input_ids[0, start:]
    picked = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return [float(value) for value in picked]


def answer_confidence(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    final_char_offset: int | None = None,
) -> ConfidenceResult:
    """Score an already-written answer in one forward pass.

    `final_char_offset` is an index into `answer` marking where the final
    answer begins; tokens from there on are averaged separately. Passed in
    rather than detected here so this stays independent of GSM8K's `####`
    convention.

    Note this returns the model's raw probabilities, not the ones that drove
    sampling: `generate_answer` applies a repetition penalty, which distorts
    the distribution to shape output. The unpenalised value is the one that
    means "how sure was the model", so it is the right one for a gate.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    start = int(prompt_ids.shape[-1])
    encoded = tokenizer(prompt + answer, return_tensors="pt", return_offsets_mapping=True)
    input_ids = encoded["input_ids"]

    # Re-tokenising prompt+answer can merge across the join, which would shift
    # every index by one and silently score the wrong span. The prompt ends on
    # a newline so this should never fire; it is checked because the failure is
    # invisible in the output.
    if not torch.equal(input_ids[0, :start], prompt_ids[0]):
        raise ValueError("tokenising prompt+answer did not preserve the prompt prefix")
    if input_ids.shape[-1] <= start:
        raise ValueError("answer contributed no tokens to score")

    with torch.no_grad():
        logits = model(input_ids=input_ids.to(model.device)).logits
    scores = token_logprobs(logits.cpu(), input_ids, start)

    final_logprob = None
    if final_char_offset is not None:
        boundary = len(prompt) + final_char_offset
        offsets = encoded["offset_mapping"][0][start:]
        tail = [
            score
            for score, (begin, _) in zip(scores, offsets.tolist())
            if begin >= boundary
        ]
        if tail:
            final_logprob = sum(tail) / len(tail)

    return ConfidenceResult(
        mean_logprob=sum(scores) / len(scores),
        min_logprob=min(scores),
        final_logprob=final_logprob,
        scored_tokens=len(scores),
    )


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


def trim_at_end_tag(text: str, end_tag: str) -> str:
    """Cut a decoded answer at its turn-end marker.

    Generation decodes with `skip_special_tokens=False` so the marker is
    visible to cut on. Anything after it is the model talking past its turn,
    and must not survive into the stored prediction: `extract_final_answer`
    falls back to the last number in the text, so trailing chatter can quietly
    replace a correct answer.
    """
    if end_tag and end_tag in text:
        return text.split(end_tag, 1)[0].strip()
    return text.strip()


def load_base_model(cfg: ThesisConfig, for_training: bool = False):
    """Load the solver in 4-bit, through whichever class its family needs.

    Gemma 4 is a multimodal checkpoint and only loads through
    `AutoModelForImageTextToText`; a text-only solver such as Qwen needs
    `AutoModelForCausalLM`. `family_for` refuses an unregistered model rather
    than guessing, because the wrong choice here fails loudly but the wrong
    chat template downstream does not.
    """
    from transformers import AutoModelForCausalLM

    family = family_for(cfg.model_name)
    loader = AutoModelForImageTextToText if family.multimodal else AutoModelForCausalLM
    model = loader.from_pretrained(
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


def load_causal_lm(model_name: str):
    """Load a generic instruct model in 4-bit, for use as a local supervisor.

    Separate from `load_base_model` because Gemma 4 is multimodal and needs
    `AutoModelForImageTextToText`, while a text-only verifier (Qwen, Llama,
    Mistral) needs `AutoModelForCausalLM`.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config(),
        device_map="auto",
    )
    model.config.use_cache = True
    model.eval()
    return model


def generate_chat(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> GenerationResult:
    """Single-turn chat completion via the model's own chat template.

    Used for the local/self supervisor, where the verifier may be any instruct
    model rather than Gemma, so the hand-rolled `<start_of_turn>` tags in
    `gsm8k.build_prompt` do not apply.
    """
    messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:  # tokenizer without a chat template
        text = f"{system_prompt}\n\n{user_prompt}\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = model.generate(**inputs, **kwargs)
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return GenerationResult(
        text=decoded,
        tokens_generated=int(generated_ids.shape[-1]),
        prompt_tokens=prompt_tokens,
    )


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    repetition_penalty: float = 1.15,
    family: ModelFamily = GEMMA,
) -> GenerationResult:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    do_sample = temperature > 0
    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    end_turn_tokens = tokenizer.encode(family.end_tag, add_special_tokens=False)
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
    decoded = trim_at_end_tag(
        tokenizer.decode(generated_ids, skip_special_tokens=False), family.end_tag
    )
    tokens_generated = len(tokenizer.encode(decoded, add_special_tokens=False))
    return GenerationResult(
        text=decoded,
        tokens_generated=tokens_generated,
        prompt_tokens=prompt_tokens,
    )
