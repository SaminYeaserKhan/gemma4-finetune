from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any


USER_TAG = "<start_of_turn>user\n"
MODEL_TAG = "<start_of_turn>model\n"
END_TAG = "<end_of_turn>"

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?")


def build_prompt(question: str) -> str:
    return f"{USER_TAG}{question.strip()}\n{END_TAG}\n{MODEL_TAG}"


def build_completion(answer: str) -> str:
    return f"{answer.strip()}\n{END_TAG}"


def build_training_text(question: str, answer: str) -> str:
    return build_prompt(question) + build_completion(answer)


def build_fewshot_prompt(question: str, shots: list[tuple[str, str]]) -> str:
    """Multi-turn prompt with in-context (question, answer) examples.

    Used for evaluating the non-fine-tuned base model: the worked examples
    teach it the chain-of-thought + `#### <answer>` format from context.
    """
    prefix = "".join(
        build_training_text(shot_q, shot_a) + "\n" for shot_q, shot_a in shots
    )
    return prefix + build_prompt(question)


def format_gsm8k_example(example: dict[str, Any]) -> dict[str, str]:
    question = example["question"]
    answer = example["answer"]
    return {
        "question": question,
        "answer": answer,
        "prompt": build_prompt(question),
        "completion": build_completion(answer),
        "text": build_training_text(question, answer),
        "final_answer": extract_final_answer(answer) or "",
    }


def extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    marker_matches = re.findall(r"####\s*([^\n\r]+)", text)
    if marker_matches:
        return clean_answer(marker_matches[-1])
    number_matches = NUMBER_RE.findall(text)
    if number_matches:
        return clean_answer(number_matches[-1])
    return None


def clean_answer(value: str) -> str:
    value = value.strip()
    value = value.replace("$", "")
    value = value.replace("%", "")
    value = value.rstrip(".")
    return value.strip()


def canonical_number(value: str | None) -> Fraction | None:
    if value is None:
        return None
    value = clean_answer(value).replace(",", "")
    match = NUMBER_RE.search(value)
    if not match:
        return None
    token = match.group(0).replace(",", "")
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(token))
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def answers_match(prediction: str | None, gold: str | None) -> bool:
    pred_num = canonical_number(prediction)
    gold_num = canonical_number(gold)
    if pred_num is not None and gold_num is not None:
        return pred_num == gold_num
    if prediction is None or gold is None:
        return False
    return clean_answer(prediction).lower() == clean_answer(gold).lower()

