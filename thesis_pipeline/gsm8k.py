from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, NamedTuple


class ModelFamily(NamedTuple):
    """Everything about a base model that the pipeline has to vary.

    The chat tags and the loader class travel together because they are two
    faces of the same fact -- which model family this is. Separating them is
    how a Qwen run can load correctly and then prompt in Gemma's dialect,
    which produces fluent, parseable, meaningless output.
    """

    user_tag: str
    model_tag: str
    end_tag: str
    multimodal: bool


GEMMA = ModelFamily(
    user_tag="<start_of_turn>user\n",
    model_tag="<start_of_turn>model\n",
    end_tag="<end_of_turn>",
    # Gemma 4 is a multimodal checkpoint, so it loads through
    # AutoModelForImageTextToText even when only text is ever used.
    multimodal=True,
)

QWEN = ModelFamily(
    user_tag="<|im_start|>user\n",
    model_tag="<|im_start|>assistant\n",
    end_tag="<|im_end|>",
    multimodal=False,
)

_FAMILIES = (("gemma", GEMMA), ("qwen", QWEN))


def family_for(model_name: str) -> ModelFamily:
    """Chat format and loader class for a base model id.

    Deliberately refuses an unrecognised model instead of defaulting. A wrong
    chat template does not crash -- it yields a fluent run with a meaningless
    accuracy, which is far more expensive to discover than an error here.
    """
    lowered = model_name.lower()
    for needle, family in _FAMILIES:
        if needle in lowered:
            return family
    raise ValueError(
        f"No chat format registered for {model_name!r}. Add it to _FAMILIES in "
        "thesis_pipeline/gsm8k.py -- guessing the template would produce a "
        "plausible but invalid run."
    )


# Kept for the call sites that predate multi-model support.
USER_TAG = GEMMA.user_tag
MODEL_TAG = GEMMA.model_tag
END_TAG = GEMMA.end_tag

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*(?:\.\d+)?)?")


def build_prompt(question: str, family: ModelFamily = GEMMA) -> str:
    return f"{family.user_tag}{question.strip()}\n{family.end_tag}\n{family.model_tag}"


def build_completion(answer: str, family: ModelFamily = GEMMA) -> str:
    return f"{answer.strip()}\n{family.end_tag}"


def build_training_text(
    question: str, answer: str, family: ModelFamily = GEMMA
) -> str:
    return build_prompt(question, family) + build_completion(answer, family)


def build_fewshot_prompt(
    question: str,
    shots: list[tuple[str, str]],
    family: ModelFamily = GEMMA,
) -> str:
    """Multi-turn prompt with in-context (question, answer) examples.

    Used for evaluating the non-fine-tuned base model: the worked examples
    teach it the chain-of-thought + `#### <answer>` format from context.
    """
    prefix = "".join(
        build_training_text(shot_q, shot_a, family) + "\n" for shot_q, shot_a in shots
    )
    return prefix + build_prompt(question, family)


def build_retry_prompt(
    question: str,
    previous_answer: str,
    level: int,
    hint: str | None = None,
    family: ModelFamily = GEMMA,
) -> str:
    """Prompt for a second attempt after the supervisor rejected the first.

    `level` is the amount of feedback the supervisor is allowed to send back,
    which is also its cloud token cost -- the axis this thesis measures:

    - 0: the bare rejection, no diagnostic information at all.
    - 1: plus a pointer naming the step that went wrong.
    - 2: plus a correction stating the right interpretation.

    A level 2 hint still never contains the final answer; the small model has
    to do the arithmetic itself, otherwise the supervisor is solving the
    problem and the cascade proves nothing.

    Kept deliberately terse: the adapter was fine-tuned on bare questions, so
    every extra instruction word pushes the prompt further out of distribution.
    """
    if level not in (0, 1, 2):
        raise ValueError(f"feedback level must be 0, 1 or 2, got {level}")
    lines = [
        question.strip(),
        "",
        "Previous attempt:",
        previous_answer.strip(),
        "",
        "A verifier rejected that attempt.",
    ]
    if level > 0 and hint and hint.strip():
        lines.append(hint.strip())
    lines.append("Solve the problem again and finish with `#### <final answer>`.")
    return build_prompt("\n".join(lines), family)


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

