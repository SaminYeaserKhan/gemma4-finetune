"""FYDP 3 cascade: gate on-device, escalate rarely, retry with feedback.

Pipeline per question:

    cached attempt 1  ->  gate  ->  [escalated?]  ->  supervisor  ->  retry
                                          |                            ^
                                          no -> keep attempt 1         |
                                                                   until accepted
                                                                   or out of budget

Attempt 1 is never regenerated. It is greedy (temperature 0) with a fixed
adapter, so it is identical to the FYDP 2 prediction file -- reading it from
cache saves ~5 GPU-hours per arm and, more importantly, makes every arm start
from byte-identical answers so the comparisons are exactly paired.

Runs (see the plan for the full matrix):

    # phase 3/7: verdict pass only, no GPU needed for the local model
    python supervise.py --provider gemini --gate none --verdict-only

    # phases 4-6: one arm per feedback level
    python supervise.py --provider gemini --gate disagreement \
        --escalation-rate 0.3 --feedback-level 1 \
        --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl \
        --sample-bank outputs/predictions/04_answers_finetuned_try3.jsonl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from datasets import load_from_disk
from tqdm import tqdm

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.experiment_log import append_experiment_event, command_string
from thesis_pipeline.gate import (
    confidence_score,
    disagreement_score,
    length_score,
    select_for_escalation,
    tie_broken_score,
)
from thesis_pipeline.gsm8k import (
    answers_match,
    build_prompt,
    build_retry_prompt,
    extract_final_answer,
    family_for,
)
from thesis_pipeline.io_utils import append_jsonl, read_jsonl, repair_jsonl
from thesis_pipeline.timing import Stopwatch
from thesis_pipeline.supervisor_client import (
    SYSTEM_PROMPT,
    SupervisorAborted,
    SupervisorClient,
)
from thesis_pipeline.verdict_cache import VerdictCache, verdict_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FYDP 3 supervised cascade.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--adapter-dir", default=None)
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Run the base model with no LoRA adapter. Needed when evaluating a "
        "different base model, since an empty --adapter-dir falls back to the "
        "configured (Gemma) adapter.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--provider",
        default=None,
        choices=[
            "openai", "anthropic", "gemini", "llamacpp", "local", "self", "exact", "none",
        ],
    )
    parser.add_argument(
        "--supervisor-model",
        default=None,
        help="Exact judge id: a Gemini model, an HF repo, or the GGUF name served by llama-server.",
    )
    parser.add_argument(
        "--supervisor-url", default=None, help="Base URL for the llamacpp provider."
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="Client-side request pacing. 0 disables. Use ~10 on the Gemini free tier.",
    )
    parser.add_argument(
        "--verdict-cache",
        default=None,
        help="Verdict cache path. 'none' disables it and re-judges everything.",
    )
    parser.add_argument(
        "--judge-final-attempt",
        action="store_true",
        help="Also judge the last attempt. Off by default: nothing can be retried "
        "after it, so the verdict costs a call and changes no answer.",
    )
    parser.add_argument("--retry-limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None, help="Retry sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--attempt1-cache",
        default=None,
        help="Greedy attempt-1 predictions to reuse. Set to 'none' to regenerate.",
    )
    parser.add_argument(
        "--gate",
        default=None,
        choices=["none", "length", "disagreement", "combined"],
        help="On-device escalation gate. 'none' escalates every question. "
        "'combined' ranks by sample disagreement and breaks its ties with the "
        "model's own confidence, which disagreement alone cannot do.",
    )
    parser.add_argument(
        "--confidence",
        default="outputs/predictions/05_confidence_for_try1.jsonl",
        help="Confidence scores for the combined gate.",
    )
    parser.add_argument("--escalation-rate", type=float, default=None)
    parser.add_argument(
        "--sample-bank",
        action="append",
        default=[],
        help="Extra temperature-sampled prediction files, for the disagreement gate. Repeatable.",
    )
    parser.add_argument("--feedback-level", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument(
        "--verdict-only",
        action="store_true",
        help="Judge attempt 1 and stop. No retries, so the local model is never loaded.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the output file, skipping ids already present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report escalation counts and projected supervisor calls, then exit.",
    )
    return parser.parse_args()


def load_attempt1_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"attempt-1 cache not found: {path}. Run generate.py first, or pass "
            "--attempt1-cache none to regenerate on the fly."
        )
    return {int(row["id"]): row for row in read_jsonl(path)}


def load_confidence(path: str | None) -> dict[int, dict]:
    """Per-answer log-probabilities, for the combined gate.

    Unlike the analysis path this is not optional-and-silent: a run asking for
    `--gate combined` with no scores would quietly degrade to plain
    disagreement and report itself as the combined arm.
    """
    if not path:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"confidence scores not found: {resolved}. "
            "Run experiments/score_confidence.py first."
        )
    return {int(row["id"]): row for row in read_jsonl(resolved)}


def load_sample_bank(paths: list[str]) -> dict[int, list[str | None]]:
    """id -> extra sampled final answers, one per sample-bank file."""
    bank: dict[int, list[str | None]] = {}
    for path in paths:
        for row in read_jsonl(path):
            answer = row.get("pred_final_answer") or extract_final_answer(
                row.get("prediction")
            )
            bank.setdefault(int(row["id"]), []).append(answer)
    return bank


def gate_scores(
    ids: list[int],
    attempt1: dict[int, dict],
    gate_kind: str,
    bank: dict[int, list[str | None]],
    confidence: dict[int, dict] | None = None,
) -> list[float]:
    if gate_kind == "none":
        return [1.0] * len(ids)
    if gate_kind == "length":
        return [length_score(attempt1[i].get("tokens_generated")) for i in ids]
    if gate_kind in {"disagreement", "combined"}:
        if not bank:
            raise ValueError(
                f"--gate {gate_kind} needs at least one --sample-bank file "
                "(generate.py --temperature 0.7)."
            )
        scores = []
        for i in ids:
            first = attempt1[i].get("pred_final_answer") or extract_final_answer(
                attempt1[i].get("prediction")
            )
            scores.append(disagreement_score([first, *bank.get(i, [])]))
        if gate_kind == "disagreement":
            return scores
        # Disagreement over k samples has only k+1 distinct values, so it
        # cannot rank inside its own groups -- which is also why an escalation
        # rate that falls inside one has to cut it arbitrarily. Confidence is
        # continuous and still informative there, so it breaks the ties
        # without ever overriding the stronger signal.
        if not confidence:
            raise ValueError(
                "--gate combined needs --confidence "
                "(experiments/score_confidence.py)."
            )
        certainty = [
            confidence_score(confidence.get(i, {}).get("final_logprob")) for i in ids
        ]
        return tie_broken_score(scores, certainty)
    raise ValueError(f"unknown gate: {gate_kind}")


def build_local_runner(cfg: ThesisConfig, provider: str, adapter_dir: Path | None):
    """Local supervisors: a bigger model on the same GPU, or the model itself."""
    from thesis_pipeline.model_utils import (
        generate_chat,
        load_causal_lm,
        load_inference_model,
        load_tokenizer,
    )
    from transformers import AutoTokenizer

    if provider == "self":
        tokenizer = load_tokenizer(cfg)
        model = load_inference_model(cfg, adapter_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.local_supervisor_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_causal_lm(cfg.local_supervisor_model)

    def run(system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        result = generate_chat(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=cfg.supervisor_max_output_tokens,
        )
        return result.text, result.prompt_tokens, result.tokens_generated

    return run


def resolve_adapter_dir(
    adapter_dir: str | None, no_adapter: bool, default: Path
) -> Path | None:
    """Which LoRA adapter to load, including the option of none.

    An empty `--adapter-dir` falls through to the configured default, which is
    the Gemma adapter. Evaluating a different base model needs a way to say
    "no adapter" that cannot be read as "not specified", or a Qwen run quietly
    attempts to load Gemma weights onto it.
    """
    if no_adapter:
        return None
    return Path(adapter_dir) if adapter_dir else default


def main() -> None:
    args = parse_args()
    cfg = ThesisConfig()
    if args.model_name:
        cfg = ThesisConfig(**{**cfg.__dict__, "model_name": args.model_name})
    # Chat tags and stop token follow the base model; a mismatch produces a
    # fluent, parseable, meaningless run rather than an error.
    family = family_for(cfg.model_name)

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else cfg.dataset_dir
    adapter_dir = resolve_adapter_dir(
        args.adapter_dir, args.no_adapter, cfg.final_adapter_dir
    )
    retry_limit = args.retry_limit if args.retry_limit is not None else cfg.retry_limit
    gate_kind = args.gate or cfg.gate_kind
    escalation_rate = (
        args.escalation_rate if args.escalation_rate is not None else cfg.escalation_rate
    )
    feedback_level = (
        args.feedback_level if args.feedback_level is not None else cfg.feedback_level
    )
    temperature = args.temperature if args.temperature is not None else cfg.retry_temperature
    max_new_tokens = args.max_new_tokens or cfg.max_new_tokens
    provider = args.provider or cfg.supervisor_provider
    max_attempts = 1 if args.verdict_only else retry_limit + 1

    if gate_kind == "none":
        escalation_rate = 1.0

    output = Path(args.output) if args.output else (
        cfg.outputs_dir
        / "predictions"
        / f"cascade_{provider}_{gate_kind}_L{feedback_level}_{args.split}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_from_disk(str(dataset_dir))[args.split]
    end = len(dataset) if args.limit is None else min(len(dataset), args.offset + args.limit)
    rows = dataset.select(range(args.offset, end))
    ids = [args.offset + i for i in range(len(rows))]

    cache_path = args.attempt1_cache if args.attempt1_cache is not None else str(cfg.attempt1_cache)
    use_cache = cache_path.lower() != "none"
    attempt1 = load_attempt1_cache(Path(cache_path)) if use_cache else {}
    if use_cache:
        missing = [i for i in ids if i not in attempt1]
        if missing:
            raise ValueError(
                f"{len(missing)} ids missing from the attempt-1 cache "
                f"(first: {missing[:5]}). Regenerate it or pass --attempt1-cache none."
            )

    bank = load_sample_bank(args.sample_bank)
    confidence = load_confidence(args.confidence) if gate_kind == "combined" else {}
    scores = (
        gate_scores(ids, attempt1, gate_kind, bank, confidence)
        if use_cache
        else [1.0] * len(ids)
    )
    escalated_positions = select_for_escalation(scores, escalation_rate)
    escalated_ids = {ids[p] for p in escalated_positions}

    if args.dry_run:
        # Attempts that get judged: all of them, minus the last one, which
        # nothing can be retried after. Never below 1, so --verdict-only still
        # judges its single attempt.
        judged_attempts = max(1, max_attempts - (0 if args.judge_final_attempt else 1))
        escalated_n = len(escalated_ids)
        print(f"examples:          {len(ids)}")
        print(f"gate:              {gate_kind} @ {escalation_rate:.0%}")
        print(f"escalated:         {escalated_n}")
        print(f"supervisor calls:  {escalated_n} verdicts on attempt 1")
        if judged_attempts > 1:
            print(f"                   + up to {escalated_n * (judged_attempts - 1)} on retries")
        print(f"                   = {escalated_n * judged_attempts} worst case, before cache replay")
        print(f"local generations: {0 if args.verdict_only else escalated_n * retry_limit} (worst case)")
        return

    done_ids: set[int] = set()
    if args.resume and output.exists():
        # A run interrupted by power loss can leave a half-written final row.
        # Repair before reading, or the resume path crashes on exactly the
        # failure it exists to recover from.
        damaged = repair_jsonl(output)
        if damaged:
            print(f"Repaired {output}: dropped {damaged} incomplete row(s).")
        done_ids = {int(row["id"]) for row in read_jsonl(output)}
        print(f"Resuming: {len(done_ids)} rows already in {output}")
    elif args.offset == 0:
        output.write_text("", encoding="utf-8")

    local_runner = (
        build_local_runner(cfg, provider, adapter_dir)
        if provider in {"local", "self"}
        else None
    )
    supervisor = SupervisorClient(
        cfg,
        provider,
        local_runner=local_runner,
        model=args.supervisor_model,
        base_url=args.supervisor_url,
        rpm=args.rpm if args.rpm is not None else cfg.supervisor_rpm,
    )

    cache_arg = args.verdict_cache if args.verdict_cache is not None else str(cfg.verdict_cache)
    cache = VerdictCache(None if cache_arg.lower() == "none" else cache_arg)
    new_calls_total = 0

    def cached_judge(question: str, candidate: str, gold: str) -> tuple[object, bool]:
        """Judge, replaying an identical earlier judgment when there is one.

        Attempt 1 is byte-identical across L0/L1/L2, so this is what makes the
        three arms reject exactly the same set -- the difference between them
        has to be hint content, not which questions happened to be retried.
        """
        key = verdict_key(
            supervisor.provider, supervisor.model, SYSTEM_PROMPT, question, candidate
        )
        hit = cache.get(key)
        if hit is not None:
            return hit, False
        decision = supervisor.judge(question, candidate, gold)
        cache.put(key, decision)
        return decision, True

    # The fine-tuned model is only needed to produce retries. A verdict-only
    # pass, or a run where nothing escalates, should never pay to load it.
    model = tokenizer = None

    def ensure_model():
        nonlocal model, tokenizer
        if model is None:
            from thesis_pipeline.model_utils import load_inference_model, load_tokenizer

            tokenizer = load_tokenizer(cfg)
            model = load_inference_model(cfg, adapter_dir)
        return model, tokenizer

    for position, row in enumerate(tqdm(rows, desc=f"Cascade {provider} L{feedback_level}")):
        example_id = ids[position]
        if example_id in done_ids:
            continue
        question = row["question"]
        gold_final = row.get("final_answer") or extract_final_answer(row["answer"])
        escalated = example_id in escalated_ids

        cached = attempt1.get(example_id)
        if cached is not None:
            candidate = cached.get("prediction", "")
            gen_tokens = int(cached.get("tokens_generated") or 0)
            prompt_tokens = int(cached.get("prompt_tokens") or 0)
        else:
            gen_model, gen_tokenizer = ensure_model()
            from thesis_pipeline.model_utils import generate_answer

            result = generate_answer(
                model=gen_model,
                tokenizer=gen_tokenizer,
                prompt=build_prompt(question, family),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                family=family,
            )  # not timed: attempt 1 is normally replayed from cache, so a
               # figure here would describe some runs and not others
            candidate, gen_tokens, prompt_tokens = (
                result.text,
                result.tokens_generated,
                result.prompt_tokens,
            )

        attempts: list[dict] = []
        # Where this question's wall clock goes. Tokens are a good proxy for
        # cloud cost and a poor one for latency, which is what the thesis
        # actually argues about.
        local_clock, cloud_clock = Stopwatch(), Stopwatch()
        question_started = time.monotonic()
        total_model_tokens = gen_tokens
        total_prompt_tokens = prompt_tokens
        supervisor_in = supervisor_out = supervisor_calls = 0
        supervisor_new_calls = 0
        accepted_by_supervisor = False

        for attempt_idx in range(max_attempts):
            is_final = attempt_idx == max_attempts - 1
            pred_final = extract_final_answer(candidate)
            record = {
                "attempt": attempt_idx + 1,
                "prediction": candidate,
                "pred_final_answer": pred_final,
                "exact_correct": answers_match(pred_final, gold_final),
            }

            # Un-escalated questions never leave the device: no verdict, no
            # cost, answer stands. That is the entire point of the gate.
            if not escalated:
                attempts.append({**record, "supervisor_accepted": None})
                break

            # Nothing can be retried after the last attempt, so its verdict
            # would cost a call and change no answer. `attempt_idx > 0` keeps
            # --verdict-only working, where the first attempt is also the last
            # and judging it is the entire purpose of the run.
            if is_final and attempt_idx > 0 and not args.judge_final_attempt:
                attempts.append({**record, "supervisor_accepted": None})
                break

            with cloud_clock:
                decision, was_new = cached_judge(question, candidate, gold_final)
            supervisor_calls += 1
            supervisor_new_calls += int(was_new)
            supervisor_in += decision.input_tokens
            supervisor_out += decision.output_tokens
            hint = decision.hint(feedback_level)
            attempts.append(
                {
                    **record,
                    "supervisor_accepted": decision.accepted,
                    "supervisor_pointer": decision.pointer,
                    "supervisor_correction": decision.correction,
                    "supervisor_hint_sent": hint,
                    "supervisor_raw": decision.raw_response,
                    "supervisor_input_tokens": decision.input_tokens,
                    "supervisor_output_tokens": decision.output_tokens,
                    "supervisor_error": decision.error,
                }
            )
            accepted_by_supervisor = decision.accepted
            if decision.accepted or is_final:
                break

            gen_model, gen_tokenizer = ensure_model()
            from thesis_pipeline.model_utils import generate_answer

            with local_clock:
                retry = generate_answer(
                    model=gen_model,
                    tokenizer=gen_tokenizer,
                    prompt=build_retry_prompt(
                        question, candidate, feedback_level, hint, family
                    ),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    family=family,
                )
            candidate = retry.text
            total_model_tokens += retry.tokens_generated
            total_prompt_tokens += retry.prompt_tokens

        accepted_final = extract_final_answer(candidate)
        new_calls_total += supervisor_new_calls
        append_jsonl(
            output,
            {
                "id": example_id,
                "run_name": f"cascade_{provider}_{gate_kind}_L{feedback_level}",
                "split": args.split,
                "question": question,
                "gold_answer": row["answer"],
                "gold_final_answer": gold_final,
                "gate_kind": gate_kind,
                "gate_score": scores[position],
                "escalated": escalated,
                "feedback_level": feedback_level,
                "accepted_answer": candidate,
                "accepted_final_answer": accepted_final,
                "accepted_by_supervisor": accepted_by_supervisor,
                "attempt_count": len(attempts),
                # `supervisor_calls` is the judgment count the cascade needed,
                # which is the cost a real deployment would pay. `_new_calls`
                # is what this run actually sent, after cache replay. The paper
                # reports the former; the latter is bookkeeping.
                "supervisor_calls": supervisor_calls,
                "supervisor_new_calls": supervisor_new_calls,
                "total_model_tokens": total_model_tokens,
                "total_prompt_tokens": total_prompt_tokens,
                "supervisor_input_tokens": supervisor_in,
                "supervisor_output_tokens": supervisor_out,
                "correct": answers_match(accepted_final, gold_final),
                # Wall clock, seconds. `seconds_local_generation` excludes the
                # cached attempt 1, so it is retry cost only; `seconds_total`
                # is what the user of the device would actually wait, minus
                # that same cached first answer.
                "seconds_total": round(time.monotonic() - question_started, 4),
                "seconds_local_generation": round(local_clock.seconds, 4),
                "seconds_supervisor": round(cloud_clock.seconds, 4),
                "supervisor_call_count_timed": cloud_clock.count,
                "attempts": attempts,
            },
        )

    append_experiment_event(
        "supervision",
        f"Cascade run: {provider} / {gate_kind} @ {escalation_rate:.0%} / L{feedback_level}",
        {
            "command": command_string([sys.executable, *sys.argv]),
            "provider": provider,
            "supervisor_model": supervisor.model,
            "gate": gate_kind,
            "escalation_rate": escalation_rate,
            "feedback_level": feedback_level,
            "retry_limit": retry_limit,
            "retry_temperature": temperature,
            "escalated": len(escalated_ids),
            "examples": len(ids),
            "attempt1_cache": cache_path if use_cache else "regenerated",
            "sample_bank": args.sample_bank,
            "output": str(output),
            "verdict_cache": cache_arg,
            "supervisor_new_calls": new_calls_total,
        },
        cfg.reports_dir,
    )
    stats = cache.stats()
    print(f"Saved cascade predictions to {output}")
    print(
        f"Supervisor calls: {new_calls_total} new, {stats['hits']} replayed from cache"
    )


if __name__ == "__main__":
    try:
        main()
    except SupervisorAborted as exc:
        # Rows are appended as they finish, so everything judged before the
        # failure is on disk and `--resume` will skip it.
        print(f"\nRun aborted: {exc}", file=sys.stderr)
        raise SystemExit(1)
