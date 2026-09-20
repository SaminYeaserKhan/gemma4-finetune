"""Recover the model's own confidence in answers it has already written.

The third escalation gate. `length_score` is free but weak (AUC 0.676);
`disagreement_score` is strong (AUC 0.840) but costs k full generations, so a
single question takes k times as long to answer. On the target hardware -- old
laptops, office desktops, phones, anything without a serious GPU -- generation
is already the slow step, so tripling it is the difference between a usable
answer and one nobody waits for. Confidence sits between the two in cost: it is
a byproduct of generating that was being discarded. The question this script
answers is whether it also sits between them in quality.

Nothing is regenerated. Scoring stored text is one forward pass per answer
rather than a token-by-token loop, roughly 20x faster than producing it was.

    python experiments/score_confidence.py --adapter-dir gemma4-gsm8k-final

Writes id, mean_logprob, min_logprob and final_logprob to
`outputs/predictions/05_confidence_for_try1.jsonl`, which `analyze_supervision.py`
picks up via --confidence. Interruptible: --resume skips ids already written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running `python experiments/foo.py` puts experiments/ on sys.path, not the
# repo root, so the package would not import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.gsm8k import build_prompt, extract_final_answer, family_for
from thesis_pipeline.io_utils import append_jsonl, read_jsonl, repair_jsonl
from thesis_pipeline.model_utils import answer_confidence, load_inference_model, load_tokenizer


def final_answer_offset(prediction: str) -> int | None:
    """Where the final answer starts, as a character index into `prediction`.

    Prefers the `####` marker, which is the authoritative GSM8K format and what
    the model was fine-tuned to emit. Falls back to the last occurrence of the
    extracted answer for outputs that omit it -- the same fallback ordering
    `extract_final_answer` uses, so the two never disagree about which number
    is the answer.
    """
    marker = prediction.rfind("####")
    if marker != -1:
        return marker
    answer = extract_final_answer(prediction)
    if not answer:
        return None
    position = prediction.rfind(answer)
    return position if position != -1 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score stored answers for confidence.")
    parser.add_argument("--predictions", default=None, help="Defaults to the attempt-1 cache.")
    parser.add_argument("--adapter-dir", default="gemma4-gsm8k-final")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Base model id. Must match the model that wrote the predictions "
        "being scored -- confidence is read from that model's own logits.",
    )
    parser.add_argument(
        "--output", default="outputs/predictions/05_confidence_for_try1.jsonl"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip ids already scored.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ThesisConfig()
    if args.model_name:
        cfg = ThesisConfig(**{**cfg.__dict__, "model_name": args.model_name})
    # The confidence score is read off the same prompt the answer was written
    # from; a different chat template would score a different sequence.
    family = family_for(cfg.model_name)
    source = args.predictions or cfg.attempt1_cache
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if args.resume and output.exists():
        damaged = repair_jsonl(output)
        if damaged:
            print(f"Repaired {output}: dropped {damaged} incomplete row(s).")
        done = {int(row["id"]) for row in read_jsonl(output)}
        print(f"Resuming: {len(done)} already scored.")

    rows = list(read_jsonl(source))
    if args.limit is not None:
        rows = rows[: args.limit]
    todo = [row for row in rows if int(row["id"]) not in done]
    if not todo:
        print("Nothing to do.")
        return 0

    print(f"Loading {cfg.model_name} + {args.adapter_dir} ...")
    tokenizer = load_tokenizer(cfg)
    model = load_inference_model(cfg, args.adapter_dir)

    skipped = 0
    for index, row in enumerate(todo, 1):
        prediction = row.get("prediction") or ""
        if not prediction.strip():
            skipped += 1
            continue
        try:
            result = answer_confidence(
                model,
                tokenizer,
                build_prompt(row["question"], family),
                prediction,
                final_answer_offset(prediction),
            )
        except ValueError as exc:
            # Recorded rather than silently dropped: a missing row becomes an
            # inf score in the gate, which escalates it, so the count needs to
            # be visible.
            print(f"  id {row['id']}: skipped ({exc})")
            skipped += 1
            continue

        append_jsonl(
            output,
            {
                "id": int(row["id"]),
                "mean_logprob": result.mean_logprob,
                "min_logprob": result.min_logprob,
                "final_logprob": result.final_logprob,
                "scored_tokens": result.scored_tokens,
            },
        )
        if index % 100 == 0 or index == len(todo):
            print(f"  {index}/{len(todo)}")

    print(f"Wrote {output} ({len(todo) - skipped} scored, {skipped} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
