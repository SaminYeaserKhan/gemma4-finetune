"""How long does one question actually take?

The thesis argues about time to answer on hardware without a serious GPU, but
every run so far recorded tokens and used them as a proxy. Tokens measure cloud
cost well and latency badly: a local token and a remote token make the device
wait for completely different reasons, and trading one for the other is the
entire point of the gate.

This measures the three components once each and composes them:

  * how long the fine-tuned model takes to write one answer
  * how long the checker takes to judge one answer
  * what those add up to, per question, at a given escalation rate

Nothing here re-runs an experiment. It is a benchmark over a small sample, so
it finishes in minutes rather than hours.

    python experiments/benchmark_latency.py --limit 40
    python experiments/benchmark_latency.py --limit 40 --skip-checker

**The measurement is on this machine's GPU, not on a weak device.** An RTX 4080
SUPER is not the deployment target the thesis describes, so treat the local
numbers as a floor: real edge hardware is slower, and the ratio between local
and remote cost moves further towards local. Say so when quoting these.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.gsm8k import build_prompt
from thesis_pipeline.io_utils import read_jsonl
from thesis_pipeline.timing import Stopwatch


def latency_profile(
    seconds_per_generation: float,
    seconds_per_judgement: float,
    samples: int,
    escalation_rate: float,
    rejection_rate: float,
) -> dict:
    """Compose measured component times into per-question latency.

    Two paths. A question that stays on the device pays for its `samples`
    generations and nothing else. A question that escalates pays for those,
    plus one judgement, plus a retry generation -- but only when the checker
    actually rejects, which is why `rejection_rate` scales the retry alone and
    not the judgement.

    The average is the two paths weighted by how often each happens, which is
    the number a deployment would quote.
    """
    if not 0.0 <= escalation_rate <= 1.0:
        raise ValueError(f"escalation_rate must be in [0, 1], got {escalation_rate}")
    if not 0.0 <= rejection_rate <= 1.0:
        raise ValueError(f"rejection_rate must be in [0, 1], got {rejection_rate}")

    local = samples * seconds_per_generation
    escalated = local + seconds_per_judgement + rejection_rate * seconds_per_generation
    return {
        "seconds_local_path": local,
        "seconds_escalated_path": escalated,
        "seconds_average": (1 - escalation_rate) * local + escalation_rate * escalated,
    }


def summarise(name: str, values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "what": name,
        "n": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[min(int(len(ordered) * 0.9), len(ordered) - 1)],
        "min": ordered[0],
        "max": ordered[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40, help="Questions to time.")
    parser.add_argument("--adapter-dir", default="gemma4-gsm8k-final")
    parser.add_argument("--supervisor-url", default="http://127.0.0.1:8080")
    parser.add_argument("--supervisor-model", default=None)
    parser.add_argument("--skip-checker", action="store_true")
    parser.add_argument("--escalation-rate", type=float, default=0.30)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--rejection-rate",
        type=float,
        default=332 / 396,
        help="Share of escalated questions the checker rejected in the reported run.",
    )
    parser.add_argument("--output", default="reports/latency_benchmark.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ThesisConfig()
    rows = list(read_jsonl(cfg.attempt1_cache))[: args.limit]
    print(f"Timing {len(rows)} questions.\n")

    from thesis_pipeline.model_utils import (
        generate_answer,
        load_inference_model,
        load_tokenizer,
    )

    print("Loading the fine-tuned model ...")
    load_started = time.monotonic()
    tokenizer = load_tokenizer(cfg)
    model = load_inference_model(cfg, args.adapter_dir)
    load_seconds = time.monotonic() - load_started
    print(f"  loaded in {load_seconds:.1f}s (paid once per process, not per question)")

    # The first generation of a process pays for CUDA kernel compilation and
    # cache warm-up. Timing it would inflate the per-question figure by seconds
    # and describe a cost no real question after the first one pays.
    print("Warming up ...")
    generate_answer(
        model=model, tokenizer=tokenizer,
        prompt=build_prompt(rows[0]["question"]),
        max_new_tokens=cfg.max_new_tokens, temperature=0.0,
    )

    print("Timing local generation ...")
    local_times, local_tokens = [], []
    for i, row in enumerate(rows, 1):
        watch = Stopwatch()
        with watch:
            result = generate_answer(
                model=model, tokenizer=tokenizer,
                prompt=build_prompt(row["question"]),
                max_new_tokens=cfg.max_new_tokens, temperature=0.0,
            )
        local_times.append(watch.seconds)
        local_tokens.append(result.tokens_generated)
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}")

    checker_times: list[float] = []
    checker_model = args.supervisor_model or "(server default)"
    if not args.skip_checker:
        print("Timing the checker ...")
        from thesis_pipeline.supervisor_client import SupervisorClient

        client = SupervisorClient(
            cfg, "llamacpp", model=args.supervisor_model,
            base_url=args.supervisor_url, rpm=0,
        )
        checker_model = client.model
        for i, row in enumerate(rows, 1):
            watch = Stopwatch()
            with watch:
                client.judge(row["question"], row.get("prediction", ""), None)
            checker_times.append(watch.seconds)
            if i % 10 == 0:
                print(f"  {i}/{len(rows)}")

    gen = summarise("local generation (1 answer)", local_times)
    tok = summarise("tokens per answer", [float(t) for t in local_tokens])
    parts = [gen, tok]
    if checker_times:
        parts.append(summarise("checker judgement (1 answer)", checker_times))

    print("\n" + "=" * 78)
    print(f"{'what':34s} {'n':>4s} {'mean':>8s} {'median':>8s} {'p90':>8s} {'max':>8s}")
    for p in parts:
        print(f"{p['what']:34s} {p['n']:4d} {p['mean']:8.2f} {p['median']:8.2f} "
              f"{p['p90']:8.2f} {p['max']:8.2f}")
    print(f"\nlocal throughput: {sum(local_tokens) / sum(local_times):.1f} tokens/second")

    profile = None
    if checker_times:
        profile = latency_profile(
            seconds_per_generation=gen["mean"],
            seconds_per_judgement=statistics.mean(checker_times),
            samples=args.samples,
            escalation_rate=args.escalation_rate,
            rejection_rate=args.rejection_rate,
        )
        print(f"\nPer question, with {args.samples} samples and "
              f"{args.escalation_rate:.0%} escalation:")
        print(f"  stays on the device      {profile['seconds_local_path']:6.2f} s")
        print(f"  goes to the checker      {profile['seconds_escalated_path']:6.2f} s")
        print(f"  average across all       {profile['seconds_average']:6.2f} s")
        print(f"\n  one answer, no cascade   {gen['mean']:6.2f} s  "
              f"({profile['seconds_average'] / gen['mean']:.1f}x slower with the cascade)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "questions_timed": len(rows),
        "gpu_note": "measured on an RTX 4080 SUPER; weak hardware is slower",
        "model_load_seconds": round(load_seconds, 2),
        "checker_model": checker_model,
        "components": parts,
        "local_tokens_per_second": sum(local_tokens) / sum(local_times),
        "assumptions": {
            "samples": args.samples,
            "escalation_rate": args.escalation_rate,
            "rejection_rate": args.rejection_rate,
        },
        "profile": profile,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
