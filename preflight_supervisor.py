"""Check a supervisor before committing hours or quota to it.

Answers, in about a minute, the questions that otherwise only surface halfway
through a long run:

- Does the endpoint answer at all, and as which exact model?
- Does the reply parse as a verdict, or is it silently truncated?
- How fast is it, and so how long will 1,319 judgments take?
- **Is it lenient?** A judge that accepts everything makes the cascade a no-op.
  Precision and recall against GSM8K ground truth on a small sample is the
  cheapest possible early read on that.
- For hosted providers: what is the *actual* rate limit on this account? Google
  no longer publishes free-tier numbers, but it does report them in the body of
  a 429.

Usage:
    python preflight_supervisor.py --provider llamacpp
    python preflight_supervisor.py --provider gemini --limit 30 --rpm 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from thesis_pipeline.config import ThesisConfig
from thesis_pipeline.gsm8k import answers_match, extract_final_answer
from thesis_pipeline.io_utils import read_jsonl
from thesis_pipeline.supervisor_client import (
    SupervisorAborted,
    SupervisorClient,
    parse_verdict,
)

# Google reports the breached quota inside the 429 body; this is the only place
# the real per-project limit is visible now that the docs omit it.
_QUOTA_VALUE_RE = re.compile(r'"quotaValue"\s*:\s*"?(\d+)"?')
_QUOTA_ID_RE = re.compile(r'"quotaId"\s*:\s*"([^"]+)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a supervisor provider.")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--supervisor-model", default=None)
    parser.add_argument("--supervisor-url", default=None)
    parser.add_argument("--rpm", type=float, default=None)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--attempt1-cache", default=None)
    parser.add_argument(
        "--full-scale",
        type=int,
        default=1319,
        help="Test-set size used to project total runtime.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ThesisConfig()
    provider = args.provider or cfg.supervisor_provider

    cache_path = Path(args.attempt1_cache or cfg.attempt1_cache)
    if not cache_path.exists():
        print(f"attempt-1 cache not found: {cache_path}", file=sys.stderr)
        return 1
    rows = [row for _, row in zip(range(args.limit), read_jsonl(cache_path))]
    if not rows:
        print(f"no rows in {cache_path}", file=sys.stderr)
        return 1

    local_runner = None
    if provider in {"local", "self"}:
        from supervise import build_local_runner

        local_runner = build_local_runner(cfg, provider, cfg.final_adapter_dir)

    client = SupervisorClient(
        cfg,
        provider,
        local_runner=local_runner,
        model=args.supervisor_model,
        base_url=args.supervisor_url,
        rpm=args.rpm if args.rpm is not None else cfg.supervisor_rpm,
    )

    print(f"provider:   {client.provider}")
    print(f"model:      {client.model}")
    if provider == "llamacpp":
        print(f"endpoint:   {client.base_url}")
    print(f"sample:     {len(rows)} cached attempt-1 answers from {cache_path}")
    print()

    stats = {
        "true_reject": 0, "false_reject": 0, "true_accept": 0, "false_accept": 0,
    }
    errors: list[str] = []
    unparseable = 0
    in_tokens = out_tokens = 0
    started = time.monotonic()

    for row in rows:
        candidate = row.get("prediction", "")
        gold = row.get("gold_final_answer") or extract_final_answer(row.get("gold_answer", ""))
        # The judge never sees this; it is only how we score the judge.
        actually_correct = answers_match(extract_final_answer(candidate), gold)
        try:
            decision = client.judge(row["question"], candidate, gold)
        except SupervisorAborted as exc:
            print(f"\nABORTED: {exc}", file=sys.stderr)
            _report_quota(errors)
            return 1
        if decision.error:
            errors.append(decision.error)
            continue
        # `exact` and `none` answer YES/NO by construction and are never asked
        # for JSON, so holding them to the schema would be a false alarm.
        if provider not in {"exact", "none"} and not _parses_strictly(decision.raw_response):
            unparseable += 1
        in_tokens += decision.input_tokens
        out_tokens += decision.output_tokens
        if decision.accepted:
            stats["true_accept" if actually_correct else "false_accept"] += 1
        else:
            stats["false_reject" if actually_correct else "true_reject"] += 1

    elapsed = time.monotonic() - started
    judged = sum(stats.values())

    print(f"judged:     {judged}/{len(rows)}   errors: {len(errors)}   "
          f"non-strict JSON: {unparseable}")
    print(f"elapsed:    {elapsed:.1f}s  ({elapsed / max(judged, 1):.2f}s per judgment)")
    print(f"tokens:     {in_tokens / max(judged, 1):.0f} in / "
          f"{out_tokens / max(judged, 1):.0f} out, per judgment")
    projected = elapsed / max(judged, 1) * args.full_scale
    print(f"projected:  {projected / 3600:.1f} h for a {args.full_scale}-question pass")
    print()

    if not judged:
        print("No judgments completed.", file=sys.stderr)
        _report_quota(errors)
        for err in errors[:3]:
            print(f"  {err}", file=sys.stderr)
        return 1

    rejected = stats["true_reject"] + stats["false_reject"]
    wrong = stats["true_reject"] + stats["false_accept"]
    right = stats["true_accept"] + stats["false_reject"]
    precision = stats["true_reject"] / rejected if rejected else 0.0
    recall = stats["true_reject"] / wrong if wrong else 0.0
    frr = stats["false_reject"] / right if right else 0.0

    print("Verifier quality on this sample (small -- indicative only):")
    print(f"  rejected:           {rejected}/{judged} ({rejected / judged:.0%})")
    print(f"  precision:          {precision:.3f}   (of rejections, share truly wrong)")
    print(f"  recall:             {recall:.3f}   (of wrong answers, share caught)")
    print(f"  false-reject rate:  {frr:.3f}   (of correct answers, share broken)")
    print()

    # The failure mode that would quietly sink the thesis is a judge that likes
    # everything: nothing escalates, nothing is retried, accuracy does not move.
    if rejected == 0:
        print("VERDICT: this judge accepted every answer. The cascade would be a "
              "no-op. Try a stronger model before running a full pass.")
        return 1
    if recall < 0.4:
        print(f"VERDICT: low recall ({recall:.2f}) -- most wrong answers are waved "
              "through. Usable, but expect a small accuracy gain.")
    elif frr > 0.25:
        print(f"VERDICT: high false-reject rate ({frr:.2f}) -- this judge will send "
              "correct answers into retries and may lose accuracy. Watch the flip matrix.")
    else:
        print("VERDICT: looks usable. Proceed to the full verdict pass.")
    if errors:
        print(f"\n{len(errors)} errors, first: {errors[0][:200]}")
        _report_quota(errors)
    return 0


def _parses_strictly(text: str) -> bool:
    """Did the reply arrive as clean JSON, or did the fallbacks have to rescue it?

    `parse_verdict` degrades through fenced blocks and bare YES/NO, and its last
    resort is to accept. A provider that regularly needs those fallbacks is one
    bad reply away from a silent approval, which is worth knowing before a run
    rather than after.
    """
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and "verdict" in payload


def _report_quota(errors: list[str]) -> None:
    for err in errors:
        quota = _QUOTA_VALUE_RE.search(err)
        if quota:
            name = _QUOTA_ID_RE.search(err)
            print(f"\nMeasured quota: {quota.group(1)} "
                  f"({name.group(1) if name else 'unnamed limit'})")
            return


if __name__ == "__main__":
    raise SystemExit(main())
