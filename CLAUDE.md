# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Fine-tunes `google/gemma-4-E2B-it` on GSM8K using QLoRA, evaluates exact final-answer accuracy, and runs a hybrid supervisor retry loop with an external LLM (OpenAI or Anthropic) as a binary verifier. This is a thesis project targeting hallucination reduction.

## Environment Setup

```powershell
.\venv\Scripts\Activate.ps1
python check_env.py
python check_gpu.py
```

Secrets go in environment variables only — never in files:

```powershell
$env:HUGGINGFACE_TOKEN="hf_..."
$env:OPENAI_API_KEY="sk_..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

All tuneable defaults are in `.env.example` and read at import time by `thesis_pipeline/config.py` via `os.getenv`. Override any value by setting the corresponding env var before running a script.

## Commands

```powershell
# Prepare dataset
python prepare_data.py

# Train (auto-resumes from newest checkpoints/checkpoint-*)
python train.py

# Resume with full trainer state after a crash
python train.py --resume-mode full

# Validate latest checkpoint before resuming
python check_checkpoint.py --deep

# Generate predictions (remove --limit for full test set)
python generate.py --run-name baseline --limit 5
python generate.py --adapter-dir gemma4-gsm8k-final --run-name fine_tuned --limit 5

# Evaluate and produce CSV + Markdown reports
python evaluate.py --predictions baseline=outputs/predictions/01_answers_untrained_baseline.jsonl --predictions fine_tuned=outputs/predictions/02_answers_finetuned_try1_main.jsonl

# FYDP 3 cascade (exact = no-cost oracle smoke test)
python supervise.py --provider exact --gate none --limit 5
python supervise.py --gate length --escalation-rate 0.3 --dry-run

# Start the supervisor in a separate window, then check it
.\scripts\serve_verifier.ps1 -Model glm -Download   # first time only
.\scripts\serve_verifier.ps1 -Model glm
python preflight_supervisor.py --provider llamacpp

python supervise.py --provider llamacpp --gate none --verdict-only
python supervise.py --provider llamacpp --gate disagreement --feedback-level 1 `
  --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl

# Confidence gate: recover the log-probabilities generation threw away
python experiments/score_confidence.py --resume
.\scripts\score_sample_bank.ps1          # same, for the two sampled answer sets

# Cascade with the combined gate (agreement, ties broken by confidence)
.\scripts
un_combined_gate_arm.ps1

# FYDP 3 reporting
python analyze_supervision.py --cascade L1=outputs/predictions/cascade_gemini_disagreement_L1_test.jsonl

# Stacked arm at the gate's natural 37.3% threshold (~4.5 h, needs the verifier up)
.\scripts
un_stacked_arm.ps1

# Log a thesis milestone
python log_event.py --type milestone --title "..." --include-env

# Run tests
python -m unittest discover -s tests
```

## Architecture

The pipeline has five entry-point scripts and a shared `thesis_pipeline/` package:

| Script | Role |
|---|---|
| `prepare_data.py` | Downloads GSM8K from HF Hub, formats into `gsm8k_formatted/` |
| `train.py` | QLoRA fine-tuning; saves adapter to `gemma4-gsm8k-final/` |
| `generate.py` | Runs greedy inference (base or adapter); writes JSONL to `outputs/predictions/` |
| `evaluate.py` | Reads prediction JSONL; writes CSV + Markdown to `reports/` |
| `supervise.py` | FYDP 3 cascade: cached attempt 1 → gate → supervisor verdict → retry with feedback |
| `analyze_supervision.py` | FYDP 3 reports: gate curves, verifier confusion matrix, flip analysis, McNemar |
| `preflight_supervisor.py` | ~30 judgments against a provider: model id, speed, projected runtime, leniency, measured API quota |
| `scripts/serve_verifier.ps1` | Launches `llama-server` with the GGUF and MoE offload tuned to fit beside Gemma on 16 GB |

**`thesis_pipeline/` package:**

- `config.py` — `ThesisConfig` frozen dataclass; all hyperparameters read from env vars with safe defaults. `add_common_args` / `apply_common_overrides` provide a uniform argparse interface across scripts.
- `gsm8k.py` — Prompt formatting (Gemma chat template tags), `extract_final_answer` (looks for `#### N` marker, falls back to last number), `answers_match` (numerically normalises fractions, commas, `$`, `%` before comparing). This is the critical correctness logic tested in `tests/`.
- `model_utils.py` — Loads model with 4-bit NF4 `BitsAndBytesConfig`; uses `AutoModelForImageTextToText` (Gemma 4 is multimodal). `generate_answer` stops on both EOS and `<end_of_turn>`.
- `gate.py` — On-device escalation gates. Pure functions over signals the phone already has (`length_score`, `confidence_score`, `disagreement_score`); `gate_curve` sweeps every escalation rate so thresholds are chosen during analysis, not baked in. The three gates differ mainly in what they cost the device — nothing, one forward pass, and k full generations — which is the axis the thesis argues on.
- `supervisor_client.py` — `SupervisorClient.judge()` dispatches to `exact`, `self`, `local`, `llamacpp`, `gemini`, `openai`, `anthropic`, or `none`. One call returns `SupervisorDecision(accepted, pointer, correction, tokens)`; `.hint(level)` selects how much of it each arm is allowed to send back. Missing API keys raise at construction, because `judge()` deliberately converts API failures into accepts.
- `verdict_cache.py` — content-addressed replay of verdicts, keyed on `provider|model|system_prompt|question|candidate_answer`. The candidate text must stay in the key: attempt 2 asks about a different answer to the same question.
- `experiment_log.py` — Appends structured entries to `reports/experiment_log.jsonl` and `reports/experiment_log.md`.
- `io_utils.py` — JSONL read/write helpers. `repair_jsonl` drops rows damaged by an interrupted append (half-written line, or NUL padding from a length-extended but unflushed NTFS write) and is called on every `--resume` path, because otherwise a power cut crashes the exact code that exists to recover from it. `read_jsonl` stays strict: a corrupt file reaching analysis must fail loudly.

## Thesis Write-Up Source of Truth

`docs/THESIS_DOSSIER.md` holds the research questions, every measured result with its
provenance, the design decisions and why they were made, the threats to validity, and
the anticipated defence questions with answers. **Update it whenever a run produces a
number, a design decision is made or reversed, or a limitation is discovered** — it is
what the thesis report is written from. Every quantitative claim in it must name the
file or command it came from.

## Key Constraints

- **Eval disabled during training** (`--eval-strategy no`): HF epoch-end evaluation OOMs on a 16 GB GPU (RTX 4080 SUPER). Always use `generate.py` + `evaluate.py` after training for the thesis metric.
- **Answer extraction** in `gsm8k.py` is the single most thesis-critical piece of code. The `#### N` marker is the authoritative GSM8K answer format; the last-number fallback handles model outputs that omit the marker.
- **LoRA targets** use a regex pattern in `ThesisConfig.lora_target_modules` that matches `model.language_model.layers.N.(self_attn|mlp)` — the Gemma 4 module path differs from standard Gemma 2.
- **Checkpoint resume**: `train.py` auto-selects the newest `checkpoints/checkpoint-*` directory. If optimizer state is incompatible after a package upgrade, use `--resume-mode adapter` to load only LoRA weights.
- The `Hugging face tokken.txt` file in the repo root may contain a real token — revoke and remove it.
- **FYDP 3 framing**: the constraint is edge deployment, so the supervisor is a *rare, costly fallback*, not a quality booster. Calling it on every question defeats the thesis. Report accuracy against cloud tokens per question, and always report the flip matrix (retrying can turn correct answers wrong) alongside net accuracy.
- **Attempt-1 caching**: `supervise.py` reuses `outputs/predictions/02_answers_finetuned_try1_main.jsonl` rather than regenerating greedy attempt 1. Verified byte-identical on a 20-example resample (2026-08-05). Re-verify if the adapter, decoding parameters, or transformers version change.
- **An unreadable supervisor reply means accept, not reject.** `parse_verdict` falls back to acceptance on purpose — the cascade must not manufacture rejections out of its own bugs. The corollary is that anything ambiguous reaching it becomes silent approval, so empty and truncated replies are converted to errors at the provider boundary (`_reject_truncated`), and `SUPERVISOR_ERROR_ABORT` consecutive errors stop the run. Gemini 2.5 Flash needs `thinkingConfig.thinkingBudget = 0` for this reason: thinking tokens are charged against `maxOutputTokens` and exhaust it before any JSON is emitted. Thinking cannot be disabled on the 3.x line, which is why the config pins 2.5.
- **The primary supervisor is self-hosted**, served as GGUF by `llama-server` rather than loaded through transformers. bitsandbytes quantises from fp16 weights, so a 30B MoE would mean a ~60 GB download and still would not fit; GGUF is 17.5 GB and offloads experts to RAM. This also isolates the verifier from the Gemma dependency stack and makes swapping judges a server restart.
- **The cascade only beats free self-consistency when it is stacked on top of it.** Run
  *instead of* majority voting, no arm is statistically distinguishable from it
  (p=0.062/0.275/0.751). Layered *on top* — take the vote, escalate only the split votes —
  every arm is, and L2 goes 0.5944 -> 0.6262 at identical cloud cost
  (`analyze_supervision.stack_voting`, §5.7.3). The gain is free because the disagreement
  gate already generated the samples the vote needs; the un-stacked arm just discarded
  them. Never report a cascade number without its stacked counterpart and the
  self-consistency control beside it.
- **The best gate combines agreement with confidence, lexicographically.** Rank by
  sample disagreement, break its ties with the model's own confidence:
  AUC 0.840 -> 0.869, and 105 -> 125 errors caught at a 10% escalation budget with
  0.947 precision. Letting confidence *outweigh* agreement scores worse than agreement
  alone, so `gate.tie_broken_score` compresses the secondary inside the smallest gap
  between primary values and can never reorder them — there is no weight to tune, and
  none that could have been tuned on test data. Costs nothing: both signals were
  already being computed. Use `supervise.py --gate combined --confidence <file>`.
- **A 3-sample disagreement gate has only three thresholds**: 0, 1/3, 2/3. On the test
  split those hold 427 / 400 / 492 questions, so 37.3% is the only escalation rate the
  gate genuinely expresses; 30% cuts inside the 2/3 tie group and `select_for_escalation`
  breaks the tie by index.
- **Verifier strength is a measured variable, not an assumption.** The ladder is `self` (2B) → `Qwen3.5-9B` → `GLM-4.7-Flash` (30B-A3B) → `gemini` reference → `exact` oracle. Check leniency with `preflight_supervisor.py` *before* a full pass: a judge that accepts everything makes the cascade a no-op.
