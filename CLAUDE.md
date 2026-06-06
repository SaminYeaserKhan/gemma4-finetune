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
python evaluate.py --predictions baseline=outputs/predictions/baseline_gsm8k_test.jsonl --predictions fine_tuned=outputs/predictions/fine_tuned_gsm8k_test.jsonl

# Supervised retry loop (exact = no-cost smoke test)
python supervise.py --provider exact --limit 5
python supervise.py --adapter-dir gemma4-gsm8k-final --limit 200

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
| `supervise.py` | Retry loop: generates → verifier judges → retries up to `RETRY_LIMIT` times |

**`thesis_pipeline/` package:**

- `config.py` — `ThesisConfig` frozen dataclass; all hyperparameters read from env vars with safe defaults. `add_common_args` / `apply_common_overrides` provide a uniform argparse interface across scripts.
- `gsm8k.py` — Prompt formatting (Gemma chat template tags), `extract_final_answer` (looks for `#### N` marker, falls back to last number), `answers_match` (numerically normalises fractions, commas, `$`, `%` before comparing). This is the critical correctness logic tested in `tests/`.
- `model_utils.py` — Loads model with 4-bit NF4 `BitsAndBytesConfig`; uses `AutoModelForImageTextToText` (Gemma 4 is multimodal). `generate_answer` stops on both EOS and `<end_of_turn>`.
- `supervisor_client.py` — `SupervisorClient.judge()` dispatches to `exact` (deterministic), `openai` (Responses API), or `anthropic` (Messages API). Returns `SupervisorDecision(accepted, raw_response, provider)`.
- `experiment_log.py` — Appends structured entries to `reports/experiment_log.jsonl` and `reports/experiment_log.md`.
- `io_utils.py` — JSONL read/write helpers.

## Key Constraints

- **Eval disabled during training** (`--eval-strategy no`): HF epoch-end evaluation OOMs on a 16 GB GPU (RTX 4080 SUPER). Always use `generate.py` + `evaluate.py` after training for the thesis metric.
- **Answer extraction** in `gsm8k.py` is the single most thesis-critical piece of code. The `#### N` marker is the authoritative GSM8K answer format; the last-number fallback handles model outputs that omit the marker.
- **LoRA targets** use a regex pattern in `ThesisConfig.lora_target_modules` that matches `model.language_model.layers.N.(self_attn|mlp)` — the Gemma 4 module path differs from standard Gemma 2.
- **Checkpoint resume**: `train.py` auto-selects the newest `checkpoints/checkpoint-*` directory. If optimizer state is incompatible after a package upgrade, use `--resume-mode adapter` to load only LoRA weights.
- The `Hugging face tokken.txt` file in the repo root may contain a real token — revoke and remove it.
