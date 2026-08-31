# Gemma 4 GSM8K Hallucination-Reduction Thesis Pipeline

> **New here, or writing the thesis paper?** Start with
> **[`docs/PAPER_GUIDE.md`](docs/PAPER_GUIDE.md)** — what was built, every result and
> what it means, a glossary, and the questions to expect at the defence, all in
> plain language. `outputs/predictions/README.md` explains the data files.


This project fine-tunes `google/gemma-4-E2B-it` on GSM8K with QLoRA, evaluates exact final-answer accuracy, and optionally runs a hybrid supervisor retry loop with OpenAI or Anthropic as the binary verifier.

## Repository Layout

- `prepare_data.py`: downloads GSM8K, formats user/model turns, and can prepare a small TruthfulQA-style factuality sample.
- `train.py`: resumes/starts QLoRA fine-tuning with answer-only label masking.
- `generate.py`: produces baseline or adapter predictions as JSONL.
- `evaluate.py`: creates CSV and Markdown summaries from prediction logs.
- `supervise.py`: runs supervised retry evaluation with up to two retries by default.
- `thesis_pipeline/`: shared config, prompt formatting, answer extraction, model loading, and supervisor clients.

Generated artifacts are ignored by git: `venv/`, `gsm8k_formatted/`, `checkpoints/`, `gemma4-gsm8k-final/`, `outputs/`, and `reports/`.

## Setup

Use the existing virtual environment or recreate one, then install the requirements. For PyTorch CUDA wheels, use the official PyTorch install command if plain `pip install -r requirements.txt` cannot resolve `+cu124` wheels.

```powershell
.\venv\Scripts\Activate.ps1
python check_env.py
python check_gpu.py
```

Store secrets in environment variables, not in text files:

```powershell
$env:HUGGINGFACE_TOKEN="hf_..."
$env:OPENAI_API_KEY="sk_..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

The existing `Hugging face tokken.txt` file is ignored now. Revoke or remove it if it contains a real token.

## Data

```powershell
python prepare_data.py
```

Optional secondary factuality sample:

```powershell
python prepare_data.py --with-factuality --factuality-limit 100
```

Generate and evaluate the factuality sample separately:

```powershell
python generate.py --task factuality --input-jsonl outputs/factuality_sample.jsonl --run-name fine_tuned_factuality --adapter-dir gemma4-gsm8k-final --limit 25
python evaluate.py --task factuality --predictions fine_tuned=outputs/predictions/fine_tuned_factuality_factuality_test.jsonl --csv-output reports/factuality_summary.csv --md-output reports/factuality_summary.md
```

This factuality metric is a conservative lexical proxy, not the main thesis result. Treat GSM8K exact-answer accuracy as the primary quantitative claim.

## Training

The current checkpoint is incomplete, so the default command resumes from the newest `checkpoints/checkpoint-*` directory.

```powershell
python train.py
```

If a power outage interrupts training, rerun the same command. The script will
pick the newest `checkpoints/checkpoint-*` directory and resume full trainer
state by default:

```powershell
python train.py --resume-mode full
```

For frequent outages, save more often:

```powershell
python train.py --resume-mode full --save-steps 50 --save-total-limit 4 --eval-strategy no
```

Before restarting after a crash, validate the latest checkpoint:

```powershell
python check_checkpoint.py --deep
```

If full resume ever fails because optimizer/RNG state is incompatible after a
package upgrade, salvage the LoRA adapter weights and continue with a fresh
optimizer:

```powershell
python train.py --resume-mode adapter
```

`--eval-strategy no` is intentional on a 16 GB GPU. Hugging Face's epoch-end
token-loss evaluation can require several extra GB of VRAM and may crash even
when training itself is stable. Use `generate.py` and `evaluate.py` after
training for the thesis metric.

Training uses:

- 4-bit NF4 QLoRA;
- answer-only labels, so prompt and padding tokens are ignored;
- `MAX_SEQ_LENGTH=1024`, avoiding the previous 512-token truncation issue;
- LoRA rank 16, alpha 32, dropout 0.05.

The final adapter is saved to `gemma4-gsm8k-final/`.

## Evaluation

Before major runs, add a note to the thesis experiment log:

```powershell
python log_event.py --type milestone --title "Finished Gemma GSM8K QLoRA training" --include-env --detail final_adapter=gemma4-gsm8k-final --detail final_step=1404 --detail note="Trainer epoch eval disabled after CUDA OOM; post-training exact-answer eval used instead."
```

This writes to `reports/experiment_log.md` and `reports/experiment_log.jsonl`.
`generate.py` and `evaluate.py` append entries automatically.

Generate baseline predictions:

```powershell
python generate.py --run-name baseline --limit 5
```

Generate fine-tuned predictions after training:

```powershell
python generate.py --adapter-dir gemma4-gsm8k-final --run-name fine_tuned --limit 5
```

Run full test-set generation by removing `--limit`.

Summarize one or more prediction logs:

```powershell
python evaluate.py `
  --predictions baseline=outputs/predictions/01_answers_untrained_baseline.jsonl `
  --predictions fine_tuned=outputs/predictions/02_answers_finetuned_try1_main.jsonl
```

Reports are written to `reports/gsm8k_summary.csv` and `reports/gsm8k_summary.md`.

## FYDP 3 — Supervised Escalation Cascade

The FYDP 3 pipeline is a cascade, not a plain retry loop. The fine-tuned model
answers on-device; an on-device **gate** decides whether the question is worth
escalating; only escalated questions are sent to a supervisor, which judges the
reasoning and returns a verdict plus an optional hint; the model then retries.

The headline metric is accuracy against **cloud tokens per question**, because
the thesis constraint is edge deployment.

Attempt 1 is never regenerated. It is greedy and deterministic, so
`supervise.py` reads it from `outputs/predictions/02_answers_finetuned_try1_main.jsonl`.
This was verified byte-identical on a 20-example resample, and it saves roughly
five GPU-hours per arm.

### Feedback levels

Hint length is cloud output tokens, which is the cost being minimised, so the
amount of feedback is a measured variable rather than a fixed choice:

| Level | Sent back to the model |
|---|---|
| `L0` | the bare rejection |
| `L1` | plus a pointer naming the step that went wrong |
| `L2` | plus a correction stating the right interpretation |

A level 2 hint never contains the final answer.

### Running it

No-cost smoke test using the gold answer as a deterministic oracle:

```powershell
python supervise.py --provider exact --gate none --limit 5
```

Check escalation counts and projected supervisor calls before committing:

```powershell
python supervise.py --gate length --escalation-rate 0.3 --dry-run
```

Build the sample bank (~10 h) — this single run yields the disagreement gate,
self-consistency voting, the blind-retry control, and the pass@3 ceiling:

```powershell
python generate.py --adapter-dir gemma4-gsm8k-final --run-name samplebank_s1 --temperature 0.7
python generate.py --adapter-dir gemma4-gsm8k-final --run-name samplebank_s2 --temperature 0.7
```

Start the supervisor (a separate window; leave it running):

```powershell
.\scripts\serve_verifier.ps1 -Model glm -Download   # first time only
.\scripts\serve_verifier.ps1 -Model glm
```

Check it before committing hours to it — model id, speed, projected runtime,
and whether it is lenient enough to make the cascade a no-op:

```powershell
python preflight_supervisor.py --provider llamacpp
```

Verdict pass — no retries, so the fine-tuned model is never loaded. Produces
the verifier confusion matrix:

```powershell
python supervise.py --provider llamacpp --gate none --verdict-only
```

One arm per feedback level. Attempt-1 verdicts are replayed from the verdict
cache, so these three arms reject an identical set and hint content is the only
variable between them:

```powershell
python supervise.py --provider llamacpp --gate disagreement --escalation-rate 0.3 `
  --feedback-level 1 `
  --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl `
  --sample-bank outputs/predictions/04_answers_finetuned_try3.jsonl
```

Long runs append and can be resumed after a power cut with `--resume`.

### Supervisor providers

The ladder, weakest first — the point being to measure *how good the judge has
to be*, not to assume a frontier model is required:

| Provider | Judge | Cost |
|---|---|---|
| `self` | the fine-tuned 2B critiques itself — the honest floor | free |
| `local` | a mid-size instruct model via transformers | free |
| `llamacpp` | **a GGUF model behind `llama-server` — the primary supervisor** | free |
| `gemini` | free tier, kept as a small cloud reference arm | ~free |
| `openai` / `anthropic` | paid | paid |
| `exact` | gold-answer oracle — the ceiling | free |

`llamacpp` is the default because it is the only path that can host a quantised
MoE (GLM-4.7-Flash is 30B total but 3B active, and bitsandbytes would need the
60 GB fp16 weights to quantise from), because swapping verifiers is then a
server restart rather than a code change, and because llama.cpp can constrain
decoding to the verdict JSON schema.

Two guards worth knowing about. A missing API key fails at startup rather than
silently accepting every answer. And an empty or truncated reply is recorded as
an **error**, never an acceptance — after `SUPERVISOR_ERROR_ABORT` consecutive
failures the run stops instead of approving everything left.

### Reporting

```powershell
python analyze_supervision.py `
  --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl `
  --cascade L0=outputs/predictions/cascade_gemini_disagreement_L0_test.jsonl `
  --cascade L1=outputs/predictions/cascade_gemini_disagreement_L1_test.jsonl
```

Writes `reports/fydp3_summary.md`: the headline accuracy-vs-cost table, gate
quality curves, verifier precision/recall against ground truth, the flip
analysis (whether retrying broke answers that were already correct), and exact
McNemar significance against the un-supervised baseline.

## Tests

```powershell
python -m unittest discover -s tests
```

These tests cover final-answer extraction and answer normalization, which are central to the reported GSM8K metric.
