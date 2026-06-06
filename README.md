# Gemma 4 GSM8K Hallucination-Reduction Thesis Pipeline

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
  --predictions baseline=outputs/predictions/baseline_gsm8k_test.jsonl `
  --predictions fine_tuned=outputs/predictions/fine_tuned_gsm8k_test.jsonl
```

Reports are written to `reports/gsm8k_summary.csv` and `reports/gsm8k_summary.md`.

## Supervised Retry Loop

For a no-cost smoke test that uses the known GSM8K answer as a deterministic stand-in:

```powershell
python supervise.py --provider exact --limit 5
```

For the thesis supervisor run, use one external provider:

```powershell
$env:SUPERVISOR_PROVIDER="openai"
python supervise.py --adapter-dir gemma4-gsm8k-final --limit 200
```

or:

```powershell
$env:SUPERVISOR_PROVIDER="anthropic"
python supervise.py --adapter-dir gemma4-gsm8k-final --limit 200
```

The JSONL output records every attempt, verifier decision, final accepted answer, retry count, and exact GSM8K correctness.

## Tests

```powershell
python -m unittest discover -s tests
```

These tests cover final-answer extraction and answer normalization, which are central to the reported GSM8K metric.
