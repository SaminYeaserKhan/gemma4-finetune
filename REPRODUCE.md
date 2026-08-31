# Reproducing the experiment

Every command below was actually run to produce the numbers in
`docs/THESIS_DOSSIER.md`. Expected outputs are given so each stage can be checked
before moving on.

Total cost from scratch: roughly **30 GPU-hours**, **25 GB of downloads**, and no
money. Stages 0-2 can be skipped if you already have the trained adapter.

---

## 0. Prerequisites

| | |
|---|---|
| GPU | NVIDIA, **16 GB VRAM** (reference: RTX 4080 SUPER, driver 560.94) |
| System RAM | 32 GB (the supervisor offloads ~7 GB of experts to it) |
| Disk | ~30 GB free |
| OS | Windows 11 + PowerShell (the shell scripts are `.ps1`) |
| Python | 3.11 |

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_env.py
python check_gpu.py
```

Torch must be the CUDA build. If `pip` cannot resolve the `+cu124` wheels, install
them from the official PyTorch index first, then the rest of `requirements.txt`.

Secrets go in environment variables, never in files:

```powershell
$env:HUGGINGFACE_TOKEN="hf_..."     # needed: google/gemma-4-E2B-it is gated
$env:GEMINI_API_KEY="..."           # optional, only for the cloud reference arm
```

---

## 1. Data

```powershell
python prepare_data.py
```

Writes `gsm8k_formatted/`. The test split must contain **1,319** questions.

## 2. Train the solver — ~6 h

```powershell
python train.py
```

QLoRA, 4-bit NF4, rank 16 / alpha 32, `MAX_SEQ_LENGTH=1024`, answer-only label
masking. Auto-resumes from the newest `checkpoints/checkpoint-*` if interrupted.
The adapter lands in `gemma4-gsm8k-final/`.

Epoch-end evaluation is deliberately off (`--eval-strategy no`): it OOMs on 16 GB.
Accuracy is measured after training instead, in stage 3.

## 3. Baseline and fine-tuned accuracy — ~5 h

```powershell
python generate.py --run-name baseline --few-shot 8
python generate.py --adapter-dir gemma4-gsm8k-final --run-name fine_tuned
python evaluate.py `
  --predictions baseline=outputs/predictions/01_answers_untrained_baseline.jsonl `
  --predictions fine_tuned=outputs/predictions/02_answers_finetuned_try1_main.jsonl
```

The 8-shot baseline is the fair comparison — the base model cannot do this task
zero-shot, so a zero-shot baseline would overstate the fine-tuning gain.

**Expected** (`reports/gsm8k_summary.md`):

| Condition | Correct | Accuracy | Avg total tokens |
|---|---|---|---|
| baseline (8-shot) | 479 / 1319 | 0.3632 | 1738.3 |
| fine_tuned (0-shot) | 737 / 1319 | **0.5588** | **213.5** |

`02_answers_finetuned_try1_main.jsonl` is reused as cached attempt 1 by every later stage,
so all arms start from byte-identical answers. Greedy decoding makes this sound;
it was verified byte-identical on a 20-example resample.

## 4. Sample bank — ~10 h

```powershell
python generate.py --adapter-dir gemma4-gsm8k-final --run-name samplebank_s1 --temperature 0.7
python generate.py --adapter-dir gemma4-gsm8k-final --run-name samplebank_s2 --temperature 0.7
```

**Run this before any supervisor work.** One file yields four things: the pass@3
ceiling, the disagreement gate, self-consistency, and the blind-retry control.

```powershell
python analyze_supervision.py `
  --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl `
  --sample-bank outputs/predictions/04_answers_finetuned_try3.jsonl `
  --csv-output reports/fydp3_samplebank.csv --md-output reports/fydp3_samplebank.md
```

**Expected:**

```
local only (1 sample):          737/1319 acc=0.5588
blind retry, take last:         658/1319 acc=0.4989
self-consistency@3 (majority):  779/1319 acc=0.5906
pass@3 CEILING:                 962/1319 acc=0.7293
```

Gate AUC: length **0.6757**, disagreement **0.8404**.

## 4b. Confidence gate — ~20 min

```powershell
python experiments/score_confidence.py --resume
```

Recovers the log-probability the model assigned to each answer it already
wrote — a byproduct of generation that `generate_answer` discards. Scoring
stored text is one forward pass per answer rather than a generation loop, so
this is far cheaper than producing the answers was, and nothing is regenerated.

Gives the third gate, and the one that matters most for the edge argument:
disagreement needs k full generations, so a question takes k times as long to
answer on hardware where generation is already the slow step, while confidence
needs no extra sampling at all. Three measures are recorded from the same pass
— mean, minimum, and final-answer-only log-probability — because which of them
predicts errors best is an empirical question.

Feeds `analyze_supervision.py --confidence`, which is on by default and
silently ignored if the file is absent.

## 4c. The combined gate -- free, and the best one measured

```powershell
.\scripts\score_sample_bank.ps1     # ~40 min, scores the two sampled answer sets
```

Ranking by sample agreement and breaking its ties with the model's own confidence
beats either signal alone, and costs nothing new -- both were already computed:

```
length (free)                                        AUC 0.6757
confidence: final-answer logprob (1 forward pass)     AUC 0.7149
disagreement (k samples)                              AUC 0.8404
combined: disagreement, ties broken by confidence     AUC 0.8685
```

At a 10% escalation budget it catches 125 of the 582 errors against
disagreement's 105, at 0.947 precision. Reported automatically by stage 10
whenever both a sample bank and a confidence file are present.

The combination is strictly lexicographic -- confidence orders questions inside a
disagreement group and can never reorder the groups themselves. Letting it
outweigh agreement measures *worse* than agreement alone. There is no weight
parameter, so there is none that could have been fitted to the test set.

## 5. Install and start the supervisor — ~1 h, mostly downloading

```powershell
.\scripts\install_llamacpp.ps1                      # pinned build b10453, CUDA 12.4
.\scripts\serve_verifier.ps1 -Model glm -Download   # 17.5 GB, verified by byte count
.\scripts\serve_verifier.ps1 -Model glm             # leave this window open
```

`install_llamacpp.ps1` pins the llama.cpp build because the verifier's behaviour is
part of the experimental record. Pick the CUDA variant that matches your driver —
12.4 works on driver 560.x; the 13.x builds need 580+.

If something else is using the GPU, lower the supervisor's footprint:
`.\scripts\serve_verifier.ps1 -Model glm -NCpuMoe 38`.

Check it in a second window before spending hours on it:

```powershell
python preflight_supervisor.py --provider llamacpp
```

**Expected:** ~2.2 s per judgment, ~355 input / ~57 output tokens, **0 malformed
JSON**, and a non-zero rejection rate. A judge that accepts everything makes the
cascade a no-op — stop and change models if that happens.

## 6. Judge prompt calibration — ~1 h

The judging prompt needs calibrating, and it must be calibrated on data the thesis
does not report on:

```powershell
python generate.py --adapter-dir gemma4-gsm8k-final --run-name promptval_train `
  --split train --limit 150
python experiments/judge_prompt_ab.py `
  --predictions outputs/predictions/checks/checker_prompt_tuning_on_training_questions.jsonl
```

**Expected** (held-out train split, 96 correct / 54 wrong):

```
original   precision=0.495  recall=0.926  false_reject=0.531  -> expected net  -3.2
tightened  precision=0.623  recall=0.796  false_reject=0.271  -> expected net  +2.6
```

The original prompt is **net negative**: it rejects so many correct answers that
the retries lose more than they gain. To see why, read the objections themselves:

```powershell
python experiments/inspect_rejections.py --limit 30 --show 4
```

## 7. Verdict pass — ~45 min

```powershell
python supervise.py --provider llamacpp --gate none --verdict-only --resume
```

Judges all 1,319 cached attempt-1 answers. Produces the verifier confusion matrix
and populates `outputs/verdict_cache.jsonl`, so the arms in stage 8 get their
attempt-1 verdicts for free — and provably reject the identical set, which is what
makes the L0/L1/L2 comparison valid. llama.cpp is not bit-reproducible, so without
the cache the arms would differ by *which* questions were retried rather than by
hint content.

## 8. Cascade arms — ~4 h each

```powershell
foreach ($level in 0, 1, 2) {
  python supervise.py --provider llamacpp --gate disagreement --escalation-rate 0.3 `
    --feedback-level $level --resume `
    --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl `
    --sample-bank outputs/predictions/04_answers_finetuned_try3.jsonl
}
```

Interruptible: `--resume` skips ids already written and re-judging is free from the
cache.

**Surviving a power cut.** Rows are appended one per question, so an abrupt loss
costs only the question in flight. Bring everything back with:

```powershell
.\scripts\resume_after_interruption.ps1
```

It restarts the verifier if needed, waits for it to load, reports how far each arm
got, and relaunches with `--resume`. It is idempotent — safe to run when you are
not sure what survived.

A cut during an append can leave a half-written final line, or a run of NUL bytes
where NTFS extended the file before the data reached disk. `repair_jsonl` drops
those rows before the resume scan reads the file and prints what it dropped; the
affected question is simply answered again. Without that, `--resume` would crash on
exactly the failure it exists to recover from.

## 8b. The stacked arm -- no extra compute

The headline result is a recombination of files stage 8 already wrote, so it costs
nothing to produce and is reported automatically by stage 10:

```
| L2            | 784 | 1319 | 0.5944 | 296.0 cloud tokens/q |
| L2 + voting   | 826 | 1319 | 0.6262 | 296.0 cloud tokens/q |
```

The disagreement gate generates 3 samples per question in order to decide what to
escalate. The plain arm then throws their majority away and keeps the greedy answer.
`analyze_supervision.stack_voting` keeps it instead on the questions the gate did *not*
escalate, and keeps the cascade's answer on the ones it did. Supervisor calls are
unchanged -- the escalated set is the arm's own -- so cloud cost is identical.

Against free self-consistency (`reports/fydp3_summary.md` section 5b), the plain arms are
not significant (p=0.062 / 0.275 / 0.751) and the stacked arms all are
(p=0.034 / 0.0018 / <0.00001).

The natural threshold for a 3-sample gate is 37.3%, not 30%: the score takes only the
values 0, 1/3 and 2/3, and 492 of 1,319 questions sit at 2/3. Escalating that whole group:

```powershell
.\scripts
un_stacked_arm.ps1
```

~4.5 h. Attempt-1 verdicts for the 396 questions already judged replay free from the
cache, so it costs ~96 new judgments.

## 9. Verifier ladder — how strong must the judge be?

```powershell
python supervise.py --provider self --gate none --verdict-only          # the 2B floor
.\scripts\serve_verifier.ps1 -Model qwen                                # middle rung
python supervise.py --provider llamacpp --supervisor-model Qwen3.5-9B-UD-Q4_K_XL `
  --gate none --verdict-only
python supervise.py --provider gemini --gate none --verdict-only --limit 300 --rpm 10
python supervise.py --provider exact --gate none --verdict-only         # the oracle ceiling
```

The `exact` oracle must score precision 1.000, recall 1.000, false-reject 0.000. If
it does not, the scoring code is broken, not the judge.

## 10. Final analysis

```powershell
python analyze_supervision.py `
  --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl `
  --sample-bank outputs/predictions/04_answers_finetuned_try3.jsonl `
  --cascade L0=outputs/predictions/10_pipeline_glm30b_hint_none.jsonl `
  --cascade L1=outputs/predictions/11_pipeline_glm30b_hint_short.jsonl `
  --cascade L2=outputs/predictions/12_pipeline_glm30b_hint_full.jsonl
```

Writes the headline accuracy-vs-cloud-cost table, gate curves, the verifier
confusion matrix, the flip matrix, and exact McNemar significance against the
un-supervised baseline.

---

## Tests

```powershell
python -m unittest discover -s tests
```

**120 tests**, covering answer extraction and normalisation, the gates, retry-prompt
construction, the verdict cache, log-probability extraction, the stacked arm, and the supervisor
client's failure modes — including that a truncated or empty reply becomes an error
rather than an approval.

---

## What is not in this repository

| Missing | Why | How to get it |
|---|---|---|
| `gemma4-gsm8k-final/` | LoRA adapter weights, too large for git | stage 2 (~6 h) |
| `checkpoints/` | training checkpoints | stage 2 |
| `gsm8k_formatted/` | derived from HF | stage 1 (~1 min) |
| `*.gguf` | 17.5 GB | stage 5 |
| llama.cpp binaries | ~640 MB | stage 5 |

`outputs/predictions/*.jsonl` and `reports/` **are** committed, so the analysis can
be re-run and the reported numbers checked without any GPU.

## Known reproducibility limits

- **llama.cpp is not bit-reproducible.** The same prompt and model at temperature 0
  gave false-reject rates of 0.500 and 0.417 across two runs. Expect drift in the
  third decimal; the effects reported are much larger than that.
- **Solver generation is deterministic.** Greedy decoding with a fixed adapter
  reproduces byte-identically, which is what licenses attempt-1 caching.
- **Hosted models are not pinned.** Free-tier Gemini models get swapped without
  notice, which is why the primary supervisor is a self-hosted GGUF with its file
  name recorded. Every run logs the exact model id to `reports/experiment_log.md`.
