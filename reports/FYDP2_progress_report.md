# FYDP 2 — Progress Report

**Project:** Reducing AI Hallucination via Fine-Tuning and Supervised Verification
**Phase:** FYDP 2 of 3 — Fine-Tuning and Baseline-vs-Fine-Tuned Evaluation
**Report date:** 2026-05-16
**Status:** FYDP 2 complete

> This is a data and methodology record, not the final FYDP 2 paper. It documents
> every experiment, decision, and result produced in this phase, so the figures and
> tables here can be carried directly into the written paper.

---

## 1. Project Context and Objective

This thesis investigates a pipeline for reducing unreliable output ("hallucination")
in a small, quantized language model while keeping token usage low. The full project
spans three phases:

- **FYDP 1** — Project setup, dataset selection, and training infrastructure.
- **FYDP 2 (this report)** — Fine-tune the base model on a grade-school math dataset
  to teach chain-of-thought reasoning, then evaluate and compare the fine-tuned model
  against the non-fine-tuned base model.
- **FYDP 3 (future)** — Integrate an external supervisor model that verifies the
  fine-tuned model's reasoning and answer, returning a binary YES/NO decision that
  triggers a retry when the answer is rejected.

**FYDP 2 objective:** Confirm that fine-tuning improves the model's mathematical
reasoning accuracy, quantify that improvement against a fair baseline, and measure the
token cost of each approach. The baseline-vs-fine-tuned comparison is the primary
quantitative result of this phase.

---

## 2. Experimental Setup

### 2.1 Model

- **Base model:** `google/gemma-4-E2B-it` — a 2-billion-parameter, instruction-tuned
  model with a multimodal (image-text-to-text) architecture.
- The model is loaded in **4-bit quantized** form (see Section 3.1) to fit within the
  16 GB GPU.

### 2.2 Dataset

- **Dataset:** `openai/gsm8k`, configuration `main` (GSM8K — Grade School Math 8K).
- **Size:** 7,473 training examples / 1,319 test examples.
- **Task:** Multi-step grade-school arithmetic word problems. Each example contains a
  `question` and an `answer` that includes step-by-step reasoning terminated by a
  `#### <final answer>` marker.
- **Example record:**
  > Question: *"Natalia sold clips to 48 of her friends in April, and then she sold
  > half as many clips in May. How many clips did Natalia sell altogether in April
  > and May?"*
  > Answer: *"Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips
  > altogether in April and May. #### 72"*

### 2.3 Hardware and Software Environment

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER (16 GB VRAM) |
| CPU | AMD Ryzen 7 7700X |
| RAM | 32 GB DDR5 5600 MHz |
| OS | Windows 11 Pro |
| Python | 3.11.9 |
| PyTorch | 2.6.0 + CUDA 12.4 |
| transformers | 5.8.1 |
| trl | 1.4.0 |
| peft | 0.19.1 |
| bitsandbytes | 0.49.2 |
| accelerate | 1.13.0 |
| datasets | 4.8.5 |

---

## 3. Fine-Tuning Methodology

### 3.1 QLoRA Configuration

Fine-tuning used **QLoRA** — Quantized Low-Rank Adaptation — which trains a small set
of adapter weights on top of a frozen, 4-bit-quantized base model.

**Quantization (base model, frozen):**
- 4-bit quantization, NF4 (4-bit NormalFloat) quant type
- Double quantization enabled
- Compute dtype: bfloat16

**LoRA adapter (trainable):**
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention projections) and
  `gate_proj`, `up_proj`, `down_proj` (MLP projections), applied across all
  language-model transformer layers.

### 3.2 Training Procedure

- **Sequence length:** 1,024 tokens maximum.
- **Label masking:** Answer-only. Prompt tokens are set to label `-100` so the
  training loss is computed only over the model's reasoning and final answer, not
  over the question. This focuses the model on producing the correct output format
  and reasoning.
- **Prompt format:** Gemma chat template —
  `<start_of_turn>user\n{question}\n<end_of_turn>\n<start_of_turn>model\n{answer}\n<end_of_turn>`
- **Batch size:** 1 per device, gradient accumulation 16 → effective batch size 16.
- **Optimizer schedule:** Learning rate 2×10⁻⁴, cosine decay, 50 warm-up steps.
- **Duration:** 3 epochs = 1,404 optimizer steps.

### 3.3 Training Outcomes

| Metric | Value |
|---|---|
| Final training loss | 0.09843 |
| Mean token accuracy (training) | 92.4% |
| Eval token accuracy (step 468) | 89.0% |
| Total optimizer steps | 1,404 |
| Epochs completed | 3 |
| Training runtime | ~68 minutes |

The final adapter is saved at `gemma4-gsm8k-final/`.

### 3.4 Training Incident — CUDA Out-of-Memory

During training, Hugging Face's epoch-end token-loss evaluation attempted an
additional 4.38 GiB VRAM allocation, which exceeded the 16 GB GPU at step 936 and
caused an out-of-memory crash. **Resolution:** training was resumed from the last
valid checkpoint (`checkpoint-900`) with in-training evaluation disabled
(`--eval-strategy no`). All final accuracy metrics in this report were therefore
produced by a **separate post-training evaluation pipeline** (`generate.py` +
`evaluate.py`), not by in-training evaluation. This is standard practice when
training on consumer GPUs and does not affect the validity of the results.

---

## 4. Evaluation Methodology

### 4.1 Evaluation Metric

- **Metric:** Exact match on the canonical final answer.
- **Extraction:** The `#### <answer>` marker is located by regular expression; if no
  marker is present, the last number in the output is used as a fallback.
- **Normalization:** Before comparison, answers are normalized — fractions, decimals,
  thousands separators (commas), currency symbols, and percent signs are all handled
  so that, e.g., `72.0`, `72`, and `$72` are treated as equal.
- The extraction and normalization logic is unit-tested (4 tests, all passing).

### 4.2 Decoding Configuration

The same decoding configuration was applied to **both** the baseline and the
fine-tuned model, so the only differences between conditions are the LoRA adapter and
the prompt style.

| Parameter | Value |
|---|---|
| Decoding | Greedy (`temperature = 0`) — deterministic, reproducible |
| Repetition penalty | 1.15 |
| Max new tokens | 512 |
| Stop condition | Generation halts at the `<end_of_turn>` token sequence |

### 4.3 Baseline Prompting — Methodology Iteration

This is an important point of method and is documented here in full.

**First baseline attempt (discarded).** The base model was first evaluated with a
bare chat-template prompt containing the question only, with no instruction or
examples. It scored **1.5% accuracy** (20/1,319 correct), and 566 of 1,319 outputs
produced no extractable answer at all — the base model collapsed into repetition
loops and emitted chat-template control tokens instead of reasoning.

**Why this was rejected as a baseline.** Comparing an uninstructed, untrained model
against a fine-tuned one is not a fair comparison — it measures the model's failure
to guess an unstated output format, not its mathematical ability. Such a baseline
would be a strawman and is not defensible.

**Corrected baseline.** The baseline was re-run using **8-shot chain-of-thought
prompting** — the standard academic method for evaluating a base model on GSM8K
(Wei et al., 2022). Eight worked examples from the training split are prepended to
each test question, demonstrating the chain-of-thought format and the `#### <answer>`
convention through in-context examples. The fine-tuned model is evaluated
**zero-shot**, because fine-tuning has already taught it the format.

The discarded run is preserved as
`outputs/predictions/baseline_bare_prompt_1.5pct.jsonl` as documented evidence of
this methodology iteration.

### 4.4 Token Accounting

To support the project's token-efficiency goal, every generation records both:
- **Prompt tokens** — the input length (large for the 8-shot baseline, small for the
  zero-shot fine-tuned model).
- **Generated tokens** — the output length.

Total token cost per problem = prompt tokens + generated tokens.

---

## 5. Pipeline Engineering and Fixes

The following code-level issues were identified and corrected during FYDP 2
evaluation. They are recorded here because they affect reproducibility and explain
the evaluation behavior.

1. **Repetition penalty.** A `repetition_penalty` of 1.15 was added to generation to
   suppress degenerate repetition loops observed in the base model.

2. **End-of-turn stop condition.** The Gemma 4 tokenizer does **not** represent
   `<end_of_turn>` as a single token — it tokenizes as a 7-token subword sequence
   (`[236820, 643, 236779, 1340, 236779, 887, 236813]`), and
   `convert_tokens_to_ids("<end_of_turn>")` returns an unrelated token (ID 3). The
   original stop logic therefore never triggered, and the model generated the full
   512-token budget on every example. A custom `StoppingCriteria` (`_MultiTokenStop`)
   was implemented to detect the full 7-token end-of-turn sequence. This both
   corrected the token-count metric and substantially reduced generation time.

3. **Token tracking.** Generation now returns a structured result carrying the
   decoded text, the generated-token count, and the prompt-token count.

4. **Few-shot prompting.** A `build_fewshot_prompt()` function and a `--few-shot N`
   command-line option were added to support the corrected 8-shot baseline (Section
   4.3).

5. **Crash-safe generation.** The generation script writes results incrementally
   (one line per example) and only clears the output file on a fresh run, so a run
   interrupted by a power outage can be resumed from where it stopped.

6. **Report clarity.** The summary report omits the supervisor-related columns when
   no supervisor was used, so non-supervised runs are not mislabeled with a "0%"
   supervisor rate.

All unit tests (4) pass after these changes.

---

## 6. Results

### 6.1 Primary Comparison

Evaluation was run on the **full GSM8K test set (1,319 examples)** for both
conditions. Source report: `reports/gsm8k_summary.md`.

| Run | Prompt | Correct / Total | Accuracy | Avg prompt tokens | Avg generated tokens | Avg total tokens |
|---|---|---|---|---|---|---|
| **baseline** | 8-shot | 479 / 1319 | **36.3%** | 1,617.7 | 120.6 | 1,738.3 |
| **fine_tuned** | zero-shot | 737 / 1319 | **55.9%** | 87.7 | 125.9 | 213.5 |

### 6.2 Accuracy

- The fine-tuned model improves accuracy by **+19.6 percentage points**
  (36.3% → 55.9%).
- In absolute terms this is **258 additional correct answers** out of 1,319.
- Relative improvement: **+53.9%** more correct answers than the baseline.
- The fine-tuned model produced an extractable final answer for **all 1,319**
  examples; the baseline failed to produce one for 16 examples.
- Because the baseline is fairly prompted (8-shot chain-of-thought), the gain is
  genuinely attributable to fine-tuning, not to differences in output formatting.

### 6.3 Token Efficiency

- The fine-tuned model's total token cost per problem (213.5) is approximately
  **12.3% of the baseline's** (1,738.3) — an **~8.1× reduction**.
- The two models generate a similar number of *output* tokens (120.6 vs 125.9). The
  efficiency difference comes almost entirely from the **prompt**: the 8-shot
  baseline must carry ~1,618 tokens of in-context examples on every query, while the
  fine-tuned model needs only ~88 tokens because the reasoning format is internalized
  in its weights.
- This is a concrete, measured result: fine-tuning delivers higher accuracy *and*
  lower token cost simultaneously.

### 6.4 Run Times

| Run | Examples | Wall-clock time | Per-example |
|---|---|---|---|
| baseline (8-shot) | 1,319 | 3 h 12 min | 8.72 s |
| fine_tuned (zero-shot) | 1,319 | 5 h 01 min | 13.71 s |

The fine-tuned run is slower per example despite a much shorter prompt. This is
expected: QLoRA inference runs the 4-bit base model **plus** the unmerged LoRA
adapter layers, which adds per-token compute. It is a property of QLoRA inference,
not a defect, and does not affect accuracy.

---

## 7. Analysis

- **The fine-tuning objective for FYDP 2 is met.** Fine-tuning the 2B model on GSM8K
  produced a large, fairly-measured accuracy gain (+19.6 pp) and a large token-cost
  reduction (~8×).
- **Two thesis claims are supported by the data:**
  1. *Reasoning reliability* — fine-tuning substantially increases the proportion of
     correct answers.
  2. *Token efficiency* — the fine-tuned model achieves this at a fraction of the
     baseline's token cost, because it requires no in-context examples.
- **Failure modes observed in the baseline** were genuine reasoning errors — for
  example, mis-scaling numbers or applying the wrong operation — rather than
  format failures, once the model was fairly prompted. This confirms the 8-shot
  baseline measures mathematical ability, as intended.

---

## 8. Scope and Limitations

Recorded honestly so the written paper can frame the contribution accurately.

- **The FYDP 2 result is a reproduction, not a novel finding.** That fine-tuning a
  model on a task's training data improves test performance is well established. The
  value of FYDP 2 is a correctly-measured, fairly-baselined demonstration on this
  specific model and pipeline — it is the foundation for FYDP 3, not a standalone
  research claim.
- **Single deterministic run.** Greedy decoding makes the runs reproducible, but the
  evaluation does not report confidence intervals or multiple seeds.
- **Terminology.** GSM8K errors are arithmetic and reasoning mistakes, not factual
  fabrication. The term "hallucination" should be explicitly defined in the paper to
  mean unreliable or incorrect reasoning output in this context.
- **Absolute accuracy.** 55.9% for a fine-tuned 2B-class model is a reasonable result
  but not a state-of-the-art one; the contribution is the comparison and the
  pipeline, not the absolute score.

---

## 9. Artifacts and Reproducibility

| Artifact | Path | Description |
|---|---|---|
| Final LoRA adapter | `gemma4-gsm8k-final/` | The fine-tuned QLoRA adapter weights |
| Final training checkpoint | `checkpoints/checkpoint-1404/` | Full trainer state at step 1404 |
| Baseline predictions | `outputs/predictions/baseline_gsm8k_test.jsonl` | 1,319 rows, 8-shot |
| Fine-tuned predictions | `outputs/predictions/fine_tuned_gsm8k_test.jsonl` | 1,319 rows, zero-shot |
| Discarded baseline (evidence) | `outputs/predictions/baseline_bare_prompt_1.5pct.jsonl` | Bare-prompt run, 1.5% |
| Comparison report | `reports/gsm8k_summary.md` / `.csv` | Summary table |
| Experiment log | `reports/experiment_log.md` / `.jsonl` | Timestamped record of every run |

Each prediction file records, per example: the question, gold answer, model
prediction, extracted final answers, correctness, generated-token count, and
prompt-token count. The experiment log records the exact command, hyperparameters,
and environment for every run.

---

## 10. Next Phase — FYDP 3

FYDP 3 will integrate the **supervisor layer**: an external high-capability model
(e.g. GPT-5-class or Claude) that reviews the fine-tuned model's full output — the
question, the chain-of-thought reasoning, and the final answer — and returns a binary
YES/NO verdict. A NO triggers the fine-tuned model to retry. The supervisor is
intended to inspect the **reasoning process**, not just the final answer, so that a
correct answer reached through flawed reasoning can still be rejected.

To make the FYDP 3 result defensible, the evaluation should include control
conditions that isolate the supervisor's specific contribution — in particular, a
"retry without supervisor" condition — and should quantify the actual token/cost
trade-off of the supervisor calls rather than assuming one.

**FYDP 2 is complete.** The data in this report is ready to be written into the
FYDP 2 paper.
