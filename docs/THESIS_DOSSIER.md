# Thesis Dossier — Edge-Constrained Supervised Cascade for GSM8K

> **Writing the paper? Read `docs/PAPER_GUIDE.md` instead.** It covers the same
> material in plain language, with a glossary, a section-by-section plan for the
> report, and the questions to expect at the defence. This document is the
> technical record: it exists so every number can be traced to the file and
> command that produced it. Use it to check a figure, not to learn the project.
>
> **Prior work lives in `docs/RELATED_WORK.md`**, with `docs/references.bib` for
> the citations. This file deliberately holds only our own measurements; when a
> result here needs positioning against published work, that file does it.


**Living document. Last updated: 2026-08-29.**

This is the write-up source of truth: every measured number with its provenance, every
design decision with the reason it was made, every known weakness, and the answers to
the questions a defence panel is likely to ask. Nothing here is a plan — plans live in
`.claude/plans/`. Everything here either happened or is explicitly labelled as
projected.

**Rule for this file:** every quantitative claim names the file or command it came
from. If a number cannot be traced, it does not belong here.

---

## 1. Research questions

The project has one framing constraint and four questions under it.

> **Constraint.** The solver must run on a low-power edge device (phone-class). That
> forces a small, heavily quantised model, which costs accuracy. Cloud help is
> available but is the expensive resource being minimised.

| ID | Question | Status |
|---|---|---|
| **RQ1** | How much accuracy does QLoRA fine-tuning recover for a 4-bit 2B model on GSM8K, and at what token cost? | **Answered** (FYDP 2) |
| **RQ2** | Can the device tell, on its own, when it is likely wrong — well enough to escalate rarely? | **Answered** (§5.3) |
| **RQ3** | Does a supervisor's verdict add anything beyond simply retrying, or beyond free self-consistency? | **Partly answered** — the controls are measured (§5.4), the cascade arms are not yet run |
| **RQ4** | Does the *content* of the feedback matter? Is a pointer (L1) or a correction (L2) better than a bare "NO" (L0), per cloud token spent? | **Open — the central question** |
| **RQ5** | How strong does the judge have to be? Does a self-hosted open-weight verifier match a frontier one? | **Open** (verifier ladder, §7) |

**RQ4 is the thesis.** RQ1-RQ3 establish that the question is well-posed and that the
measurement apparatus is trustworthy.

### 1.1 The sharpened form of RQ4

§5.4 measured that a blind resample fixes a wrong answer **24.2%** of the time. So RQ4
reduces to a single falsifiable comparison:

> Does supervisor feedback raise the repair rate above 24.2%, and by enough to beat
> **self-consistency@3 at 59.06%**, which costs zero cloud tokens?

Both outcomes are publishable. If hints raise the repair rate materially, the cascade
occupies a Pareto point neither endpoint reaches. If they do not, the honest finding is
that self-consistency is the better edge strategy — a negative result the project's own
control group proves rather than assumes.

---

## 2. What was built

```
                    ON DEVICE                    │        ESCALATED ONLY
                                                 │
  question ─► Gemma-4-E2B-it (4-bit QLoRA) ─► answer
                        │                        │
                        ├─► gate (free/cheap) ───┼─► supervisor ─► verdict + hint
                        │                        │        │
                        └─► answer stands        │        └─► retry with feedback
                            (no cloud cost)      │
```

| Component | Where | Role |
|---|---|---|
| Solver | `gemma4-gsm8k-final/` | 4-bit NF4 QLoRA adapter over `google/gemma-4-E2B-it` |
| Gate | `thesis_pipeline/gate.py` | Decides what escalates. Pure functions; thresholds swept at analysis time, never baked in |
| Supervisor | `thesis_pipeline/supervisor_client.py` | Judges reasoning, returns verdict + pointer + correction |
| Verdict cache | `thesis_pipeline/verdict_cache.py` | Replays identical judgments — see §6.4 |
| Orchestrator | `supervise.py` | cached attempt 1 → gate → verdict → retry |
| Analysis | `analyze_supervision.py` | Every reported table |
| Preflight | `preflight_supervisor.py` | Validates a judge before spending hours on it |

Feedback levels, which are the RQ4 treatment arms:

| Level | Sent back to the solver | Measured output tokens |
|---|---|---|
| L0 | the bare rejection | ~12 |
| L1 | + a pointer naming the faulty step | ~30 |
| L2 | + a correction stating the right interpretation | ~57 (§5.5) |

A level-2 hint never contains the final answer — enforced by stripping `####` markers
in `_clean_hint`, and tested (`test_strips_leaked_final_answer_from_hint`).

---

## 3. Environment of record

Pin these in the write-up; several results depend on them.

| | |
|---|---|
| GPU | NVIDIA RTX 4080 SUPER, 16,376 MiB |
| System RAM | 31.1 GB |
| Solver stack | transformers 5.8.1, torch 2.6.0+cu124, peft 0.19.1, bitsandbytes 0.49.2 |
| Verifier runtime | llama.cpp build **b10453**, CUDA 12.4 Windows x64 |
| Verifier weights | `GLM-4.7-Flash-UD-Q4_K_XL.gguf`, **17,520,169,312 bytes** (verified against the published size) |
| Dataset | GSM8K `main`, test split, **1,319** questions |

---

### 3.1 Full configuration of record

Everything a methods section needs. Defaults are in `thesis_pipeline/config.py`;
each is overridable by the matching environment variable, and every run logs the
values it used to `reports/experiment_log.jsonl`.

**Fine-tuning (QLoRA)**

| Setting | Value |
|---|---|
| Base model | `google/gemma-4-E2B-it` (2B, multimodal) |
| Quantisation | 4-bit NF4, double quantisation, bfloat16 compute |
| LoRA rank / alpha / dropout | **16 / 32 / 0.05** |
| LoRA targets | `model.language_model.layers.N.(self_attn\|mlp)` — the Gemma 4 module path differs from Gemma 2 |
| Max sequence length | 1024 |
| Epochs | 3 (completed: step 1404, epoch 3) |
| Batch size / gradient accumulation | 1 / 16 (effective batch 16) |
| Learning rate / warmup | 2e-4 / 50 steps |
| Label masking | Answer-only (the prompt is masked out of the loss) |
| Final train loss / token accuracy | 0.0984 / 0.924 |
| Training wall clock | **4,080 s (68 min)** |
| Epoch-end evaluation | Disabled — it OOMs on 16 GB; accuracy is measured after training instead |

**Inference**

| Setting | Value |
|---|---|
| Attempt 1 | Greedy (temperature 0), max 512 new tokens |
| Extra samples | Temperature 0.7, 2 of them |
| Retries | Temperature 0.7, `retry_limit` 2 (so up to 3 attempts) |
| Escalation rate | 0.30 default; 0.373 is the disagreement gate's natural threshold |
| Supervisor output cap | 200 tokens default; **512 required for Qwen** (§5.5.2a) |

**Hardware**

| | |
|---|---|
| GPU | NVIDIA RTX 4080 SUPER, 16 GB, driver 560.94 |
| System RAM | 32 GB (the 30B checker offloads ~7 GB of experts to it) |
| OS / Python | Windows 11, Python 3.11 |

**Data.** GSM8K, official split. Train 7,473 / test **1,319**. Every reported
number is the full test split. The checker's prompt was calibrated on 150
**train**-split questions and never on test (§5.6).

## 4. Results ledger — FYDP 1-2

Source: `reports/gsm8k_summary.md`.

| Condition | Correct | Accuracy | Avg prompt tok | Avg gen tok | Avg total tok |
|---|---|---|---|---|---|
| 8-shot base model (fair baseline) | 479 / 1319 | **36.32%** | 1617.7 | 120.6 | 1738.3 |
| QLoRA fine-tuned (0-shot) | 737 / 1319 | **55.88%** | 87.7 | 125.9 | **213.5** |

**+19.6 accuracy points at 12.3% of the token cost.** The token reduction is a real
part of the contribution and is easy to under-sell: the baseline needs 8 in-context
exemplars, the fine-tuned model needs none.

---

## 5. Results ledger — FYDP 3

### 5.1 Measurement apparatus validated

| Check | Result | Source |
|---|---|---|
| Attempt-1 caching is sound | 20/20 regenerated predictions **byte-identical**, token counts identical | determinism check, 2026-08-05 |
| Analysis reproduces the known baseline | `737/1319 acc=0.5588` exactly | `analyze_supervision.py` |
| Oracle verifier scores perfectly | precision 1.000, recall 1.000, false-reject 0.000 over all 1,319 | `reports/smoke_fydp3.md` §3 |
| Verdict cache replays | 20 new calls → 0 new, 20 replayed | §6.4 |
| Test suite | **74 tests passing** | `python -m unittest discover -s tests` |

Attempt-1 caching is what makes every arm exactly paired: all arms start from
byte-identical answers, so differences between them cannot come from decoding noise.

### 5.2 Error profile

- Only **1 of 1,319** outputs lacks the `#### N` marker — there is no format-failure
  story to tell; extraction is not the bottleneck.
- Wrong answers average **146** generated tokens vs **110** for correct ones. This is
  what makes the free length gate work at all.
- The errors are **comprehension errors, not arithmetic errors**: the model misreads a
  phrase ("half that much", "3 times a week") and then reasons correctly from the
  misreading. Its `<<2*2=4>>` calculator steps are internally consistent.

This grounds the RQ4 hypothesis: a bare "NO" should reproduce the same misreading,
while a pointer naming the misread phrase should not.

### 5.3 Gate quality (RQ2) — measured

Source: `reports/fydp3_samplebank.md` §2, from
`python analyze_supervision.py --sample-bank samplebank_s1 --sample-bank samplebank_s2`
(2026-08-28). Confidence scores from `experiments/score_confidence.py` over all 1,319
attempt-1 answers, written to `outputs/predictions/05_confidence_for_try1.jsonl`.

Ordered by what the signal costs the device, which is the axis this thesis argues on:

| Gate | AUC | Device cost |
|---|---|---|
| Output length | 0.6757 | free — already known after generating |
| Confidence: min logprob | 0.6699 | 1 forward pass over text already written |
| Confidence: mean logprob | 0.7087 | 1 forward pass |
| Confidence: final-answer logprob | 0.7149 | 1 forward pass |
| **Sample disagreement (k=3)** | **0.8404** | **2 extra full generations** |

At a **30% escalation rate** (396 of 1,319 questions escalated, 582 wrong in total):

| Gate | Errors caught | Recall | Precision |
|---|---|---|---|
| Length | 249 / 582 | 42.8% | 62.9% |
| Confidence: min logprob | 239 / 582 | 41.1% | 60.4% |
| Confidence: mean logprob | 262 / 582 | 45.0% | 66.2% |
| Confidence: final-answer logprob | 267 / 582 | 45.9% | 67.4% |
| **Disagreement** | **315 / 582** | **54.1%** | **79.6%** |

**RQ2 is answered affirmatively.** At 30% escalation the disagreement gate sends 4
genuinely-wrong answers to the cloud for every 1 correct one. The device can tell when
it is confused, using only local computation.

**The confidence gate does not replace disagreement.** It was added to test whether the
strong gate's cost could be avoided: log-probabilities are a discarded byproduct of
generation, so scoring them needs one forward pass over text the model has already
written, against the two extra full generations disagreement requires. It closes about a
quarter of the gap between length and disagreement (0.676 → 0.715 → 0.840) and catches
18 more errors than length at the same escalation rate. That is a real improvement for
no extra sampling, but disagreement still finds 48 more errors than the best free gate,
so the k-sample cost buys something the model's own certainty cannot supply.

Two results here were not what was predicted before the run, and are recorded as such:

- **Minimum-token log-probability is the worst gate of the five**, below even output
  length. The "weakest link" intuition — that one improbable token marks the bad step —
  does not hold. A single low-probability token turns out to be routine in fluent text
  and carries no signal about arithmetic being wrong.
- **Final-answer log-probability is the best of the three confidence measures**, which
  was predicted to be useless. Its absolute values are tiny (≈ −0.0003 on early
  examples) because writing `#### 18` after computing 18 is near-deterministic copying.
  The prediction confused a small mean with a small *variance*: the model does hesitate
  measurably at that token when its own working has not converged, and ranking on that
  hesitation is informative. Because AUC is rank-based, the scale of the values is
  irrelevant — only their ordering is.

**Why this strengthens the edge argument.** The target hardware is any machine without a
serious GPU -- ageing laptops, office desktops, phones -- where generation is already the
slow step. Sampling k=3 times therefore triples the time to answer a single question,
which is the real cost of the strong gate, not a hardware-purchase cost. The thesis can
now report a three-point cost curve rather than a two-point one, and state the trade-off
concretely: a deployment that cannot afford three generations per question still gets a
gate at AUC 0.715, retaining roughly 85% of the errors-caught that the expensive gate
finds. The main cascade arms use disagreement, because the measured gap is large enough
to matter and the sample bank already exists.

### 5.3.1 The combined gate (RQ2, extended) — MEASURED 2026-08-29

Proposed by the author: rank questions by how often the samples agree *and* by
how confident the model was, rather than by agreement alone. Measured on data
already on disk — no new generation, no supervisor calls.

| Gate | Device cost | AUC |
|---|---|---|
| length | free | 0.6757 |
| confidence (final answer) | 1 forward pass | 0.7149 |
| disagreement | k generations | 0.8404 |
| **combined: disagreement, ties broken by confidence** | **k generations** | **0.8685** |

Errors caught at a fixed escalation budget, which is the number that matters —
the thesis argument is about sending *few* questions:

| Escalated | Disagreement | Combined | Gain | Combined precision |
|---|---|---|---|---|
| 5% (66) | 53 | **63** | +19% | 0.955 |
| 10% (132) | 105 | **125** | +19% | 0.947 |
| 20% (264) | 214 | **230** | +7% | 0.871 |
| 30% (396) | 315 | **325** | +3% | 0.821 |

At a 10% budget the combined gate catches a fifth more errors for the same
cloud spend, and **95% of what it escalates is genuinely wrong** — which also
means the judge's false rejections have far fewer correct answers to damage.
The ceiling accuracy the cascade could reach at 30% escalation rises from
0.7574 to 0.8052.

**Why it must be lexicographic.** Letting confidence outweigh agreement scores
*worse* than agreement alone (AUC 0.860 at weight 2.0 against 0.840). The
combination therefore ranks strictly by disagreement and uses confidence only
inside a tie group: `tie_broken_score` compresses the rank-normalised secondary
to fit inside the smallest gap between distinct primary values, so it can never
reorder them. There is no weight parameter — which also means there is no
weight that could have been tuned on the test set.

**It repairs the resolution problem in §5.7.3.** Disagreement over k=3 samples
takes three values, so only four escalation rates were expressible and 30% had
to cut a tie group by index. The combined gate is continuous, so any operating
point is now available and the arbitrary cut is gone.

**Cost: nothing.** Both signals were already computed — disagreement to run the
gate, confidence for §5.3. The un-combined gate was discarding one of them.
Implemented as `gate.tie_broken_score`, tested in
`tests/test_gate.py::TieBrokenScoreTests` and
`tests/test_supervise_gate.py`; available as `supervise.py --gate combined`.

### 5.3.2 Confidence-based answer selection — MEASURED 2026-08-29, NULL RESULT

The second half of the same proposal: instead of always keeping the first (greedy)
answer, pick among the three samples using confidence as well as vote count.
Required scoring the two sampled answer sets
(`scripts/score_sample_bank.ps1`, ~40 min, `confidence_samplebank_s{1,2}_test.jsonl`).

| Selection rule | Correct | Accuracy | On the 492 all-differ |
|---|---|---|---|
| always the first answer | 737 | 0.5588 | 99 / 492 |
| majority vote | 779 | 0.5906 | 99 / 492 |
| highest confidence only | 761 | 0.5770 | 109 / 492 |
| confidence-weighted vote | 679 | 0.5148 | 109 / 492 |
| **vote first, confidence breaks ties** | **789** | **0.5982** | 109 / 492 |
| perfect oracle pick | 962 | 0.7293 | 239 / 492 |

The same ordering principle wins here as in §5.3.1: **agreement must lead and
confidence may only break its ties.** Letting confidence lead loses 18 answers;
weighting the vote by confidence loses 100. This is now measured twice on
independent uses of the two signals, which is worth stating as a finding in its
own right rather than as a tuning detail.

**But the gain is not significant, and it is worth nothing to the cascade.**
Against plain voting: 51 fixed, 41 broken, net +10, exact McNemar **p=0.348**.
And the full stacked system scores **843/1319 either way — identical.**

The reason is mechanical and complete. Confidence can only change the chosen
answer when the vote does not already settle it:

- all three agree (427 questions) — every candidate is the same answer, so the
  tie-break is a no-op;
- two agree (400) — the winning cluster's members carry the same answer, so
  again a no-op;
- all three differ (492) — the only place it acts, worth +10.

Those 492 are exactly the questions the gate escalates, and the supervisor's
retry replaces the answer anyway. **The rule improves precisely the subset whose
answer the cascade overwrites.** It is therefore reported as a negative result:
confidence-based answer selection is a small, non-significant improvement to
majority voting alone, and no improvement at all to the system that was built.

One use survives and is untested: the cascade currently sends the supervisor the
*first* answer, which is correct on 99 of those 492, where the
highest-confidence answer is correct on 109. Sending a better candidate might
change what the supervisor does with it, but that cannot be measured from stored
data — the retry depends on what was sent — and the expected effect is ~10
questions, so it is recorded here rather than run.

### 5.4 The controls (RQ3) — measured

Source: `reports/fydp3_samplebank.md` §1, from `samplebank_s1`/`s2` (temperature 0.7,
full test set).

| Condition | Correct | Accuracy | Cloud cost |
|---|---|---|---|
| Local only, 1 sample | 737 | **55.88%** | 0 |
| Blind retry, take last | 658 | **49.89%** | 0 |
| Self-consistency@3, majority vote | 779 | **59.06%** | 0 |
| **pass@3 ceiling** | 962 | **72.93%** | — |

Per-sample flip matrix, measured directly against attempt 1:

| Resample | Broke a correct answer | Fixed a wrong answer | Net |
|---|---|---|---|
| s1 | 229 / 737 = **31.1%** | 141 / 582 = **24.2%** | −88 |
| s2 | 217 / 737 = **29.4%** | 138 / 582 = **23.7%** | −79 |

Three findings, in order of importance:

1. **Blind retrying costs 6 accuracy points.** This is the control that makes the whole
   thesis meaningful: the improvement cannot be obtained by simply asking again, so any
   gain the cascade shows is attributable to the verdict.
2. **The ceiling is 72.93%.** No verifier, hint, or prompt can exceed this — if the
   model cannot produce the right answer in 3 samples, nothing recovers it. All results
   should be read against 55.88% floor / 72.93% ceiling, a 17.05-point band.
3. **Resampling is asymmetric and hostile**: it breaks correct answers (30%) more often
   than it fixes wrong ones (24%). Every unnecessary retry is expected to lose accuracy.
   This is why judge precision matters more than judge recall here.

### 5.5 The supervisor (GLM-4.7-Flash) — measured

Preflight, 30 questions, `preflight_supervisor.py --provider llamacpp`:

| | |
|---|---|
| Throughput | **2.22 s / judgment** → ~0.8 h for a full 1,319-question pass |
| Tokens | **355 input / 57 output** per judgment |
| Schema compliance | **0 / 30 malformed** — grammar-constrained decoding held |
| Errors | 0 |

Note the throughput was measured while the GPU was *shared* with a generation run and
the verifier was confined to ~5 GB of VRAM. It is a lower bound.

**Full verdict pass — all 1,319 questions** (`reports/fydp3_verdict.md` §3,
`supervise.py --provider llamacpp --gate none --verdict-only`, 1,319 calls):

| | Judged wrong | Judged right |
|---|---|---|
| **Actually wrong** (582) | 486 true reject | 96 false accept |
| **Actually right** (737) | 188 false reject | 549 true accept |

| Precision | Recall | False-reject rate | Judge accuracy |
|---|---|---|---|
| 0.7211 | 0.8351 | 0.2551 | 0.7847 |

These land within noise of the 150-question estimates in §5.6 (0.720 / 0.868 / 0.280),
so the calibration sample was representative and the small-sample decision was sound.

### 5.5.1 Verifier ladder, rung 1: can the 2B check its own work? — MEASURED 2026-08-29, NO

`preflight_supervisor.py --provider self --limit 30`, log at `outputs/preflight_self.log`.
The floor of the ladder: the fine-tuned solver marking its own homework, which if it
worked would remove the cloud supervisor from the design entirely.

| | 2B judging itself | GLM-4.7-Flash (30B-A3B) |
|---|---|---|
| Recall (share of wrong answers caught) | **0.222** | 0.835 |
| False-reject rate (correct answers broken) | **0.417** | 0.255 |
| Precision | 0.444 | 0.721 |
| Seconds per judgment | **13.04** | 2.22 |
| Malformed JSON | **19 / 30** | 0 / 30 |
| Projected full pass | 4.8 h | 0.8 h |

*(The 2B row is a 30-question preflight and its precision in particular has wide error
bars; the GLM row is the full 1,319-question pass from §5.5. Precision is also base-rate
sensitive and the two samples differ in composition. Recall and false-reject rate are the
comparable columns, and the gaps there are far too large to be sampling noise.)*

**The answer is no, on every axis at once.** The 2B catches roughly a quarter as many
errors as GLM while breaking correct answers **1.6× more often**. A judge that rejects
42% of what is already right and finds 22% of what is wrong cannot pay for itself: §5.7
measured that only ~25% of correct rejections become repairs, so the damage arrives at
full strength while the benefit is discounted fourfold. The full 4.8-hour pass was not
run — the preflight exists precisely to make that decision cheaply, and it did.

**The most useful finding here is the speed column, and it is counter-intuitive.** The
2B model is **6× slower per judgment than the 30B**. Size is not the cost axis. GLM-4.7-Flash
activates only 3B of its 30B parameters per token and is served by llama.cpp with
grammar-constrained decoding; the 2B runs through transformers with bitsandbytes 4-bit
and generates 122 unconstrained output tokens against GLM's 57. **The serving stack and
the active-parameter count dominate the parameter count.** Any claim in this thesis about
"a smaller judge is cheaper" must be qualified accordingly — measured here, it was false.

The malformed-JSON column is the same story from the other side: GLM's 0/30 is not model
quality, it is llama.cpp grammar constraint. The 2B path has no such constraint and
produced unparseable output 63% of the time. `parse_verdict` treats an unreadable reply
as *accept* (§6.8), so those 19 responses became silent approvals — which is itself part
of why recall is so low, and a reason the `self` rung would need grammar-constrained
serving before its quality could be judged fairly.

**Rung 2 (Qwen3.5-9B) is the remaining open question**, and is the one that can actually
answer RQ5, since it shares GLM's serving stack and so isolates model strength from
infrastructure. Weights downloaded 2026-08-29
(`Qwen3.5-9B-UD-Q4_K_XL.gguf`, 5,966,095,584 bytes).

**Config bug found and fixed while doing this.** The `qwen` preset in
`scripts/serve_verifier.ps1` named a file that does not exist in the repository
(`Qwen3.5-9B-Q4_K_XL.gguf`; the real name carries unsloth's `UD-` prefix). Anyone
following REPRODUCE.md stage 9 would have hit a 404. Corrected, with the byte count
recorded so the download self-verifies as the GLM one does. The `glm-reap` preset was
checked against the Hub at the same time and is correct.

### 5.5.2 The verifier ladder (RQ5) — MEASURED 2026-08-29, full passes

Three judges, each over all 1,319 cached attempt-1 answers, same judging prompt,
same ground truth. Qwen: `outputs/predictions/08_checker_qwen9b_marks_all.jsonl`,
`reports/fydp3_verdict_qwen.md`. GLM: §5.5.

| Judge | Errors caught (of 582) | Correct answers broken (of 737) | Precision | Recall | False-reject | Judge accuracy | s / judgment |
|---|---|---|---|---|---|---|---|
| 2B, self *(30-q preflight)* | — | — | 0.444 | 0.222 | 0.417 | — | 13.04 |
| **Qwen3.5-9B** | **517** | **137** | **0.7905** | **0.8883** | **0.1859** | **0.8469** | **1.23** |
| GLM-4.7-Flash 30B-A3B | 486 | 188 | 0.7211 | 0.8351 | 0.2551 | 0.7847 | 2.22 |

**The 9B beats the 30B on every measure, and is 1.8× faster.** It catches **31 more
errors** and breaks **51 fewer correct answers**. There is no axis on which the larger
model wins. Judge accuracy 0.847 against 0.785.

**This overturns RQ5's expected answer.** The ladder was built to find how strong a judge
must be, on the assumption that quality rises with size and the question was where it
saturates. It does not rise monotonically. Between 2B and 9B it rises sharply — recall
0.222 → 0.888. Between 9B and 30B it *falls* on every measure. The useful claim is
therefore not "bigger is better up to a point" but **"there is a size that fits the task,
and 30B overshoots it"**: GLM's extra capacity shows up as over-rejection (0.255 against
0.186), which is the failure mode §5.6 established as the expensive one, since a broken
correct answer costs a repair that never had to happen.

**GLM should be replaced by Qwen3.5-9B as the primary supervisor.** It is a better judge,
1.8× faster, and a 5.6 GB file against 16.3 GB — which materially changes the deployment
story, since the checker now fits comfortably beside the solver on a 16 GB card.

**Consequence for §5.7 that must be stated plainly: every cascade arm in this thesis used
the weaker judge.** The measured 0.6391 is a floor, not a ceiling. Re-running the best
configuration (stacked, combined gate) against Qwen is the single highest-value experiment
remaining (~3.5 h) and is expected to improve on it, since the cascade's gain is bounded by
the share of true rejections that become repairs and Qwen supplies 31 more true rejections
while wasting 51 fewer retries on answers that were already right.

#### 5.5.2a A preflight this size can invert the answer — a methods warning

The 30-question preflight measured Qwen's false-reject rate at **0.417**. The full pass
measured **0.1859**. On that basis §5.5.2 was first written to say Qwen was *"not
recommended as the primary supervisor"*. The full pass says the opposite, and the
recommendation above is the reversed one.

The preflight was not wrong to run — it correctly screened out the 2B in fifteen minutes
and correctly said Qwen was worth an hour. But **a 30-sample estimate of a rate near 0.2
has a 95% interval roughly ±0.15**, which is wide enough to move a decision from "reject"
to "adopt". The rule this thesis should state: **preflight results decide whether to spend
the GPU time, never what to conclude.** No number from a `--limit 30` run appears in this
dossier's results without the sample size beside it, and none should reach the report.

The truncation finding from that preflight does survive, because it is a mechanism rather
than a rate: at the default 200-token budget Qwen truncated 5 of 30 replies, and a
truncated reply becomes an error which §6.8 converts to an *accept*. The full pass ran at
`SUPERVISOR_MAX_OUTPUT_TOKENS=512`, which `scripts/run_verdict_pass.ps1` now sets by
default. Qwen averaged 97 output tokens against GLM's 57, so the larger budget is required
rather than merely prudent.

#### 5.5.2b The cache-corruption hazard, and the guard added for it

`verdict_key` hashes `provider | model | system_prompt | question | candidate_answer`.
Judging with Qwen while the config still names GLM writes Qwen's verdicts under GLM's key
and silently corrupts `outputs/verdict_cache.jsonl` — which every arm in §5.7 replays from,
and which is the control that makes L0/L1/L2 comparable. This is easy to trigger:
`preflight_supervisor.py` was observed printing `model: GLM-4.7-Flash-UD-Q4_K_XL` while
Qwen was answering, because the client reports the *configured* name, not the served one.

`scripts/run_verdict_pass.ps1` now queries `/v1/models` and refuses to start unless the
served model matches the requested one, so the mistake cannot be made by hand. The cache
was audited before and after the Qwen pass and is intact.

A residual weakness: cache rows *hash* the model id without *storing* it, so corruption of
this kind could not be detected afterwards by inspection — only by arms mysteriously
disagreeing. Storing the model id per row is item 1b in §7.

**Config bug found and fixed.** The `qwen` preset in `scripts/serve_verifier.ps1` named a
file absent from the repository (`Qwen3.5-9B-Q4_K_XL.gguf`; the real name carries unsloth's
`UD-` prefix). Anyone following REPRODUCE.md stage 9 would have hit a 404. Corrected, with
the byte count (5,966,095,584) recorded so the download self-verifies. `glm-reap` was
checked against the Hub at the same time and is correct.

### 5.6 Judge prompt calibration — measured, and a correction

**The first judging prompt was defective and would have sunk the result.** It told the
judge to reject any step that was "wrong, unjustified, or misreads the question".
GLM-4.7-Flash applied that hyper-literally and fabricated objections with no content:

> *"The four apples cost \$1.50 x 4 = \$6.00, not \$6."*
> *"the cost of one pair of shoes is \$42.00, not \$42"*

and, on another question, a pointer that quoted the model's own correct line back as
the error. A/B over **150 test questions** (82 correct, 68 wrong):

| Prompt | Rejected | Precision | Recall | **False-reject rate** |
|---|---|---|---|---|
| Original | 111 | 0.595 | 0.971 | **0.549** — rejected 45 of 82 correct answers |
| **Tightened (adopted)** | 82 | 0.720 | 0.868 | **0.280** — rejects 23 of 82 |

The tightened prompt names a closed list of real defects and explicitly excludes
formatting, units, trailing zeros, and any objection that would not change a number.
It is in `supervisor_client.SYSTEM_PROMPT` with these figures recorded above it.

**Why this is the right trade, quantitatively.** Using the §5.4 rates, the tightened
prompt wins whenever `p_break / p_fix > 0.318`. Measured, that ratio is
`0.30 / 0.242 = 1.26` — four times over the threshold. Expected net gain per 150
questions: original ≈ +2.3, tightened ≈ +7.3.

**Non-determinism observed.** The same prompt and model at temperature 0 scored
false-reject 0.500 in one run and 0.417 in another (n=30). llama.cpp is not
bit-reproducible. This is the concrete justification for the verdict cache (§6.4) and
should be disclosed.

**Contamination check — passed.** The prompt was originally selected on test-split
questions, which would have been mild test-set contamination. It was therefore re-run on
150 **train-split** questions (`outputs/predictions/checks/checker_prompt_tuning_on_training_questions.jsonl`,
96 correct / 54 wrong) that played no part in the selection:

| Prompt | Split | Precision | Recall | False-reject |
|---|---|---|---|---|
| Original | test | 0.595 | 0.971 | 0.549 |
| Original | **train (held out)** | 0.495 | 0.926 | **0.531** |
| Tightened | test | 0.720 | 0.868 | 0.280 |
| Tightened | **train (held out)** | 0.623 | 0.796 | **0.271** |

The false-reject rate halves on both splits (0.549 → 0.280 and 0.531 → 0.271), so the
choice is not an artifact of the selection data and the test split remains clean.

Applying the §5.4 flip rates to the held-out counts makes the case stronger still:

| Prompt (train split) | Repairs | Breakages | Expected net |
|---|---|---|---|
| Original | 50 caught × 0.242 = +12.1 | 51 broken × 0.30 = −15.3 | **−3.2** |
| Tightened | 43 caught × 0.242 = +10.4 | 26 broken × 0.30 = −7.8 | **+2.6** |

On held-out data the original prompt is **net negative** — it would have made the
cascade actively harmful. This is the strongest single justification for the
calibration step.

*Caveat to state:* train-split accuracy is 64.0% versus 55.9% on test, because the
solver was fine-tuned on those questions. The split is used only to compare two judges
against each other on identical inputs, not to estimate the solver's accuracy.

### 5.7 Cascade result (RQ4) — MEASURED

Source: `reports/fydp3_summary.md`, from the three arms run 2026-08-28 09:07–19:31
(`scripts/run_cascade_arms.ps1`, GLM-4.7-Flash via llama.cpp b10453, disagreement gate
at 30%, retry limit 3). Each arm: 1,319 rows, 712 supervisor calls (396 replayed from
`outputs/verdict_cache.jsonl` + 316 new), **0 supervisor errors**.

| Arm | Feedback given on rejection | Correct | Accuracy | vs. baseline |
|---|---|---|---|---|
| baseline | — | 737 | 0.5588 | — |
| **L0** | verdict only | 756 | 0.5732 | +1.44 pt |
| **L1** | verdict + pointer | 765 | 0.5800 | +2.12 pt |
| **L2** | verdict + pointer + correction | **784** | **0.5944** | **+3.56 pt** |

All three beat the un-supervised baseline with statistical significance (exact McNemar):
L0 p=0.034, L1 p=0.0018, **L2 p<0.00001**.

**RQ4 is answered: hint content matters, and monotonically.** The flip matrix shows the
mechanism — richer feedback both repairs more and damages less:

| Arm | wrong→right | right→wrong | net |
|---|---|---|---|
| L0 | 46 | 27 | +19 |
| L1 | 52 | 24 | +28 |
| L2 | **70** | **23** | **+47** |

Going from "try again" to a named mistake plus the correct interpretation raises repairs
from 46 to 70 (+52%) while *reducing* breakage from 27 to 23. A blind retry is close to
a fresh sample; a directed one is not.

**The experimental control held exactly.** All three arms judged the same 396 attempt-1
answers and returned identical verdicts (283 true-reject / 33 false-reject / 48
true-accept / 32 false-accept in every arm). Feedback level is therefore the only
variable separating them, which is what the verdict cache was built to guarantee (§6.4).
Without it, llama.cpp's non-determinism would have made the arms differ by *which*
questions were retried as well.

#### 5.7.1 The result that must not be buried: free self-consistency matches it

| Condition | Correct | Accuracy | Cloud calls | Cloud tokens/q |
|---|---|---|---|---|
| self-consistency@3 (majority vote) | 779 | 0.5906 | **0** | **0.0** |
| L2 cascade | 784 | 0.5944 | 712 | 296.0 |

L2 beats the free control by **5 answers out of 1,319**. Paired exact McNemar against
self-consistency: L0 p=0.062, L1 p=0.275, **L2 p=0.751** — *none of the arms is
statistically distinguishable from the free control.*

Worse, the costs are not comparable in the cascade's favour. The disagreement gate needs
the same 3 samples that self-consistency needs, so the cascade pays **self-consistency's
entire local cost plus 712 supervisor calls** and buys nothing measurable with them.

**As configured, the supervisor cascade is not justified over majority voting.** This is
the honest headline and it must be stated before any comparison against the baseline,
because "+3.56 points over the un-supervised model, p<0.00001" is true and, on its own,
misleading — the un-supervised single sample is not the right thing to beat.

What this does *not* say: hints are worthless (they are worth +47 net flips, §5.7),
or that the cascade cannot win. It says this configuration does not. Three concrete
routes remain, all untested:

1. **Pair the gate with a free signal instead of disagreement.** The confidence gate
   (§5.3, AUC 0.715) needs no extra samples, so a cascade built on it would cost one
   generation plus 712 calls rather than three generations plus 712 calls — a genuinely
   different cost point that self-consistency cannot reach. Requires a new arm (~3.5 h).
2. **Escalate on top of self-consistency rather than instead of it.** Take the majority
   answer, then send only the questions where the vote was split. The two mechanisms fix
   different questions — 82 questions are right in L2 and wrong under self-consistency,
   against 77 the other way — so the union is larger than either. Requires a new arm.
3. **Raise the repair rate.** Every arm is bounded by how often a retry fixes a genuinely
   wrong answer. L2 reached 70 repairs from 283 correctly-rejected answers (24.7%), which
   is barely above the blind-resample rate of 24.2% measured in §5.4 — the hint improved
   *which* answers got retried far more than it improved the retry itself.

**Route 2 has since been measured, and it works. See §5.7.3.**

#### 5.7.3 The stacked arm (RQ4, resolved) — MEASURED 2026-08-29

Route 2 above. Take the majority vote of the three samples the gate has already generated;
send to the supervisor only the questions the vote could not settle. Source:
`analyze_supervision.stack_voting`, reported in `reports/fydp3_summary.md` §1 and §5b.

| Condition | Correct | Accuracy | Cloud calls | Cloud tokens/q |
|---|---|---|---|---|
| local only (greedy, 1 sample) | 737 | 0.5588 | 0 | 0.0 |
| self-consistency@3 (free control) | 779 | 0.5906 | **0** | **0.0** |
| L2 cascade | 784 | 0.5944 | 712 | 296.0 |
| **L2 + voting (stacked)** | **826** | **0.6262** | 712 | 296.0 |

Paired exact McNemar **against free self-consistency** — the control that matters:

| Condition | Control only right | Arm only right | p | Significant |
|---|---|---|---|---|
| L0 | 81 | 58 | 0.062 | no |
| L1 | 78 | 64 | 0.275 | no |
| L2 | 77 | 82 | 0.751 | no |
| L0 + voting | 27 | 46 | 0.034 | **yes** |
| L1 + voting | 24 | 52 | 0.0018 | **yes** |
| **L2 + voting** | 23 | 70 | **<0.00001** | **yes** |

**This is the result that answers RQ4.** The supervisor cascade *is* justified over
majority voting — but only when it is stacked on top of voting rather than run instead
of it. Un-stacked, no arm is distinguishable from the free control (§5.7.1). Stacked,
every arm is, and L2 gains **+3.56 points over self-consistency** and **+6.74 over the
un-supervised model**.

**It costs nothing extra.** The escalated set, and therefore every one of the 712
supervisor calls, is unchanged — cloud tokens per question are identical to the
un-stacked arm at 296.0. The disagreement gate had already generated the three samples
in order to decide what to escalate; the un-stacked arm simply discarded their majority
and kept the greedy answer. Stacking is the removal of a waste, not the addition of a
cost.

**Why it works: the two repairs are disjoint.** Voting fixes answers where the model
already knew better and greedy decoding took the wrong branch. The supervisor fixes
answers the model gets wrong however it is sampled. Stacking gains **exactly +42
answers on every arm** (L0 756→798, L1 765→807, L2 784→826) because the gain lands
entirely on the 923 un-escalated questions, which are the same set in all three arms.
That the increment is identical to the digit across three independent arms is a
consistency check on the mechanism, not a coincidence: feedback level cannot touch a
question that was never escalated.

Note the symmetry in the table above: "L2 + voting vs self-consistency" gives 23/70 at
p<0.00001, which are the *same* discordant counts as "L2 vs greedy baseline" in §5.7.
Stacking applies the cascade on top of voting exactly as the plain arm applies it on
top of greedy decoding. The supervisor's contribution is unchanged; what changed is
what it was layered onto.

**The gate's natural threshold is 37.3%, not 30%.** With k=3 the disagreement score
takes only three values, holding 427 / 400 / 492 questions:

| Disagreement | Questions | Vote differs from greedy |
|---|---|---|
| 0 (all 3 agree) | 427 | 0 |
| 1/3 (two agree) | 400 | **81** |
| 2/3 (all differ) | 492 | 0 |

A 30% rate therefore cuts *inside* the 2/3 group, escalating 396 of those 492 by index
(`select_for_escalation` breaks ties that way). 37.3% escalates the whole group and is
the only threshold this gate genuinely expresses. Measured 2026-08-29 via
`scripts/run_stacked_arm.ps1`:

| Condition | Correct | Accuracy | Escalation | Cloud calls | Cloud tokens/q |
|---|---|---|---|---|---|
| self-consistency@3 (free control) | 779 | 0.5906 | — | 0 | 0.0 |
| L2 @ 30% | 784 | 0.5944 | 0.300 | 712 | 296.0 |
| L2 @ 30% + voting | 826 | 0.6262 | 0.300 | 712 | 296.0 |
| L2 @ 37.3% | 801 | 0.6073 | 0.373 | 889 | 369.0 |
| **L2 @ 37.3% + voting** | **843** | **0.6391** | 0.373 | 889 | 369.0 |

Exact McNemar against free self-consistency: L2@37.3% alone p=0.127 (**not** significant,
84 vs 106) — the un-stacked cascade fails to beat the free control at *both* thresholds,
which strengthens §5.7.1 rather than weakening it. Stacked: 30 vs 94, **p<0.00001**.

**Best measured configuration: 63.91%**, +4.85 points over free self-consistency and
+8.03 over the un-supervised model, against a pass@3 ceiling of 72.93%.

The extra 96 escalations moved 18 correct answers to 32 (**net +14**) for 177 extra
supervisor calls and 73 more cloud tokens per question. That is a markedly worse rate
than the first 396 escalations bought, which is the Pareto trade the figure in §7 item 2
exists to show: the gate is picking the questions it is most confident about first, so
each additional slice of escalation is less productive than the last.

**Why the stacking gain is +42 on every arm at both thresholds.** Voting can only change
an answer in the 1/3 bucket — the other two buckets are unanimous or three-way split, and
`majority_answer` breaks a three-way tie toward the earliest sample, which is greedy
attempt 1. Both thresholds escalate exclusively out of the 2/3 bucket, so the entire 1/3
bucket stays on-device in every arm and contributes the identical +42. The increment is
structural, not a coincidence and not a bug.

#### 5.7.2b The headline figure

`reports/figures/pareto.png` (and `.pdf`), generated by `experiments/plot_pareto.py`
directly from `reports/fydp3_summary.csv` so it cannot drift from the tables.

The Pareto frontier — the only configurations a deployer would ever choose, because
nothing else is at least as accurate for no more cloud cost — has exactly three points:

| Cloud tokens/q | Accuracy | Configuration |
|---|---|---|
| 0.0 | 0.5906 | self-consistency@3 (nothing leaves the device) |
| 296.0 | 0.6262 | L2 + voting |
| 369.0 | 0.6391 | L2 @ 37.3% + voting |

**Every un-stacked cascade arm is off the frontier.** L0, L1, L2 and L2@37% are each
dominated by a free or cheaper alternative, which is §5.7.1 stated in the form a reader
acts on: those four configurations pay ~300 cloud tokens per question for accuracy that
costs nothing to obtain locally. The figure draws them hollow for that reason.

#### 5.7.3b The combined gate end to end — MEASURED 2026-08-29, gate gain does not survive

§5.3.1 measured the combined gate as a *gate*. This arm measures whether that
converts into accuracy. Identical feedback level, identical 30% budget,
identical supervisor and identical verdict cache as the L2 arm in §5.7 — the
only variable is which 396 questions are escalated.
(`scripts/run_combined_gate_arm.ps1`,
`outputs/predictions/14_pipeline_glm30b_hint_full_smart_gate.jsonl`.)

| Stacked arm | Correct | Accuracy | Cloud tokens/q |
|---|---|---|---|
| disagreement gate @ 30% | 826 | 0.6262 | 296.0 |
| **combined gate @ 30%** | **837** | **0.6346** | 307.5 |
| disagreement gate @ 37.3% | 843 | 0.6391 | 369.0 |

The gate did what §5.3.1 predicted it would: it swapped out 76 of the 396
escalated questions and raised the share that were genuinely wrong from
**0.795 to 0.821** (315 -> 325 errors caught).

**But the accuracy gain is not significant.** Paired exact McNemar against the
disagreement gate at the same budget: 61 fixed, 50 broken, net **+11**,
**p=0.343**.

The arithmetic explains it and should be stated rather than hidden. A better
gate can only pay off through the supervisor's repair rate, which §5.7 measured
at 24.7% of correctly-rejected answers. Ten additional errors caught therefore
buys an expected ~2.5 additional repairs — far below the run-to-run noise
measured in §5.7.4, where re-running the same arm moved 40 answers each way.
**A gate improvement of this size cannot be detected end to end**, and no arm at
this scale could have shown it. That is a limitation of the experiment, not
evidence that the gate is no better.

**What it does buy is cost.** Against the 37.3% arm the combined gate at 30% is
statistically indistinguishable (60 fixed, 54 broken, net +6, **p=0.640**) while
using **17% fewer cloud tokens per question** (307.5 against 369.0). Both sit on
the Pareto frontier; the combined gate reaches essentially the same accuracy for
less money, which is the axis this thesis argues on.

Frontier after this arm (`reports/figures/pareto.png`):

| Cloud tokens/q | Accuracy | Configuration |
|---|---|---|
| 0.0 | 0.5906 | self-consistency@3 |
| 296.0 | 0.6262 | L2 + voting, disagreement gate |
| 307.5 | 0.6346 | L2 + voting, **combined gate** |
| 369.0 | 0.6391 | L2 + voting, disagreement gate @ 37.3% |

**Honest summary for the write-up.** The combined gate is a better gate
(AUC 0.840 -> 0.869, p-free ranking measurement) and a cheaper route to the same
accuracy, but it is *not* a demonstrated accuracy improvement. Report the gate
metric as the finding and the accuracy as indistinguishable.

#### 5.7.5 Best configuration, with the better judge — MEASURED 2026-08-29

§5.5.2 established that Qwen3.5-9B is a better judge than GLM-4.7-Flash. Every arm
above used GLM, so all of them understate the cascade. This re-runs the best
configuration — combined gate at 30%, L2 feedback, stacked on the vote — changing
only the judge. `scripts/run_combined_gate_arm.ps1 -Model qwen`,
`outputs/predictions/15_pipeline_qwen9b_BEST_RESULT.jsonl`.

| Configuration | Correct | Accuracy | Cloud tokens/q |
|---|---|---|---|
| local only (greedy) | 737 | 0.5588 | 0.0 |
| self-consistency@3 (free control) | 779 | 0.5906 | 0.0 |
| GLM, combined gate @ 30%, stacked | 837 | 0.6346 | 307.5 |
| GLM, disagreement gate @ 37.3%, stacked *(previous best)* | 843 | 0.6391 | 369.0 |
| **Qwen, combined gate @ 30%, stacked** | **903** | **0.6846** | 345.8 |
| pass@3 ceiling | 962 | 0.7293 | — |

**0.6846 — a 4.55-point gain over the previous best, from changing the judge alone.**
Paired exact McNemar: against the previous best, 114 fixed / 54 broken, net **+60**,
**p=0.000004**; against GLM at the identical 30% budget, 100 fixed / 34 broken, net
**+66**, **p<0.000001**. This is now **93.9% of the pass@3 ceiling** — the cascade
recovers almost every answer the local model was capable of producing.

**The un-stacked arm now beats the free control too, for the first time in this
study.** Qwen's cascade without voting scores 861/1319 = 0.6528 against
self-consistency's 0.5906: 148 fixed / 66 broken, **p<0.000001**. Every GLM arm failed
that test (§5.7.1). The null result in §5.7.1 was therefore **a property of the judge,
not of the cascade design** — a conclusion that was not available until a better judge
was measured, and which materially changes the thesis's answer to RQ4. Stacking still
helps (+42, unchanged and for the same structural reason as §5.7.3), but it is no longer
what rescues the approach.

Judge quality on the escalated 396, same gate, same questions:

| Judge | True reject | False reject | Precision | Recall |
|---|---|---|---|---|
| GLM-4.7-Flash | 297 | 35 | 0.8946 | 0.9138 |
| **Qwen3.5-9B** | **301** | **31** | **0.9066** | **0.9262** |

**Limitation — truncation, bounded and measured.** Qwen produced 50 truncated replies
across 728 calls (6.9%): 18 on attempt 1, 32 on retries, even at
`SUPERVISOR_MAX_OUTPUT_TOKENS=512`. §6.8 converts a truncated reply to an error and an
error to an *accept*, so each is a silent approval. Of the 18 attempt-1 truncations, 10
were on answers that were already correct (no harm) and **8 were wrong answers waved
through**, worth roughly 2 answers at the measured 24.7% repair rate. **The 0.6846 is
therefore a slight underestimate, not an overestimate**, and raising the budget further
is a cheap improvement rather than a correction. GLM truncated 0 times; Qwen averages 97
output tokens against GLM's 57.

**Cost.** 345.8 cloud tokens per question against GLM's 307.5 at the same escalation
rate and the same 728 calls — the difference is Qwen's longer replies, not more of them.
It remains cheaper than the previous best (369.0) while scoring 4.55 points higher, so it
dominates that configuration outright. Frontier (`reports/figures/pareto.png`):

| Cloud tokens/q | Accuracy | Configuration |
|---|---|---|
| 0.0 | 0.5906 | self-consistency@3 |
| 307.5 | 0.6346 | GLM, combined gate, stacked |
| 345.8 | **0.6846** | **Qwen, combined gate, stacked** |

#### 5.7.4 Run-to-run variance — measured, and it matters

The 37.3% arm re-ran 396 retries that the 30% arm had already run a day earlier, which
gives a free replication (`scratchpad/variance.py` logic; both files in §10).

| Measure | Result |
|---|---|
| Attempt-1 verdicts identical | **396 / 396** |
| Final answers identical | 127 / 396 (32.1%) |
| Accuracy on the shared set | 0.3232 vs 0.3308 |
| Discordant pairs | 40 vs 43, McNemar **p=0.826** |

Two things follow. First, the verdict cache did exactly its job: every attempt-1 judgment
replayed byte-identically across runs a day apart, so the arms differ only where they are
supposed to. All 397 new calls in this run were attempt-2 judgments on freshly sampled
retries; all 492 attempt-1 verdicts replayed free from the pass in §5.5.

Second, and this belongs in the limitations: **the retry is reproducible in aggregate and
not at all reproducible per question.** Two thirds of retried answers came out different,
because retries are temperature-sampled — yet the accuracy they produce is statistically
indistinguishable (p=0.83). No claim about *which* question a hint repaired is safe. Every
claim in §5.7 is about rates over 1,319 questions, and should be written that way.

### 5.7.2 Cost of each feedback level

Counterfactual cloud output tokens, reconstructed from the recorded hint text:

| Feedback level | Cloud output tokens | Per question | Accuracy |
|---|---|---|---|
| L0 | 5,696 | 4.3 | 0.5732 |
| L1 | 17,949 | 13.6 | 0.5800 |
| L2 | 31,429 | 23.8 | 0.5944 |

L2 costs 5.5× L0's output tokens for +2.1 points. Within the cascade family that is a
good trade — output tokens are the cheap half and 24 tokens per question is small. It
does not rescue the comparison in §5.7.1.

### 5.8 The gate captures nearly all the gain at a third of the cost — projected

*(Projection made before the arms ran. Retained for provenance: it predicted ~+59 net
and ~60.3% for the cascade; L2 measured +47 net and 59.44%, so the projection was
optimistic by roughly a fifth. The over-estimate came from assuming the hint would lift
the repair rate above the 24.2% blind-resample baseline; it did not — see §5.7.1 route 3.)*


Repeating the §5.7 arithmetic with the measured full-set judge rates (§5.5), at two
escalation rates:

| Escalation | Supervisor calls | Wrong rejected | Correct rejected | Repairs | Breakages | Net |
|---|---|---|---|---|---|---|
| **30%** (disagreement gate) | 396 | 263 | 21 | +63.6 | −6.2 | **+57.4** |
| **100%** (no gate) | 1,319 | 486 | 188 | +117.6 | −56.4 | **+61.2** |

**Escalating everything costs 3.3× the cloud calls to gain 6% more accuracy.** This is
the Pareto argument the thesis is built on, and it holds because the gate concentrates
escalation on questions that are actually wrong (79.6% precision at 30%), so the judge's
false rejections have far fewer correct answers to damage.

Still a projection — it assumes hints perform like blind resampling. The arms in §7
item 3 measure it.

---

### 5.9 Latency — MEASURED 2026-08-29, and it reframes the cost argument

Every cost figure until now was in tokens. Tokens measure cloud spend well and
latency badly, and latency is what the thesis actually argues about. Measured
with `experiments/benchmark_latency.py --limit 40`, results in
`reports/latency_benchmark.json`.

| Component | Mean | Median | p90 | Throughput |
|---|---|---|---|---|
| Local model writes one answer | **15.12 s** | 13.57 s | 22.94 s | 8.8 tokens/s |
| Qwen3.5-9B judges one answer | **1.00 s** | 1.00 s | 2.02 s | — |
| GLM-4.7-Flash judges one answer *(729 requests, server log)* | 2.07 s | 2.04 s | 2.60 s | 44.6 tokens/s |

Composed at the reported operating point (3 samples, 30% escalation, 332/396
rejection rate):

| Path | Seconds per question |
|---|---|
| Stays on the device | 45.37 |
| Goes to the checker | 59.05 |
| **Average across all questions** | **49.47** |
| *One answer, no cascade at all* | *15.12* |

**The cascade costs 3.3x the wall clock of a single answer. Where that time goes
is the finding:**

| | Seconds | Share |
|---|---|---|
| Three local samples | 45.37 | **91.7%** |
| Retry generation (30% of questions) | 3.80 | 7.7% |
| **Waiting on the checker** | **0.30** | **0.6%** |

**The cloud is 0.6% of the time a user waits.** The expensive thing is not
escalation — it is *sampling three times to decide whether to escalate*. The
gate's own cost dominates the cost of everything it gates.

**What this does and does not change.** It does not touch any accuracy result.
It does not weaken the privacy argument: 70% of questions still never leave the
device, and that is a claim about data, not seconds. What it changes is the
framing of §6.1. "The supervisor is a rare, costly fallback" is true of cloud
tokens and false of latency — measured, the supervisor is the cheapest component
in the system. Any sentence in the report that justifies rare escalation on
*speed* grounds must be rewritten to justify it on *data-leaving-the-device*
grounds, which the evidence does support.

**It also confirms a claim `gate.py` had been making without evidence.** The
module docstring says the disagreement gate "multiplies the wall-clock time per
question by k". Measured: 45.37 s against 15.12 s, which is 3.00x. The reasoning
was right; only the magnitude of the comparison was unknown.

**The local model is the bottleneck, and it need not be.** 8.8 tokens/second is
slow, and the cause is the serving stack rather than the model: a 2B model
through transformers with bitsandbytes 4-bit is **15x slower per answer** than a
**9B** model through llama.cpp with a grammar constraint. This is the same
inversion found in §5.5.1, where the 2B self-judge was 6x slower per judgement
than the 30B. Serving the solver as GGUF would very likely cut the 45 seconds
substantially without touching accuracy, and is the single largest latency
improvement available. Untested, and recorded as future work rather than claimed.

**Measurement caveat, which must be stated wherever these numbers appear.** They
were taken on an RTX 4080 SUPER, which is not the weak hardware the thesis
describes. Treat the local figures as a floor: on an old laptop generation is
slower still, which moves the balance *further* towards local cost and makes the
0.6% cloud share smaller, not larger. A genuine edge measurement needs the
benchmark run on such a device; it requires only the solver, not the checker.

### 5.10 Does the gate transfer to a different solver? — DESIGNED, NOT RUN (decision 2026-09-11)

> **Status: deferred by the user on 2026-09-11, to be run only if the supervisor
> asks for it.** The design below is complete and the code is written, tested and
> committed, so it can be executed later with one command. **No result exists, and
> none is claimed anywhere in this thesis.** The corresponding limitation is stated
> in §8.9 — a single-solver evaluation — which is how the thesis handles this gap.
>
> A partial file of 51 generated answers exists at
> `outputs/predictions/17_qwen15b_answers_try1.jsonl`. It is 3.9% of the test set,
> its confidence interval spans roughly 45%-72%, and **it must not be quoted as a
> result** in any form. It is kept only as a resume point.

**The gap this would close.** Every number above comes from one solver. A reviewer can
fairly say the gate might be exploiting a quirk of `gemma-4-E2B-it` rather than a
general property of sampled reasoning, and nothing measured so far can refute that.
The literature search (`docs/RELATED_WORK.md` §G2) sharpened this into a concrete
threat: **Qwen2.5-1.5B-Instruct is reported at 73.2% on GSM8K** (4-shot, official
Qwen2.5 Technical Report, arXiv:2412.15115) — a *smaller* model, with no
fine-tuning, no voting and no checker, above our full 68.46% cascade.

**Design, if it is ever run.** Run the identical pipeline with `Qwen/Qwen2.5-1.5B-Instruct` as the
solver, keeping the Qwen3.5-9B checker, the combined gate, and the 30% escalation
rate fixed. Two questions, one run:

1. What does that model actually score on **our** harness — 4-bit NF4, strict
   `#### N` extraction with last-number fallback, the full 1,319-question test set?
   The published figure uses a different protocol and is not comparable.
2. Do the disagreement gate, the confidence tie-break and the stacking result
   reproduce on a solver nothing in this project was tuned against?

**The solver is used stock — no fine-tuning, no adapter.** Deliberate: the claim
under test is about the routing mechanism, not about training, and leaving the
solver untouched removes any suspicion that the gate was co-tuned with it.

#### 5.10.1 Protocol decisions, and why they differ from the Gemma arm

**Zero-shot, not 8-shot. Measured, not assumed.** On an 8-question smoke test the
8-shot completion-style prompt used for the Gemma baseline scored 3/8 and produced
one degenerate answer — 512 tokens of a single repeated digit — while zero-shot
scored 3/5 with coherent, extractable reasoning. Qwen2.5-Instruct is a chat model
and few-shot completion prompting takes it out of distribution. This is consistent
with arXiv:2604.07035, which found GSM8K the most prompt-sensitive of its four
benchmarks, with a 0.560 spread between strategies on a single model.

*Consequence for reporting:* the Qwen arm is zero-shot and the Gemma baseline is
8-shot, so **the two base-model numbers are not a like-for-like comparison and must
never be presented as one.** Within the Qwen arm every condition is zero-shot, which
is what the transfer claim needs.

**Token budget raised to 768** (default 512). Zero-shot Qwen is verbose — one smoke
answer reached 502 tokens — and truncating its reasoning would understate it. The
Gemma arm keeps 512; this is a per-arm setting, recorded here because it is a
difference between arms.

#### 5.10.2 Code changes this required

Multi-solver support did not exist; the pipeline was Gemma-only in three places.
Added under test (`tests/test_model_family.py`, 21 new cases, suite 138 -> 159):

- `gsm8k.ModelFamily` / `family_for()` — chat tags and loader class travel together,
  because they are two faces of one fact. An unregistered model **raises** rather
  than defaulting: the wrong chat template does not crash, it yields a fluent,
  parseable, meaningless run, which is far more expensive to discover than an error.
- `model_utils.trim_at_end_tag()` — the stop marker is family-specific. Left in
  place, `<|im_end|>` would sit inside the stored prediction where the last-number
  fallback in `extract_final_answer` could read digits out of trailing chatter.
- `supervise.resolve_adapter_dir()` + `--no-adapter` — an empty `--adapter-dir`
  falls through to the configured *Gemma* adapter, so evaluating another base model
  needed a way to say "no adapter" that cannot be read as "not specified".

Defaults are unchanged throughout and pinned by a regression test, so every
existing Gemma result reproduces byte-identically.

#### 5.10.3 Results — NONE. The run was not performed.

There is no results table because there are no results. The run was stopped after
51 of 1,319 questions and deliberately not resumed.

Cost if it is resumed later: ~14.4 h of generation (measured at 13.07 s/question,
3 samples x 1,319), ~1 h confidence scoring, ~1.5 h for the cascade.

Reproduce with `scripts/run_qwen_solver_arm.ps1`, then the cascade command it
prints. Outputs are `outputs/predictions/17_`–`23_qwen15b_*`. Log:
`reports/qwen_solver_arm.log`.

**Pre-registered interpretation, written before the run was stopped and kept here
unchanged.** If this is ever executed, these readings stand as written; they were not
authored with any result in view:

- *Gate AUC holds near 0.84–0.87 and stacking still wins* → the mechanism is a
  property of sampled reasoning, not of Gemma. This is the result that closes §G2.
- *Gate AUC collapses* → the gate was exploiting something specific to our
  fine-tuned solver, and the thesis must say so. That is a real finding and gets
  reported either way.
- *Qwen scores far below 73.2% on our harness* → the published figure is
  protocol-dependent, which is worth stating plainly, **but it is not a defence of
  our number** and must not be used as one.

## 6. Design decisions and why

### 6.1 The supervisor is a rare fallback, not a quality booster
Calling the supervisor on every question — including the 737 already correct — is
strictly worse on cost than just asking the frontier model once, and would make the
pipeline unable to win its own argument. The gate is the central design element.

### 6.2 Inference-time only
Hints are in-context. Weights never change after fine-tuning. A second weight-update
round was considered and rejected as out of scope.

### 6.3 The supervisor never sees the gold answer
`_prompt()` takes no gold argument at all, so there is no code path by which a real
supervisor could mark its own homework. Only the `exact` oracle receives it. Tested
(`test_gold_answer_never_reaches_the_prompt`). Expect this question at the defence.

### 6.4 Verdict caching is a validity mechanism, not an optimisation
Attempt 1 is byte-identical across L0/L1/L2. Because the judge is not deterministic
(§5.6), independently re-judging it per arm would make the arms reject *different
sets* — so any measured difference would confound hint content with which questions
happened to be retried. Caching makes "all arms reject the same set" structural.

### 6.5 Self-hosted supervisor via llama.cpp, not transformers
bitsandbytes quantises from fp16 weights, so hosting a 30B MoE that way needs a ~60 GB
download and still would not fit in 16 GB. GGUF is 17.5 GB and offloads experts to RAM
(only ~3B of 30B parameters are active per token). Also isolates the verifier from the
solver's dependency stack, and makes swapping judges a server restart rather than a
code change — which is what makes the RQ5 ladder affordable.

### 6.6 Gemini demoted to a small reference arm
Google stopped publishing free-tier quotas; they are per-project and visible only in
AI Studio. Reported figures for `gemini-2.5-flash` range from 20 to 1,500 requests/day
across sources, after a 50-80% cut on 2025-12-07. Building a thesis on an unmeasurable,
silently-changing quota was judged a bad bet. Self-hosting also tells a better edge
story. Gemini remains as a ~300-question reference for RQ5.

### 6.7 Thinking mode off for main runs
The reported cost metric is supervisor output tokens. A verifier that reasons for
several hundred tokens per judgment breaks the premise that feedback is a 12-57 token
hint. A thinking-on ablation on ~300 questions will quantify what that costs.

### 6.8 Three safety properties in the client
An unreadable reply is treated as **accept**, deliberately — the cascade must not
manufacture rejections from its own bugs. The corollary is that anything ambiguous
becomes silent approval, so:
- empty or truncated replies raise an error at the provider boundary, never reach the parser;
- Gemini gets `thinkingConfig.thinkingBudget = 0`, because thinking tokens are billed
  against `maxOutputTokens` and exhaust it before any JSON is emitted;
- N consecutive failures abort the run, so a dead endpoint cannot approve the remaining
  questions and exit reporting success.

---

## 7. Work remaining

**Done:** train-split prompt re-validation (§5.6); full verdict pass (§5.5); confidence
gate (§5.3); cascade arms L0/L1/L2 (§5.7); stacked arm (§5.7.3) — the result that
answers RQ4; natural-threshold arm at 37.3% (§5.7.3); combined gate, as a gate (§5.3.1)
and end to end (§5.7.3b); confidence-based answer selection (§5.3.2, null); Pareto
figure (§5.7.2b); run-to-run variance (§5.7.4).

**Best measured configuration: 0.6846** (903/1319) — combined gate at 30%, L2 feedback,
stacked on the vote, judged by **Qwen3.5-9B** (§5.7.5). That is 93.9% of the pass@3
ceiling, and +12.6 points over the un-supervised fine-tuned model.

| # | Item | Blocks | Est. | Status |
|---|---|---|---|---|
| 1 | Commit the repository; revoke the token in `Hugging face tokken.txt` | publication | ~15 min | **deferred by the user 2026-08-29 until the thesis is finished — do not commit unprompted** |
| 1b | Store the supervisor model id in each `verdict_cache.jsonl` row | makes §5.5.2's corruption hazard detectable | ~20 min | found 2026-08-29 |
| 2 | 100 hand-labelled examples for reasoning validity | §8.3 — no "lucky correct" claim without it | ~2 h manual | needs the user |
| 3 | ~~Verifier ladder (RQ5)~~ **done, §5.5.1 / §5.5.2 — the 9B beats the 30B on every measure and is 1.8x faster** | — | — | complete |
| 3b | ~~Re-run the best cascade against Qwen3.5-9B~~ **done, §5.7.5 — 0.6846, p<0.00001** | — | — | complete |
| 3c | Re-run with a larger output budget (1024) to remove Qwen's 6.9% truncation | worth ~2 answers; 0.6846 is an underestimate | ~3.5 h | low priority |
| 3d | Serve the solver as GGUF and re-measure latency | §5.9 — the solver is 15x slower per answer than the 9B checker, and it is the stack not the model | ~2 h | **largest latency win available** |
| 3e | Run `benchmark_latency.py` on an actual old laptop | §5.9 — current numbers are from a 4080, not the deployment target | ~30 min | needs the hardware |
| 4 | No-previous-answer ablation, ~200 questions | anchoring (§8.4) | ~1 h | |
| 5 | Thinking-on ablation, ~300 questions | §6.7 | ~1 h | |
| 6 | Gemini reference, ~300 questions | RQ5, needs `GEMINI_API_KEY` | ~30 min | |

Item 1 is now first on risk, not value: eleven months of work exists only in one working
tree, and a file in the repo root may still hold a live Hugging Face token.

**Exact commands for item 3, so the next session starts cold without re-deriving them.**
The verifier ladder is a server swap, not a code change; both rungs are already wired.

```powershell
# rung 1 - the 2B fine-tuned model judging its own work. No download, no server.
python preflight_supervisor.py --provider self --limit 30      # ~15 min, check leniency FIRST
python supervise.py --provider self --gate none --verdict-only --resume

# rung 2 - Qwen3.5-9B, ~6 GB download, then restart the server on it
.\scripts\serve_verifier.ps1 -Model qwen -Download
.\scripts\serve_verifier.ps1 -Model qwen
python preflight_supervisor.py --provider llamacpp
python supervise.py --provider llamacpp --supervisor-model Qwen3.5-9B-UD-Q4_K_XL `
    --gate none --verdict-only --resume
```

Run the preflight before each full pass. A judge that accepts everything makes the
cascade a no-op, and that is 45 minutes to discover rather than 2 hours. The `self`
rung is the one that matters most: if a 2B model can check its own work, the cascade
needs no cloud supervisor at all, and if it cannot, that is the cleanest possible
justification for having one.

Note that `--verdict-only` measures judge *quality* (precision, recall, false-reject
rate) without running any retries, which is all RQ5 needs. A full cascade arm per rung
is only worth running if a rung's confusion matrix looks competitive with GLM's
(283 TR / 33 FR / 48 TA / 32 FA, §5.5).

**A note on statistical power, which item 3 must be planned around.** §5.7.3b and §5.3.2
both measured real mechanism improvements that did not survive as accuracy. The chain is
long — gate catches an error, judge rejects it, retry repairs it — and only ~25% of
correct rejections become repairs (§5.7). Against the run-to-run noise measured in
§5.7.4 (40 answers moving each way on a re-run of the same arm), an intervention needs
to catch roughly 40+ additional errors before it can be distinguished end to end on
1,319 questions. Report gate-level metrics as gate-level findings; do not expect small
ones to appear in the accuracy column, and say so before a panel asks why.

---

## 8. Threats to validity — state these before a panel does

### 8.1 Everything is bounded by pass@3 = 72.93%
Not a flaw, but it caps the claim. Report all results against the 55.88% / 72.93% band.

### 8.2 GSM8K is saturated for frontier models
Frontier models score ~95-98%. This is a limitation of the benchmark, not the work.
Frame GSM8K as a *controlled, gradeable probe of multi-step reasoning*, never as a
workload anyone would ship on a phone. **Action: source exact frontier and
comparable-small-model GSM8K figures for the write-up rather than asserting them.**

### 8.3 "Correct reasoning" has no ground truth here
The judge is scored against *final-answer* correctness, but it is asked to grade
*reasoning*. A right answer from faulty reasoning is a true reject by the prompt and a
false reject by the metric. Item 7 in §7 exists to bound this; until then, any claim
about "lucky-correct" answers is unfalsifiable and must not be made.

### 8.4 Anchoring
The retry prompt includes the rejected answer, which may lock the model into its own
misreading. Already observed: in the 5-example oracle smoke, the "half that much" robe
problem repeated its wrong answer **verbatim** under an uninformative rejection. Item 8
in §7 ablates it.

### 8.5 Judge prompt selected on test data — resolved
Selection was re-validated on 150 held-out train-split questions and replicates
(§5.6). Disclose the procedure in the write-up as good practice, but it is no longer a
threat: the reported test results do not depend on a choice made using them.

### 8.6 Non-determinism in the verifier
llama.cpp is not bit-reproducible (§5.6). Mitigated by the verdict cache for
cross-arm comparability; should still be disclosed.

### 8.7 No on-device measurements
Everything is a device-agnostic proxy — tokens, escalation rate, memory footprint. **Do
not make latency or battery claims without measuring on real hardware.**

### 8.8 Reasoning validity was not hand-checked
The 100-example manual labelling (§7 item 2) was not carried out. **No claim is made
anywhere that answers are correct for the right reasons**, only that the final numbers
match. A model can reach the right total by a wrong route, and this thesis cannot
distinguish those cases. State this plainly rather than letting a reader assume
otherwise.

### 8.9 A single solver — the generality of the gate is untested
Every result comes from one base model, `gemma-4-E2B-it`. The gate, the confidence
tie-break and the stacking result are therefore demonstrated *on that model*, not shown
to be general properties of sampled reasoning. A transfer experiment on a second solver
from a different family was designed and implemented (§5.10) but **deliberately not
run**; the code is committed and it can be executed later.

This is the most substantive open question in the thesis, and it should be volunteered
in the limitations section rather than waited for. The honest framing: *the mechanism
is shown to work; how far it generalises across models is future work.*

---

## 9. Likely defence questions, and the answers

**Q. Why not just call the frontier model for everything? It gets 97%.**
Because it is not on the phone, and because that is a different point on the cost curve,
not a competitor. The claim is about accuracy *per cloud token*. A frontier solver costs
~1,700 input tokens per question at full price for every question; the cascade sends
~355 tokens for 30% of them. The thesis maps the frontier between those endpoints.

**Q. How do you know the supervisor did anything? Maybe retrying is what helps.**
It is not. Blind retrying *loses* 6 accuracy points (55.88% → 49.89%, §5.4). Resampling
breaks correct answers 30% of the time and fixes wrong ones 24% of the time. Any gain
the cascade shows is attributable to the verdict, because the un-verdicted version of
the same operation is measurably harmful. This control was run *before* the treatment.

**Q. Isn't 72.9% low?**
That is the ceiling, not the score; the operating point is 55.88%. And the model is 2B
parameters at 4-bit precision — roughly 1% of the size of the systems scoring in the
90s, running under a phone-class memory budget. The interesting axis is accuracy per
token under that constraint, where no saturated answer exists.

**Q. Does the supervisor see the answer? Isn't it just solving the problem?**
No. `_prompt()` takes no gold argument, so there is no code path that could leak it
(§6.3), and hints are stripped of any `####` marker the judge emits anyway. Only the
`exact` oracle — used as a ceiling, never as a result — receives the gold answer.

**Q. Why an open-weight judge instead of GPT/Claude/Gemini?**
Three reasons: a pinned GGUF checkpoint is reproducible forever whereas free-tier hosted
models are swapped without notice; it removes an unmeasurable quota from the critical
path (§6.6); and a fully self-hosted cascade is a stronger edge-deployment claim than
one with a cloud dependency. A Gemini arm is retained as a reference point.

**Q. Your judge rejects 28% of correct answers. Isn't that bad?**
Yes, and it is reported rather than hidden — the flip matrix is presented alongside net
accuracy throughout. It is also 28% *of the ~20% of escalated questions that were
already correct*, because the gate filters first. And it is a large improvement over the
54.9% of the first prompt, which was measured and fixed (§5.6).

**Q. Why 30% escalation? Isn't that tuned to flatter the result?**
No threshold is baked in. `gate_curve` sweeps every escalation rate from 5% to 100% and
the full curve is reported (§5.3). 30% is presented as one operating point on a
published curve, not as a chosen constant.

---

## 10. Artifact index

| Artifact | Path |
|---|---|
| FYDP 2 headline | `reports/gsm8k_summary.md` |
| Sample bank analysis (§5.3, §5.4) | `reports/fydp3_samplebank.md` / `.csv` |
| Attempt-1 predictions (greedy, cached) | `outputs/predictions/02_answers_finetuned_try1_main.jsonl` |
| Sample bank | `outputs/predictions/samplebank_s{1,2}_gsm8k_test.jsonl` |
| Oracle smoke / apparatus checks | `reports/smoke_fydp3.md` |
| Verdict cache | `outputs/verdict_cache.jsonl` |
| Run-by-run provenance | `reports/experiment_log.md` / `.jsonl` |
| Judge prompt + calibration figures | `thesis_pipeline/supervisor_client.py` (`SYSTEM_PROMPT`) |
| Cascade arms L0/L1/L2 (§5.7) | `outputs/predictions/cascade_llamacpp_disagreement_L{0,1,2}_test.jsonl` |
| Cascade + stacked results (§5.7.3) | `reports/fydp3_summary.md` / `.csv` |
| Stacked-arm logic | `analyze_supervision.stack_voting`, tested in `tests/test_analysis.py::StackVotingTests` |
| Natural-threshold arm (37.3%) | `scripts/run_stacked_arm.ps1` -> `outputs/predictions/13_pipeline_glm30b_hint_full_more_escalation.jsonl` |
| Headline Pareto figure (§5.7.2b) | `reports/figures/pareto.png` / `.pdf`, from `experiments/plot_pareto.py` |

Every `generate.py`, `evaluate.py` and `supervise.py` run appends to the experiment log
with its full command line, model id and parameters. That log is the audit trail for
everything in §4 and §5.
