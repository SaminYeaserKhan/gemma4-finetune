# Every experiment, one folder each

**Who this is for:** anyone writing the paper. Each folder below holds one system we
tested on all 1,319 test questions: a plain description, a diagram of what it is made
of, a diagram of what happens to the questions (with the real counts), and its numbers.

> **Generated file — do not edit by hand.** Every result here is counted from the
> result files by `scripts/build_experiment_catalogue.py`; the few measurements that
> were never saved to a file (the large marker's speed, the 30-question self-check)
> are quoted with their source. To update, re-run the script.

## The folders

Folder names describe the system: `+` joins the parts it is built from. For example
`finetuned-model+vote+smart-gate+checker_qwen-9b_full-hint` means *the trained AI, with
a majority vote, a smart way of picking hard questions, and a 9-billion-parameter
marker that gives a full hint*.

- [`01_base-model-only/`](01_base-model-only/README.md) — The small AI before we trained it
- [`02_finetuned-model-only/`](02_finetuned-model-only/README.md) — The small AI after training, answering once
- [`03_finetuned-model+blind-retry/`](03_finetuned-model+blind-retry/README.md) — Trained AI, asked again, keeping the second answer
- [`04_finetuned-model+majority-vote/`](04_finetuned-model+majority-vote/README.md) — Trained AI answers three times, most common answer kept
- [`05_finetuned-model+checker-marks-all-answers_glm-30b/`](05_finetuned-model+checker-marks-all-answers_glm-30b/README.md) — A large AI marks every answer (measuring the marker)
- [`06_finetuned-model+checker-marks-all-answers_qwen-9b/`](06_finetuned-model+checker-marks-all-answers_qwen-9b/README.md) — A medium AI marks every answer (measuring the marker)
- [`07_finetuned-model+perfect-checker-marks-all-answers/`](07_finetuned-model+perfect-checker-marks-all-answers/README.md) — A perfect marker marks every answer (the upper limit)
- [`08_finetuned-model+gate+checker_glm-30b_no-hint/`](08_finetuned-model+gate+checker_glm-30b_no-hint/README.md) — Trained AI + selection of hard questions (30%) + large AI marker, none hint
- [`09_finetuned-model+gate+checker_glm-30b_short-hint/`](09_finetuned-model+gate+checker_glm-30b_short-hint/README.md) — Trained AI + selection of hard questions (30%) + large AI marker, short hint
- [`10_finetuned-model+gate+checker_glm-30b_full-hint/`](10_finetuned-model+gate+checker_glm-30b_full-hint/README.md) — Trained AI + selection of hard questions (30%) + large AI marker, full hint
- [`11_finetuned-model+gate+checker_glm-30b_full-hint_37pct/`](11_finetuned-model+gate+checker_glm-30b_full-hint_37pct/README.md) — Trained AI + selection of hard questions (37.3%) + large AI marker, full hint
- [`12_finetuned-model+smart-gate+checker_glm-30b_full-hint/`](12_finetuned-model+smart-gate+checker_glm-30b_full-hint/README.md) — Trained AI + smart selection of hard questions (30%) + large AI marker, full hint
- [`13_finetuned-model+vote+gate+checker_glm-30b_no-hint/`](13_finetuned-model+vote+gate+checker_glm-30b_no-hint/README.md) — Trained AI + majority vote + selection of hard questions (30%) + large AI marker, none hint
- [`14_finetuned-model+vote+gate+checker_glm-30b_short-hint/`](14_finetuned-model+vote+gate+checker_glm-30b_short-hint/README.md) — Trained AI + majority vote + selection of hard questions (30%) + large AI marker, short hint
- [`15_finetuned-model+vote+gate+checker_glm-30b_full-hint/`](15_finetuned-model+vote+gate+checker_glm-30b_full-hint/README.md) — Trained AI + majority vote + selection of hard questions (30%) + large AI marker, full hint
- [`16_finetuned-model+vote+gate+checker_glm-30b_full-hint_37pct/`](16_finetuned-model+vote+gate+checker_glm-30b_full-hint_37pct/README.md) — Trained AI + majority vote + selection of hard questions (37.3%) + large AI marker, full hint
- [`17_finetuned-model+vote+smart-gate+checker_glm-30b_full-hint/`](17_finetuned-model+vote+smart-gate+checker_glm-30b_full-hint/README.md) — Trained AI + majority vote + smart selection of hard questions (30%) + large AI marker, full hint
- [`18_finetuned-model+smart-gate+checker_qwen-9b_full-hint/`](18_finetuned-model+smart-gate+checker_qwen-9b_full-hint/README.md) — Trained AI + smart selection of hard questions (30%) + medium AI marker, full hint
- [`19_finetuned-model+vote+smart-gate+checker_qwen-9b_full-hint/`](19_finetuned-model+vote+smart-gate+checker_qwen-9b_full-hint/README.md) — Trained AI + majority vote + smart selection of hard questions (30%) + medium AI marker, full hint — BEST RESULT

---

## Table 1 — How many questions each system got right

"Turned right" and "turned wrong" are counted against the trained AI answering once
(system 02). A system can gain answers and lose others at the same time; the score
alone hides that.

| # | System | Right answers | Score | vs. trained AI alone | Turned right | Turned wrong |
|---|---|---|---|---|---|---|
| [01](01_base-model-only/README.md) | The small AI before we trained it | 479 | 36.32% | -19.56 | 138 | 396 |
| [02](02_finetuned-model-only/README.md) | The small AI after training, answering once | 737 | 55.88% | +0.00 | 0 | 0 |
| [03](03_finetuned-model+blind-retry/README.md) | Trained AI, asked again, keeping the second answer | 658 | 49.89% | -5.99 | 138 | 217 |
| [04](04_finetuned-model+majority-vote/README.md) | Trained AI answers three times, most common answer kept | 779 | 59.06% | +3.18 | 54 | 12 |
| [05](05_finetuned-model+checker-marks-all-answers_glm-30b/README.md) | A large AI marks every answer (measuring the marker) | 737 | 55.88% | +0.00 | 0 | 0 |
| [06](06_finetuned-model+checker-marks-all-answers_qwen-9b/README.md) | A medium AI marks every answer (measuring the marker) | 737 | 55.88% | +0.00 | 0 | 0 |
| [07](07_finetuned-model+perfect-checker-marks-all-answers/README.md) | A perfect marker marks every answer (the upper limit) | 737 | 55.88% | +0.00 | 0 | 0 |
| [08](08_finetuned-model+gate+checker_glm-30b_no-hint/README.md) | Trained AI + selection of hard questions (30%) + large AI marker, none hint | 756 | 57.32% | +1.44 | 46 | 27 |
| [09](09_finetuned-model+gate+checker_glm-30b_short-hint/README.md) | Trained AI + selection of hard questions (30%) + large AI marker, short hint | 765 | 58.00% | +2.12 | 52 | 24 |
| [10](10_finetuned-model+gate+checker_glm-30b_full-hint/README.md) | Trained AI + selection of hard questions (30%) + large AI marker, full hint | 784 | 59.44% | +3.56 | 70 | 23 |
| [11](11_finetuned-model+gate+checker_glm-30b_full-hint_37pct/README.md) | Trained AI + selection of hard questions (37.3%) + large AI marker, full hint | 801 | 60.73% | +4.85 | 94 | 30 |
| [12](12_finetuned-model+smart-gate+checker_glm-30b_full-hint/README.md) | Trained AI + smart selection of hard questions (30%) + large AI marker, full hint | 795 | 60.27% | +4.40 | 80 | 22 |
| [13](13_finetuned-model+vote+gate+checker_glm-30b_no-hint/README.md) | Trained AI + majority vote + selection of hard questions (30%) + large AI marker, none hint | 798 | 60.50% | +4.62 | 100 | 39 |
| [14](14_finetuned-model+vote+gate+checker_glm-30b_short-hint/README.md) | Trained AI + majority vote + selection of hard questions (30%) + large AI marker, short hint | 807 | 61.18% | +5.31 | 106 | 36 |
| [15](15_finetuned-model+vote+gate+checker_glm-30b_full-hint/README.md) | Trained AI + majority vote + selection of hard questions (30%) + large AI marker, full hint | 826 | 62.62% | +6.75 | 124 | 35 |
| [16](16_finetuned-model+vote+gate+checker_glm-30b_full-hint_37pct/README.md) | Trained AI + majority vote + selection of hard questions (37.3%) + large AI marker, full hint | 843 | 63.91% | +8.04 | 148 | 42 |
| [17](17_finetuned-model+vote+smart-gate+checker_glm-30b_full-hint/README.md) | Trained AI + majority vote + smart selection of hard questions (30%) + large AI marker, full hint | 837 | 63.46% | +7.58 | 134 | 34 |
| [18](18_finetuned-model+smart-gate+checker_qwen-9b_full-hint/README.md) | Trained AI + smart selection of hard questions (30%) + medium AI marker, full hint | 861 | 65.28% | +9.40 | 136 | 12 |
| [19](19_finetuned-model+vote+smart-gate+checker_qwen-9b_full-hint/README.md) | **Trained AI + majority vote + smart selection of hard questions (30%) + medium AI marker, full hint — BEST RESULT** | 903 | **68.46%** | +12.59 | 190 | 24 |
| — | *Upper limit: counted right if any of the three attempts was right* | 962 | *72.93%* | *+17.06* | — | — |

---

## Table 2 — What each system costs

| # | Attempts per question | Tokens written per question | Tokens read per question | Sent to marker | Marker calls | Marker tokens per question | Time per question* | Time for all 1,319* |
|---|---|---|---|---|---|---|---|---|
| [01](01_base-model-only/README.md) | 1.00 | 120.6 | 1617.7 | 0 (0%) | 0 | 0.0 | 13.7 s † | 5 h 00 min † |
| [02](02_finetuned-model-only/README.md) | 1.00 | 125.9 | 87.7 | 0 (0%) | 0 | 0.0 | 14.3 s | 5 h 14 min |
| [03](03_finetuned-model+blind-retry/README.md) | 2.00 | 258.4 | 175.3 | 0 (0%) | 0 | 0.0 | 29.3 s | 10 h 44 min |
| [04](04_finetuned-model+majority-vote/README.md) | 3.00 | 389.8 | 263.0 | 0 (0%) | 0 | 0.0 | 44.2 s | 16 h 11 min |
| [05](05_finetuned-model+checker-marks-all-answers_glm-30b/README.md) | 1.00 | 125.9 | 87.7 | 1,319 (100%) | 1,319 | 499.1 | 16.3 s | 5 h 59 min |
| [06](06_finetuned-model+checker-marks-all-answers_qwen-9b/README.md) | 1.00 | 125.9 | 87.7 | 1,319 (100%) | 1,319 | 565.1 | 15.3 s | 5 h 35 min |
| [07](07_finetuned-model+perfect-checker-marks-all-answers/README.md) | 1.00 | 125.9 | 87.7 | 1,319 (100%) | 1,319 | 0.0 | 14.3 s | 5 h 14 min |
| [08](08_finetuned-model+gate+checker_glm-30b_no-hint/README.md) | 3.46 | 476.8 | 399.1 | 396 (30%) | 712 | 300.9 | 55.1 s | 20 h 12 min |
| [09](09_finetuned-model+gate+checker_glm-30b_short-hint/README.md) | 3.46 | 470.8 | 405.4 | 396 (30%) | 712 | 297.9 | 54.5 s | 19 h 57 min |
| [10](10_finetuned-model+gate+checker_glm-30b_full-hint/README.md) | 3.45 | 466.2 | 415.1 | 396 (30%) | 712 | 296.0 | 53.9 s | 19 h 46 min |
| [11](11_finetuned-model+gate+checker_glm-30b_full-hint_37pct/README.md) | 3.57 | 484.5 | 451.9 | 492 (37%) | 889 | 369.0 | 56.3 s | 20 h 37 min |
| [12](12_finetuned-model+smart-gate+checker_glm-30b_full-hint/README.md) | 3.48 | 472.7 | 426.2 | 396 (30%) | 728 | 307.5 | 54.7 s | 20 h 03 min |
| [13](13_finetuned-model+vote+gate+checker_glm-30b_no-hint/README.md) | 3.46 | 476.8 | 399.1 | 396 (30%) | 712 | 300.9 | 55.1 s | 20 h 12 min |
| [14](14_finetuned-model+vote+gate+checker_glm-30b_short-hint/README.md) | 3.46 | 470.8 | 405.4 | 396 (30%) | 712 | 297.9 | 54.5 s | 19 h 57 min |
| [15](15_finetuned-model+vote+gate+checker_glm-30b_full-hint/README.md) | 3.45 | 466.2 | 415.1 | 396 (30%) | 712 | 296.0 | 53.9 s | 19 h 46 min |
| [16](16_finetuned-model+vote+gate+checker_glm-30b_full-hint_37pct/README.md) | 3.57 | 484.5 | 451.9 | 492 (37%) | 889 | 369.0 | 56.3 s | 20 h 37 min |
| [17](17_finetuned-model+vote+smart-gate+checker_glm-30b_full-hint/README.md) | 3.48 | 472.7 | 426.2 | 396 (30%) | 728 | 307.5 | 54.7 s | 20 h 03 min |
| [18](18_finetuned-model+smart-gate+checker_qwen-9b_full-hint/README.md) | 3.44 | 464.1 | 442.8 | 396 (30%) | 728 | 345.8 | 53.1 s | 19 h 28 min |
| [19](19_finetuned-model+vote+smart-gate+checker_qwen-9b_full-hint/README.md) | 3.44 | 464.1 | 442.8 | 396 (30%) | 728 | 345.8 | 53.1 s | 19 h 28 min |

\* **Time is estimated, not measured.** None of the full runs recorded how long it took.
Time is built from speeds that *were* measured: the small AI writes
**8.8 tokens per second**, the medium marker takes
**1.00 s** per check, and the large marker **2.07 s**
(sources below). Estimated time = tokens written ÷ writing speed + number of checks ×
time per check. Measured on an RTX 4080 SUPER; ordinary hardware would be slower.

† This estimate does not hold. The writing speed was measured with the training add-on
loaded; the untrained model runs without it, and its real run took about
3 h 19 min (table below).

### How accurate are the time estimates?

Three jobs ran back to back from a script, so the gap between their log entries is
close to how long they really took. Each is compared with the same estimation method,
applied to what that job actually had to do (earlier answers were reused from disk):

| Job | Estimated | Real (from the log) | Difference |
|---|---|---|---|
| One full set of 1,319 answers (second sample run) | 5 h 30 min | 5 h 19 min | +3.4% |
| System 09 as run: retries and new checks only | 3 h 32 min | 3 h 28 min | +2.1% |
| System 10 as run: retries and new checks only | 3 h 21 min | 3 h 16 min | +2.5% |
| System 01, the untrained model | 5 h 00 min | 3 h 19 min | **+51% — does not hold** |

So the estimates are reliable to within a few percent for every system built on the
trained AI, and should not be used for system 01.

### Two different "times"

The column above is how long each **system** would take to answer all 1,319 questions
from scratch. That is the number to compare systems by. The **experiments** themselves
ran faster, because they reused answers already saved to disk: systems 13–17 and 19 are
not separate runs at all, but the runs of 08–12 and 18 scored a second way.

**About "marker tokens".** In this project the marker ran on the same machine, so
nothing actually left it. The column counts the tokens that *would* travel to an
outside service if the marker were one — which is the cost the thesis argues about.

---

## Table 3 — How good each marker is

These three experiments only grade answers; nothing is retried, so they do not change
the score. They measure the marker itself.

| # | Marker | Wrong answers caught | Right answers wrongly failed | Correct when it said 'wrong' | Time per check |
|---|---|---|---|---|---|
| [05](05_finetuned-model+checker-marks-all-answers_glm-30b/README.md) | a large AI marker (GLM-4.7-Flash, 30 billion) | 83.5% | 25.5% | 72.1% | 2.07 s |
| [06](06_finetuned-model+checker-marks-all-answers_qwen-9b/README.md) | a medium AI marker (Qwen3.5-9B, 9 billion) | 88.8% | 18.6% | 79.1% | 1.00 s |
| [07](07_finetuned-model+perfect-checker-marks-all-answers/README.md) | a perfect marker that is shown the answer key | 100.0% | 0.0% | 100.0% | instant |

The 2-billion-parameter AI checking its own work was only tested on 30 questions, so it
has no folder here: it caught 22% of wrong answers and wrongly failed 42% of right ones
(`docs/THESIS_DOSSIER.md` §5.5.1).

---

## Measured speeds these estimates use

| What | Speed | Measured on | Source |
|---|---|---|---|
| Trained small AI writing an answer | 8.83 tokens/s | 40 questions | `reports/latency_benchmark.json` |
| Medium marker (Qwen3.5-9B) checking one answer | 1.00 s | 40 checks | `reports/latency_benchmark.json` |
| Large marker (GLM-4.7-Flash) checking one answer | 2.07 s (44.6 tokens/s) | 729 checks | llama-server log, `docs/THESIS_DOSSIER.md` §5.9 |
| Training the small AI | 4,080 s (68 min) | the full training run | `reports/experiment_log.jsonl` |

## What is deliberately not in these tables

- **The 30-question self-check** (above) — too small to stand beside full runs.
- **The Qwen2.5-1.5B solver experiment** — stopped after 51 of 1,319 questions and not
  resumed, so it has no result (`docs/THESIS_DOSSIER.md` §5.10).
- **Smoke tests** of 3–20 questions, run only to check the code worked.
