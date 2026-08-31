# What is in this folder

Every file is one line per question in **JSON Lines** format — a plain text file
where each line is one record you can read in a text editor. All the main files
have **1,319 lines**, one for every question in the GSM8K maths test set.

You do not need to open these to write the paper. The numbers are already
summarised in `reports/` and explained in `docs/PAPER_GUIDE.md`. This page is
here so that anyone browsing the repository can tell what they are looking at.

## Stage 1 — what the model answered

| File | What it holds |
|---|---|
| `01_answers_untrained_baseline.jsonl` | The model **before** fine-tuning, shown 8 worked examples first. The fair comparison point. |
| `02_answers_finetuned_try1_main.jsonl` | The fine-tuned model's main answer to each question. **The most important file here** — every later stage starts from it. |
| `03_answers_finetuned_try2.jsonl` | The same model asked again, with randomness turned on. |
| `04_answers_finetuned_try3.jsonl` | And a third time. |

Files 02, 03 and 04 are the "ask it three times" idea. Comparing them tells us
when the model is unsure of itself.

## Stage 2 — how sure the model was

| File | What it holds |
|---|---|
| `05_confidence_for_try1.jsonl` | How confident the model was in each answer in file 02. |
| `06_confidence_for_try2.jsonl` | Same, for file 03. |
| `07_confidence_for_try3.jsonl` | Same, for file 04. |

These are not new answers. They re-read answers the model had already written and
recover the certainty score it produced at the time and then discarded.

## Stage 3 — a checker marking the answers, with no retries

| File | What it holds |
|---|---|
| `08_checker_qwen9b_marks_all.jsonl` | The Qwen 9B checker marking all 1,319 answers right or wrong. Used to measure how good the checker is. |
| `09_checker_perfect_oracle_marks_all.jsonl` | A "perfect" checker that is allowed to see the real answer. Not a real option — it exists to prove the scoring code is correct. |

## Stage 4 — the full pipeline, end to end

These runs do the whole thing: answer, decide whether to escalate, get a verdict,
retry with a hint.

| File | What it holds |
|---|---|
| `10_pipeline_glm30b_hint_none.jsonl` | Checker says only "wrong, try again". |
| `11_pipeline_glm30b_hint_short.jsonl` | Checker also points at the wrong step. |
| `12_pipeline_glm30b_hint_full.jsonl` | Checker points at the wrong step **and** explains the right reading. |
| `13_pipeline_glm30b_hint_full_more_escalation.jsonl` | Same, but sends more questions to the checker (37% instead of 30%). |
| `14_pipeline_glm30b_hint_full_smart_gate.jsonl` | Same, but chooses *which* questions to send using confidence as well as disagreement. |
| `15_pipeline_qwen9b_BEST_RESULT.jsonl` | **The headline result: 68.5% correct.** Same as file 14 but with the Qwen 9B checker instead of the 30B one. |
| `16_pipeline_glm30b_no_gate_every_question.jsonl` | Sends **every** question to the checker. A cost comparison, not a proposal. |

"GLM 30B" and "Qwen 9B" are the two checker models. "Hint" is how much guidance the
checker is allowed to send back.

## `checks/` — development and sanity tests

Not results. These were used while building the pipeline: 1-question and
5-question smoke tests, a check that the model gives the identical answer when
re-run, and the tuning of the checker's instructions (done on **training**
questions, deliberately never on test questions).

## What each line contains

The pipeline files (stage 4) carry the most:

| Field | Meaning |
|---|---|
| `id` | Which question, 0 to 1318 |
| `question` | The maths problem |
| `gold_final_answer` | The correct answer |
| `escalated` | Was this question sent to the checker? |
| `attempts` | Every try, with the checker's verdict and hint |
| `accepted_final_answer` | The answer the system finally gave |
| `correct` | Was that answer right? |
| `supervisor_calls` | How many times the checker was asked |
| `supervisor_input_tokens` / `_output_tokens` | The cloud cost |
