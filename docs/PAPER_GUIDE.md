# Thesis Paper Guide

**Read this first. It is written for the people writing the paper, not for engineers.**

This document explains, in plain language, what we built, what we measured, what
every number means, and what an examiner is likely to ask. Everything you need for
the written report is here.

There is a second document, `docs/THESIS_DOSSIER.md`. That is the technical record —
it says exactly which file and which command produced every number. Use it when you
need to check a figure. You do not need to read it to write the paper.

---

## Contents

1. [The one-paragraph summary](#1-the-one-paragraph-summary)
2. [The problem we set out to solve](#2-the-problem-we-set-out-to-solve)
3. [How the system works](#3-how-the-system-works)
4. [Every result, with what it means](#4-every-result-with-what-it-means)
5. [The three findings the paper is built on](#5-the-three-findings-the-paper-is-built-on)
   - [5.4 Engineering quality](#54-engineering-quality--worth-a-paragraph-in-the-paper)
6. [Dictionary of terms](#6-dictionary-of-terms)
7. [What goes in each section of the paper](#7-what-goes-in-each-section-of-the-paper)
8. [Limitations we must state ourselves](#8-limitations-we-must-state-ourselves)
9. [Defence questions and how to answer them](#9-defence-questions-and-how-to-answer-them)
10. [Where everything lives](#10-where-everything-lives)

---

## 1. The one-paragraph summary

We took a small AI model that can run on an ordinary laptop, trained it to solve
grade-school maths problems, and then built a system where a second, larger model
checks its work — but only on the questions where the small model is probably
wrong, so the expensive checking stays rare. Accuracy went from **36.3%** before
training, to **55.9%** after training, to **68.5%** with the checking system. Along
the way we found that the obvious way to build such a system does not work, that a
9-billion-parameter checker beats a 30-billion one, and that the ceiling on this
whole approach is **72.9%**.

---

## 2. The problem we set out to solve

Large AI models are accurate but need expensive hardware and an internet
connection. Small models run on ordinary machines — old laptops, office desktops,
phones — but make more mistakes.

**Our question: can a small local model reach useful accuracy by occasionally
asking a bigger model for help, while keeping that help rare enough to be
practical?**

The cost that matters to us is **time to answer** and **how much data leaves the
device**. Not money, not electricity. If the system has to phone home for every
question, it has failed — you may as well have used the big model directly.

### The five research questions

| | Question | Where it is answered |
|---|---|---|
| **RQ1** | Does fine-tuning a small model on maths actually help? | §4.1 — yes, 36.3% to 55.9% |
| **RQ2** | Can the device tell, on its own, when it is probably wrong? | §4.3 — yes, quite well |
| **RQ3** | Do we even need a checker, or would a free trick do? | §4.2 and §5.1 |
| **RQ4** | Does the checking system improve accuracy? | §5.1 and §5.3 |
| **RQ5** | How strong does the checker have to be? | §5.2 — smaller than we assumed |

---

## 3. How the system works

Six steps, per question.

**Step 1 — The small model answers the question three times.** Once normally,
twice with randomness switched on so it explores different approaches.

**Step 2 — We compare the three answers.** Three outcomes:

| Situation | How many of our 1,319 questions |
|---|---|
| All three agree | 427 |
| Two agree, one differs | 400 |
| All three differ | 492 |

**Step 3 — If there is a majority, take it.** When all three agree, or when two of
three agree, we keep that answer and stop. Nothing leaves the device. That covers
**923 of 1,319 questions (70%)**.

**Step 4 — If all three differ, consider sending it up.** There is no majority to
take, so the model has genuinely failed to settle. We rank these by how uncertain
the model was and send the **396** least confident to the checker.

**Step 5 — The checker marks the answer.** It sees the question and the model's
working — but **never the correct answer**. It replies with a verdict and, if it
rejects, one sentence naming the first wrong step and one sentence explaining the
right reading. It rejected **332 of the 396**.

**Step 6 — The model tries again with the hint.** Its new answer is final.

### Why steps 3 and 5 are both needed

This is the heart of the paper, so it is worth being precise.

The two mechanisms fix **different questions**:

- **Taking the majority** fixes cases where the model actually knew the answer but
  its first attempt was unlucky.
- **The checker** fixes cases where the model misunderstands the question and gets
  it wrong however many times you ask.

Neither covers the other. Using only the checker and ignoring the majority vote
throws away free accuracy. That was our original design, and it did not work.

---

## 4. Every result, with what it means

All numbers are on the **complete GSM8K test set: 1,319 questions.** No sampling,
no subsets. Every method was tested on the same questions, so comparisons are made
question-by-question rather than average-against-average.

### 4.1 Fine-tuning worked (RQ1)

| Condition | Correct | Accuracy | Words generated per question |
|---|---|---|---|
| Untrained model, shown 8 worked examples first | 479 | **36.32%** | 1,738 |
| After our fine-tuning, shown nothing | 737 | **55.88%** | 214 |

**Plus 19.6 percentage points.** The model also became **8x more concise**, because
it learned the answer format instead of needing to be shown it every time.

> **Why "shown 8 worked examples"?** Without examples the untrained model does not
> understand the task at all and scores near zero. Comparing against that would
> flatter us. We gave the baseline the strongest fair help we could.

### 4.2 What the model can do with no checker at all (RQ3)

Before adding a checker we measured what the model manages on its own. These cost
nothing — no cloud, no second model.

| Method | Correct | Accuracy | What it is |
|---|---|---|---|
| One answer | 737 | 55.88% | Just ask once |
| Ask again, keep the second answer | 658 | **49.89%** | Retrying blindly makes things **worse** |
| Ask 3 times, take the majority | 779 | **59.06%** | The free trick — our real competition |
| Ask 3 times, count it right if **any** is right | 962 | **72.93%** | **The ceiling** |

Two things to take from this table.

**Blind retrying is harmful.** Simply asking again and keeping the new answer loses
6 points. Any gain we report has to beat this, or we have only shown that random
resampling helps.

**72.93% is a hard ceiling.** If none of the three attempts contains the right
answer, no checker on earth can find it. Everything we build is a race towards that
number, not past it. Going higher means generating more attempts, which makes the
system slower — the exact cost we are trying to avoid.

### 4.3 Can the device tell when it is wrong? (RQ2)

Before sending a question up, the device must guess which questions are worth
sending. We tested four signals it can compute by itself.

Scores below are **AUC** — the chance the signal ranks a wrong answer as more
suspicious than a right one. 0.50 means worthless, 1.00 means perfect.

| Signal | What it costs the device | Score |
|---|---|---|
| Length of the answer | Nothing | 0.676 |
| The model's own confidence | One quick re-read | 0.715 |
| Whether the three answers disagree | Two extra answers | 0.840 |
| **Disagreement, with confidence breaking ties** | Two extra answers | **0.869** |

The last row is ours and is worth explaining, because it is a contribution.

Disagreement between three answers can only take three values — all agree, two
agree, all differ. That is very coarse. It cannot tell you which of the 492
all-differ questions is *most* hopeless. Confidence is a smooth number that can.

So we rank by disagreement first, and use confidence only to break ties. At a 10%
budget this catches **125 errors instead of 105**, and **95% of what it sends up is
genuinely wrong**.

> **Important detail for the paper.** We tried letting confidence *outweigh*
> disagreement. It scored **worse** than disagreement alone. Confidence must only
> break ties. This also means there is no weight we tuned — and therefore no knob we
> could have fitted to the test data. Say this explicitly; examiners look for it.

### 4.4 How good are the checkers? (RQ5)

Each checker marked all 1,319 answers right or wrong. We compared its marks to the
truth.

| Checker | Size | Of 582 wrong answers, caught | Of 737 right answers, wrongly failed | Seconds per check |
|---|---|---|---|---|
| The small model checking itself | 2B | 22% | 42% | 13.0 |
| **Qwen 3.5** | **9B** | **89%** | **19%** | **1.23** |
| GLM 4.7 Flash | 30B | 84% | 26% | 2.22 |

*(The 2B row comes from a 30-question trial, not the full set — see §8.6.)*

**The 9B beats the 30B on every measure and is nearly twice as fast.** This was not
what we expected, and it is one of the paper's findings.

### 4.5 The full system

| Configuration | Correct | Accuracy | Cloud words per question |
|---|---|---|---|
| Model alone | 737 | 55.88% | 0 |
| Free trick (majority of 3) | 779 | 59.06% | 0 |
| Checker only, no hint | 756 | 57.32% | 301 |
| Checker points at the error | 765 | 58.00% | 298 |
| Checker points and explains | 784 | 59.44% | 296 |
| Add majority voting underneath | 826 | 62.62% | 296 |
| Send more questions up (37%) | 843 | 63.91% | 369 |
| Use our smarter gate | 837 | 63.46% | 308 |
| **Swap in the 9B checker** | **903** | **68.46%** | 346 |
| *Ceiling* | *962* | *72.93%* | — |

**Final result: 68.46%, which is 93.9% of everything that was achievable.**

### 4.6 Does more guidance help?

We tested three levels of how much the checker may say back:

| Level | What the model receives | Accuracy | Extra cloud words |
|---|---|---|---|
| None | "Wrong, try again" | 60.50% | 4 |
| Short | Plus the wrong step named | 61.18% | 14 |
| Full | Plus the correct reading explained | **62.62%** | 24 |

More guidance is consistently better, and 24 words is cheap. *(These use the 30B
checker with voting underneath, so they are comparable to each other.)*

**The checker is forbidden from stating the final number.** It points at the mistake
and makes the small model redo the arithmetic itself. Otherwise the big model is
just answering the question and the whole argument collapses. This rule is enforced
in the checker's instructions and deserves a sentence in the paper.

### 4.7 How long does a question actually take?

Everything above measures **words**. This measures **seconds**, which is what our
argument is really about.

| Step | Seconds |
|---|---|
| The small model writes one answer | **15.1** |
| The 9B checker marks one answer | **1.0** |
| The 30B checker marks one answer | 2.1 |

Put together, for one question:

| | Seconds |
|---|---|
| Just ask once, no system at all | 15.1 |
| Stays on the device (3 answers, majority vote) | 45.4 |
| Goes to the checker | 59.1 |
| **Average over all questions** | **49.5** |

**The whole system is 3.3x slower than a single answer.** That is the honest price
of the accuracy gain, and it belongs in the paper next to the accuracy table.

**Where that time goes is the surprise:**

| | Seconds | Share |
|---|---|---|
| Asking the model three times | 45.4 | **92%** |
| Retrying after a rejection | 3.8 | 8% |
| **Waiting for the checker** | **0.3** | **0.6%** |

**The cloud is 0.6% of the waiting.** The expensive part is not sending questions
away — it is *asking the model three times in order to decide whether to send them
away.* The gate costs far more than everything it gates.

> **This changes one sentence we might otherwise have written.** It would be wrong
> to justify rare escalation by saying the checker is slow — it isn't, it's the
> fastest part. Justify it by **how much data leaves the device**: 70% of questions
> never leave at all. That claim is about privacy and independence, and the evidence
> fully supports it.

**One more oddity worth a line in the paper.** The 2-billion model is **15x slower
per answer** than the 9-billion checker. Not because it is a worse model — because
of the software each runs on. Ours uses a general-purpose framework; the checker
uses one built for fast serving. Putting the small model on the same software as the
checker is the biggest speed improvement available to us, and we have not tried it.

> **Caveat to state whenever these numbers appear.** They were measured on a
> desktop graphics card (RTX 4080 SUPER), **not** on the weak hardware our thesis
> describes. On an old laptop everything local is slower — which pushes the cloud's
> share *below* 0.6%, so it strengthens the argument rather than weakening it. A
> real edge measurement needs the benchmark run on such a machine, and it only needs
> the small model, not the checker.

---

## 5. The three findings the paper is built on

### 5.1 Finding 1 — the obvious design does not work

**Running the checker *instead of* the free majority-vote trick is worthless.**

With the 30B checker the system scored 59.44% against the free trick's 59.06%. Five
answers out of 1,319. Statistically indistinguishable (*p* = 0.75 — meaning a 75%
chance of seeing a gap this big by luck alone).

Worse, the costs are not comparable in our favour. Deciding what to send up already
requires generating three answers — exactly what the free trick needs. So the system
pays the free trick's entire local cost, **plus** 712 cloud calls, and buys nothing
measurable.

**The fix: stack them.** Take the majority vote first, and send up only the
questions where there was no majority. That gains **42 answers at identical cloud
cost**, because the three answers were already being generated and the old design
simply discarded the vote.

This is the honest headline and it must appear **before** any comparison against the
untrained model. "Plus 12.6 points over the fine-tuned model" is true and, on its
own, misleading.

### 5.2 Finding 2 — bigger checkers are not better checkers

Going 2B to 9B and 9B to 30B does **not** improve one quantity smoothly:

- **2B to 9B** buys the ability to *notice* errors (22% to 89% caught) and buys
  nothing on wrongly-failed answers, which stay at 42%.
- **9B to 30B** buys the opposite, and badly: error-catching barely moves, while
  wrongly-failed answers get **worse** (19% to 26%).

**The 30B overshoots the task.** Its extra capacity shows up as over-rejection,
which is the expensive mistake — a wrongly-failed answer triggers a retry that was
never needed, and often makes the answer worse.

The practical claim: *there is a model size that fits this job, and it is smaller
than we assumed.* The 9B is also a 5.6 GB file against 16.3 GB, which matters for
deployment.

### 5.3 Finding 3 — Finding 1 was the checker's fault

This is the most important thing to get right in the paper, because it changes the
story.

With the 30B checker the system never beat the free trick unless stacked on top of
it. **With the 9B checker it beats the free trick on its own** — 65.28% against
59.06%, *p* < 0.000001.

So the null result in Finding 1 was **a property of one particular checker, not of
the design**. That is a much stronger paper: instead of "the obvious approach fails
and here is a workaround", we can say "the approach works once the checker is good
enough, and here is what good enough means."

Stacking still helps (plus 42, unchanged) — but it is no longer what rescues the
method.

### 5.4 Engineering quality — worth a paragraph in the paper

Examiners of an engineering thesis care whether the results can be trusted and
reproduced. Three things support that, and all are worth mentioning briefly.

**Every result is reproducible from stored files.** `REPRODUCE.md` lists the exact
commands, in order, with the expected output at each stage. The raw answers are in
the repository, so anyone can re-run the analysis and check our numbers without a
graphics card.

**We built a safeguard that turned out to be necessary.** The checkers' verdicts are
cached so that different versions of the system are guaranteed to reject *exactly the
same answers* — otherwise a comparison between them would be measuring which
questions happened to be retried, not the thing we changed. When we re-ran the same
configuration a day later, all 396 shared verdicts replayed identically, which
confirms the safeguard worked.

**We found and fixed three bugs in our own reproduction path**, all of the kind that
only appear when someone actually follows the instructions:

1. A checker model was listed under a filename that does not exist — anyone following
   our steps would have hit a download error.
2. The system reported the *configured* checker name rather than the one actually
   running, so a log could credit the wrong model.
3. Worse, that mislabelling could have silently corrupted the verdict cache — writing
   one checker's marks under another's name and invalidating the comparison that our
   main finding rests on. We added a guard that refuses to start when the running
   checker does not match the requested one, and tested that the guard actually fires.

The code has **138 automated tests** covering answer extraction, the gating logic, the
statistics, and the failure modes of the checker — including a test asserting the
checker can never see the correct answer.

---

## 6. Dictionary of terms

Terms you will meet in the code, the reports, and the technical dossier.

| Term | Plain meaning |
|---|---|
| **GSM8K** | The dataset: 1,319 grade-school maths word problems with known answers. A standard benchmark, so our numbers are comparable to published work. |
| **Fine-tuning** | Further training of an existing model on our specific task. |
| **QLoRA** | The efficient fine-tuning method we used. It trains a small add-on rather than the whole model, so it fits on one consumer graphics card. |
| **Adapter** | The small add-on file QLoRA produces. The original model is unchanged; the adapter loads on top of it. |
| **Cascade** | The whole system: answer locally, escalate rarely, retry with a hint. |
| **Gate** | The rule deciding which questions get sent to the checker. |
| **Escalate** | To send a question to the checker. |
| **Supervisor / verifier / judge / checker** | All the same thing: the bigger model that marks the answer. The code says "supervisor"; the paper should pick one word and use it throughout. |
| **Verdict pass** | Having the checker mark all 1,319 answers with no retries, purely to measure how good the checker is. |
| **Self-consistency** | The free trick: ask 3 times, take the majority. |
| **Stacked** | Majority vote first, checker only on the leftovers. |
| **pass@3** | Counting a question right if **any** of three attempts was right. Not a usable method — it needs the answer key. It is the ceiling. |
| **Recall** | Of the answers that were wrong, the share the checker caught. |
| **Precision** | Of the answers the checker rejected, the share that really were wrong. |
| **False-reject rate** | Of the answers that were **right**, the share the checker wrongly failed. Our most costly error. |
| **AUC** | A 0-to-1 score for how well a signal separates wrong answers from right ones. 0.5 is useless, 1.0 is perfect. |
| **McNemar test** | The statistical test we use to ask whether one method genuinely beats another, comparing them question by question. |
| **p-value** | The chance of seeing a difference this large by luck if the two methods were really equal. Below 0.05 is conventionally "real". |
| **Flip matrix** | A count of answers a method turned from wrong to right, and from right to wrong. We always report both. |
| **Token** | Roughly a word-piece. How model cost is measured. |
| **Greedy decoding** | Always taking the most likely next word — no randomness, identical output every time. |
| **Temperature** | How much randomness to allow. We used 0 for the main answer and 0.7 for the two extra attempts. |
| **Quantisation (4-bit, GGUF)** | Compressing a model so it fits in memory, at a small accuracy cost. |
| **MoE (mixture of experts)** | A model that stores many parameters but uses only a few per word. The 30B checker uses only 3B at a time, which is why it is fast for its size. |

---

## 7. What goes in each section of the paper

### Introduction
The problem: capable models need hardware ordinary users do not have. State the
constraint honestly — **time to answer** and **data leaving the device**, not
electricity. Give the headline: 36.3% to 55.9% to 68.5%.

### Related work
Cover parameter-efficient fine-tuning (QLoRA), self-consistency and majority
voting, model cascades and routing, and LLM-as-a-judge with its known biases. Our
contribution sits at the join of the last three.

### Method
Describe the six steps in §3. Include a pipeline diagram. Make these explicit,
because examiners look for them:

- The checker never sees the correct answer.
- The checker may not state the final number, only point at the error.
- The three answers were being generated anyway, for the gate.

### Experimental setup
Model `gemma-4-E2B-it` (2B). QLoRA fine-tuning, 4-bit, LoRA rank 16 / alpha 32 /
dropout 0.05, 3 epochs, effective batch size 16, learning rate 2e-4, sequence length
1024. Training took 68 minutes. Answers are generated greedily; the two extra samples
and all retries use temperature 0.7. One RTX 4080 SUPER (16 GB) and
32 GB of system memory. Full GSM8K test set, 1,319 questions. The checkers run
locally as compressed files through a local server — **no cloud service was used**,
which strengthens the deployment story and makes everything reproducible.

### Results
Use the tables in §4, in this order:

1. Fine-tuning gain (§4.1)
2. **The free controls (§4.2)** — put these early, they frame everything
3. Gate quality (§4.3)
4. Checker quality (§4.4)
5. The full system (§4.5)
6. Guidance levels (§4.6)

The headline figure is `reports/figures/pareto.png`: accuracy against cloud cost,
with the free trick drawn as a line. Every configuration below that line paid for
nothing.

### Discussion
Lead with Finding 1, then Finding 3. The narrative: *we built the obvious system,
measured it honestly against a free alternative, found it did not win, diagnosed
why, and fixed it in two independent ways.* That story is worth more than a system
that worked first time.

### Limitations
See §8. State them ourselves, before anyone else does.

### Conclusion
68.46%, at 93.9% of the achievable ceiling, with 70% of questions never leaving the
device.

---

## 8. Limitations we must state ourselves

**8.1 The ceiling is 72.93%, and we cannot pass it.** We are bounded by whether the
small model produces the right answer in three tries. Beating it requires more
attempts, which costs the response time we are trying to protect.

**8.2 We only measure the final number, not the reasoning.** A model can reach the
right answer through faulty working. We have no ground truth for reasoning quality.
*This is the one gap a hand-labelled sample would close, and it is still
outstanding.*

**8.3 Retries are unpredictable question-by-question.** We re-ran the same
configuration and only 32% of retried answers came out the same. Overall accuracy
was statistically identical (*p* = 0.83), so our *rates* are reliable — but no claim
about *which* question a hint repaired is safe. Write about rates.

**8.4 Small effects cannot be detected here.** Because re-running moves roughly 40
answers in each direction by chance, an improvement must catch about 40 extra errors
to be visible at all. Our smarter gate is measurably a better gate, but its accuracy
gain (plus 11) sits inside that noise. Report it as a gate improvement and a cost
saving, not as an accuracy claim.

**8.5 Anchoring.** The retry prompt shows the model its own rejected answer, which
may lock it into the same misreading. We observed this in early testing. Not yet
measured properly.

**8.6 The 2B self-checking result is from 30 questions, not 1,319.** It is
directional only. We include it because the gap is enormous (22% against 89%), but
it must be labelled as a preliminary trial. A related lesson we learned the hard
way: a 30-question trial of the 9B checker put its false-reject rate at 42%, and the
full run put it at 19%. Small trials decide whether to spend the time, never what to
conclude.

**8.7 The stacking idea was found by looking at test results.** We noticed the two
mechanisms fixed different questions and combined them. The idea is principled and
we tuned no parameters, but strictly the *decision to try it* came from the test set.
Disclose this.

**8.8 The checkers are not perfectly reproducible.** The software that runs them is
not bit-exact, so repeated runs vary slightly. Our effects are far larger than that
drift.

**8.9 Our speed numbers come from a desktop graphics card, not a weak device.**
See §4.7. The local figures are a floor — real edge hardware is slower. We report
them as measured and say where they came from.

**8.10 The 9B checker's replies were sometimes cut off.** Seven percent of its replies
hit the length limit, and our system treats an unreadable reply as approval. We
measured the cost: about 2 answers. So 68.46% is a slight **under**-estimate.

---

## 9. Defence questions and how to answer them

**"Why not just use the big model for everything?"**
Because it is not on the device. Our claim is about how often data must leave the
machine and how long an answer takes. Seventy percent of questions never leave. If
you already have the big model locally, you did not have our problem.

**"Isn't this just self-consistency with extra steps?"**
We measured exactly that, and for one checker it was a fair criticism — the system
did not beat majority voting (*p* = 0.75). We diagnosed why, and with a better
checker the system beats voting on its own (*p* < 0.000001). We are the ones who
found the objection and reported it.

**"How do you know the checker isn't just seeing the answer?"**
The function that builds the checker's prompt takes no answer argument at all — there
is no code path by which it could. An automated test asserts this. Only the
deliberately-fake "perfect oracle" checker receives the answer, and it exists solely
to verify our scoring code.

**"Did you tune anything on the test set?"**
The checker's instructions were tuned on 150 **training** questions, never test ones.
Our gate has no weight parameter to tune — confidence breaks ties and cannot outrank
disagreement. One honest caveat: the idea of combining voting with checking came from
observing test results, though no parameter was fitted. See §8.7.

**"Why 30%? Isn't that arbitrary?"**
Partly, and we found this ourselves. With three answers the disagreement signal has
only three possible values, so the only threshold it can truly express is 37.3%. We
ran that too. Our smarter gate is continuous and removes the problem entirely.

**"Your improvement is only a few percent."**
Against the fine-tuned model it is plus 12.6 points. Against the strongest free
alternative it is plus 9.4 points, *p* < 0.000001. And the maximum possible was plus
17.1, so we captured 94% of what was available.

**"Why does a 9B model beat a 30B model?"**
Measured, not assumed. The 30B over-rejects — it fails 26% of correct answers against
the 9B's 19%. Extra capacity appears to make it more willing to object, and objections
to correct answers are the costly error. We report it as an empirical finding at this
task and scale, not as a general law.

**"What happens when the checker is wrong?"**
Reported explicitly in the flip matrix — we always give answers broken alongside
answers fixed, never a net figure alone. Our checker's instructions were deliberately
tightened after we found the first version rejected so much that retrying lost more
than it gained.

**"How much slower is your system than just answering once?"**
3.3 times — 49.5 seconds against 15.1, measured. And 92% of that is the small model
answering three times, not the checker, which accounts for 0.6%. We measured this
rather than assuming it, and it corrected our own framing: escalation is cheap in
time, and the reason to keep it rare is that data leaves the device, not that it is
slow.

**"Could this work on harder problems?"**
Unknown, and we should say so. GSM8K is grade-school arithmetic. The method depends on
the model sometimes producing the right answer among several attempts; on problems
where it never does, the ceiling collapses and no checker helps.

**"Why is your baseline shown 8 examples when your model is shown none?"**
Because the untrained model cannot do the task without examples, and comparing against
a near-zero score would inflate our result. We chose the harder, fairer comparison.

**"How many questions did you test on?"**
All 1,319 in the GSM8K test set, for every reported method. The only small-sample
numbers in this document are labelled as such (§8.6).

---

## 10. Where everything lives

| What | Where |
|---|---|
| **This guide** | `docs/PAPER_GUIDE.md` |
| Technical record, every number with its source | `docs/THESIS_DOSSIER.md` |
| How to re-run everything from scratch | `REPRODUCE.md` |
| The headline figure | `reports/figures/pareto.png` |
| Main results table | `reports/fydp3_summary.md` |
| Fine-tuning results | `reports/gsm8k_summary.md` |
| Checker quality | `reports/fydp3_verdict_qwen.md` |
| Speed measurements | `reports/latency_benchmark.json` |
| Raw answers, one file per stage | `outputs/predictions/` — see its own `README.md` |
| Log of every run ever made | `reports/experiment_log.md` |
| Full settings of record | `docs/THESIS_DOSSIER.md` §3.1 |

### Still to do

1. **Hand-label 100 examples for reasoning quality** — closes §8.2. This is the only
   remaining item that needs a person rather than a machine, and without it we cannot
   make any claim about *why* answers are right.
2. Three small ablations: anchoring, thinking-mode cost, and a cloud reference point.
3. Commit the repository, and revoke the access token in `Hugging face tokken.txt`.
