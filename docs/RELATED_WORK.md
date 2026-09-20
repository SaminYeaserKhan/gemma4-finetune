# Related Work — the literature review, in plain language

**Who this is for:** the teammates writing the paper. You do not need to have read
any of these papers. Each entry below tells you, in ordinary words, what someone
else already did, why it matters to us, and the one sentence you can put in our
paper about it.

**Where this fits:** this file is the source for the *Related Work* section of the
thesis (see `docs/PAPER_GUIDE.md` §7). `docs/THESIS_DOSSIER.md` holds our own
numbers; this file holds everybody else's.

---

## 0. Before you read: what are we actually claiming?

You cannot judge whether a paper is "related" until you know what it is related
*to*. Here are the five things our thesis claims. Every paper below is filed
against one of them.

| # | Our claim | Our evidence |
|---|---|---|
| **C1** | A 2-billion-parameter model can be fine-tuned cheaply to be far better at grade-school maths. | 36.3% -> 55.9% on GSM8K |
| **C2** | Asking the same model three times and taking the most common answer is a strong, *completely free* baseline that many systems forget to compare against. | 55.9% -> 59.1% |
| **C3** | A checker model that only sees hard questions can push accuracy further, and *how good the checker is* matters more than people assume — and not in a straight line. | 59.1% -> 68.5%; the 9B checker beat the 30B one |
| **C4** | The checker only pays for itself when it is **stacked on top of** the free voting trick, not used instead of it. | Un-stacked: no significant gain. Stacked: significant, +42 answers at zero extra cost |
| **C5** | Deciding *which* questions to escalate works best by combining two signals the device already has, with no tuned weight. | Gate quality 0.840 -> 0.869 |

Now the prior work.

---

## Group A — Making a small model good at maths

### A1. GSM8K itself, and the original "use a checker" idea

**Cobbe et al., 2021 — "Training Verifiers to Solve Math Word Problems."**
[arXiv:2110.14168](https://arxiv.org/abs/2110.14168)

This is the paper that **created the dataset we use.** 8,500 grade-school word
problems, split into training and test; our 1,319 test questions come from here.
You must cite this — it is not optional.

It also invented the idea our whole thesis rests on. They generated many candidate
answers and trained a *second* model whose only job was to score which candidates
looked right. Doing that let a 6-billion-parameter model beat a 175-billion one.

> **Say in the paper:** "The idea that a separate verifier can substitute for raw
> model size originates with Cobbe et al. (2021), who introduced GSM8K alongside it."

**How we differ:** their verifier is *trained* on labelled data. Ours is an
off-the-shelf instruction model that we simply *ask*, with no training at all. That
is a weaker verifier but a much cheaper one — you can swap it by restarting a server.

### A2. Chain-of-thought — why the model writes out its working

**Wei et al., 2022 — "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models."**
[arXiv:2201.11903](https://arxiv.org/abs/2201.11903)

Showed that if you make a model write out its reasoning step by step instead of
blurting the answer, it gets dramatically better at maths. Every answer format in
our project is a chain of thought ending in `#### <number>`.

*Already cited in `reports/FYDP2_progress_report.md` for our 8-shot baseline.*

### A3 & A4. How we fine-tuned cheaply

**Hu et al., 2022 — "LoRA: Low-Rank Adaptation of Large Language Models."**
[arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

**Dettmers et al., 2023 — "QLoRA: Efficient Finetuning of Quantized LLMs."** NeurIPS 2023.
[arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

LoRA is the trick that lets you fine-tune a model by training a tiny bolt-on piece
instead of the whole thing. QLoRA adds compression on top, so the frozen model takes
a quarter of the memory. Together they are the reason our training ran in **68
minutes on one consumer graphics card** rather than needing a data centre.

> **Say in the paper:** "We fine-tune with QLoRA (Dettmers et al., 2023), which
> combines low-rank adapters (Hu et al., 2022) with 4-bit quantisation of the frozen
> base model, making single-GPU fine-tuning of a 2B model feasible."

### A5. The warning about small models and long reasoning

**Li et al., 2025 — "Small Models Struggle to Learn from Strong Reasoners."**
[arXiv:2502.12143](https://arxiv.org/abs/2502.12143)

Found that models of about 3 billion parameters and under do **not** reliably benefit
from being taught the long, elaborate reasoning of bigger models — they do better
with short, simple chains. Useful for us because it explains why our 2B model
plateaus, and defends our choice not to chase longer reasoning.

---

## Group B — Getting more out of one model at answer time

### B1. The free trick we compare everything against

**Wang et al., 2023 — "Self-Consistency Improves Chain of Thought Reasoning in Language Models."** ICLR 2023.
[arXiv:2203.11171](https://arxiv.org/abs/2203.11171)

**This is the single most important paper for our thesis**, because it is the
baseline that nearly killed our result.

The method: instead of asking once, ask the same question several times with a bit of
randomness, then take the answer that came up most often. No extra model, no labels,
no cost beyond the extra generations. On a 540B model it added **+17.9 points on
GSM8K**. On our 2B model it added +3.2 points (55.9% -> 59.1%).

> **Say in the paper:** "Self-consistency (Wang et al., 2023) is the natural free
> baseline for any system that samples more than once. We report it beside every
> cascade configuration, because our disagreement gate generates the samples it needs
> anyway — making the vote available at no additional cost."

**This connects directly to our claim C4.** Our headline finding is that a checker run
*instead of* voting is not statistically better than voting; a checker run *on top of*
voting is. A reviewer who knows this paper will ask exactly that question, and we
answer it before they ask.

### B2. Do not sample a fixed number of times — stop when they agree

**Aggarwal et al., 2023 — "Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs."** EMNLP 2023.
[ACL Anthology 2023.emnlp-main.761](https://aclanthology.org/2023.emnlp-main.761/)

Their observation is the same one behind our gate: **when the samples agree, the
answer is probably right, so stop; when they disagree, keep going.** They spend the
saved budget on more samples. We spend it on a checker instead.

> **Say in the paper:** "Aggarwal et al. (2023) use sample agreement to decide when to
> stop sampling; we use the same signal to decide when to escalate. Both treat
> disagreement as the cheapest available difficulty estimate."

### B3. Spending compute at answer time beats making the model bigger

**Snell et al., 2024 — "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters."**
[arXiv:2408.03314](https://arxiv.org/abs/2408.03314)

Shows that on problems a small model can *sometimes* get right, spending extra effort
at answer time can beat a model **14 times larger**. This is the strongest theoretical
backing for our entire approach, and a good line for the introduction.

They also found the right amount of extra effort depends on how hard the question is
— which is precisely what our gate is trying to detect.

### B4. Caution — the free trick is getting weaker

**"Self-Consistency Is Losing Its Edge: Diminishing Returns and Rising Costs in Modern LLMs," 2025.**
[arXiv:2511.00751](https://arxiv.org/abs/2511.00751)

Argues that on newer, stronger models the majority-vote trick buys less and less while
costing more. Worth one sentence: it says our +3.2 points is about what you should
expect from a small model, and that the trick's value shrinks as models improve —
which makes the checker layer *more* interesting over time, not less.

---

## Group C — Cascades and routing (only send the hard ones onward)

This is the family our system belongs to.

### C1. The founding paper

**Chen, Zaharia & Zou, 2023 — "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance."**
[arXiv:2305.05176](https://arxiv.org/abs/2305.05176)

Send every question to the cheap model first. Score how confident the answer looks.
Only if that score is too low, pay for the expensive model. Reported **up to 98% cost
reduction**.

> **Say in the paper:** "Our architecture is an LLM cascade in the sense of Chen et al.
> (2023): a cheap local model answers by default and an expensive one is invoked only
> on a scored subset."

**How we differ, and this is our main structural difference:** in FrugalGPT the
expensive model *answers*. In ours the expensive model **only judges** — it never
writes the answer, and it is forbidden from stating the final number. The small model
always produces the output. That keeps the deployment story intact (the answer is
generated on the device) and it is the reason we report *checker* tokens rather than
*answer* tokens as the cost.

### C2. What to actually use as the confidence score

**Gupta et al., 2024 — "Language Model Cascades: Token-level Uncertainty and Beyond."**
[arXiv:2404.10136](https://arxiv.org/abs/2404.10136)

The most technically relevant paper to our **gate**. They study which signal should
decide escalation, and identify a trap we hit: a whole-answer confidence score suffers
from **length bias** — long answers and short answers are scored unfairly against each
other.

> **Say in the paper:** "Gupta et al. (2024) document the length bias of sequence-level
> uncertainty in cascade deferral. Our confidence gate uses a length-normalised mean
> log-probability for this reason, and our best gate avoids relying on confidence
> alone."

**How we differ:** they *learn* a deferral rule from data. We deliberately do not — our
combined gate has **no tunable weight at all** (`gate.tie_broken_score` compresses the
secondary signal inside the smallest gap between primary values, so it can never
reorder them). That is a weakness in raw performance and a strength in honesty: there
is no parameter anyone could accuse us of having tuned on the test set.

### C3. Cascades that are allowed to give up

**Zellinger, Liu & Thomson, 2025 — "Cost-Saving LLM Cascades with Early Abstention."**
[arXiv:2502.09054](https://arxiv.org/abs/2502.09054)

Adds a third option to the cascade: answer, escalate, **or refuse**. Evaluated on six
benchmarks including GSM8K. Good to cite as the obvious extension we did not build —
see our limitations.

### C4. Routing instead of cascading

**Ong et al., 2024 — "RouteLLM: Learning to Route LLMs with Preference Data."**
[arXiv:2406.18665](https://arxiv.org/abs/2406.18665)

A trained router predicts *before generating anything* whether a question needs the big
model. Over 2x cost reduction.

> **Say in the paper:** "RouteLLM (Ong et al., 2024) routes on a learned predictor of
> question difficulty. Our gate instead scores the answer *after* it is produced, which
> costs a generation but requires no router training data and no second model resident
> on the device."

### C5. The small model drafts, the big one corrects — token by token

**Kim et al., 2023 — "Speculative Decoding with Big Little Decoder."**
[arXiv:2302.07863](https://arxiv.org/abs/2302.07863)

A neighbouring idea worth one sentence for contrast: the small model writes, and the
big model steps in *mid-sentence* when the small one looks unsure. Same
small-model-first instinct, but it needs both models loaded at once — which is exactly
what an edge device cannot do, and why we escalate whole questions instead.

---

## Group D — Using one model to check another

### D1. The standard reference, and the standard warning

**Zheng et al., 2023 — "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena."** NeurIPS 2023.
[arXiv:2306.05685](https://arxiv.org/abs/2306.05685)

The paper that made "let a model grade it" respectable. A strong judge agreed with
human graders **over 80%** of the time — about as often as humans agree with each
other.

It also catalogued the biases, and **two of them we have to address directly**:

- **Self-enhancement bias** — a model rates its own output too kindly. This is
  precisely why our 2B self-check arm failed (it approved almost everything), and we
  should cite this paper as the explanation rather than presenting it as a surprise.
- **Leniency** — judges say "fine" too readily. Our `preflight_supervisor.py` exists to
  measure exactly this before committing to a multi-hour run.

> **Say in the paper:** "The known self-enhancement bias of LLM judges (Zheng et al.,
> 2023) predicts the failure of our self-verification arm, and motivates our use of a
> distinct model family as the checker."

### D2. Grading the working, not just the answer

**Lightman et al., 2023 — "Let's Verify Step by Step."**
[arXiv:2305.20050](https://arxiv.org/abs/2305.20050)

Two ways to check an answer: judge the final result (*outcome supervision*) or judge
every intermediate step (*process supervision*). They showed step-by-step judging works
considerably better.

**This matters for us** because our checker is instructed to scrutinise the *reasoning*,
not just the number — a middle ground between the two. Cite this as the justification
for that prompt design, and as the direction future work should take.

### D3. Verifier scores can be folded into the vote

**"Not All Votes Count! Programs as Verifiers Improve Self-Consistency," 2024.**
[arXiv:2410.12608](https://arxiv.org/abs/2410.12608)

Filters out candidate answers a verifier rejects, then votes over what is left —
reported up to **+18% on GSM8K**. Closely related to our stacking result, and an honest
thing to point at: they weight the vote by verifier score, whereas we take the vote
first and escalate only the ties. Their design is probably stronger; ours is cheaper,
because it calls the checker on a fraction of questions rather than all.

**Also relevant:** *V-STaR: Training Verifiers for Self-Taught Reasoners*
([arXiv:2402.06457](https://arxiv.org/abs/2402.06457)), for the trained-verifier
alternative we did not pursue.

---

## Group E — The negative results that justify our design

These two papers are why our system uses a **separate** checker rather than asking the
model to check itself. They are the most useful papers in this whole file for the
defence, because they turn one of our "failures" into an expected result.

### E1. The big one

**Huang et al., 2024 — "Large Language Models Cannot Self-Correct Reasoning Yet."** ICLR 2024.
[arXiv:2310.01798](https://arxiv.org/abs/2310.01798)

Without outside feedback, models asked to revise their own reasoning **get worse, not
better.** Held for GPT-3.5 and GPT-4 across every benchmark tested.

> **Say in the paper:** "Our blind-retry control reproduces the finding of Huang et al.
> (2024): re-answering without external feedback reduced accuracy from 55.9% to 49.9%.
> This is why the cascade requires an external checker rather than a self-revision
> loop."

**That control of ours is now a replication of a published result on a model roughly
100x smaller.** That is worth stating plainly — it is a genuine, if small,
contribution.

### E2. How strong does the checker have to be?

**Zhang et al., 2024 — "Small Language Models Need Strong Verifiers to Self-Correct Reasoning."** ACL Findings 2024.
[arXiv:2404.17140](https://arxiv.org/abs/2404.17140)
*Yunxiang Zhang, Muhammad Khalifa, Lajanugen Logeswaran, Jaekyeom Kim, Moontae Lee, Honglak Lee, Lu Wang.*

**The closest published work to our research question 5.** They took small models (13B
and under), taught them to revise their answers, and found large gains **when paired
with a strong GPT-4 checker** but clear limits **when the model checked itself**.

> **Say in the paper:** "Zhang et al. (2024) establish that small reasoners require a
> strong external verifier. We extend this from a two-point comparison (self vs. GPT-4)
> to a four-point ladder — 2B, 9B, 30B, and an oracle — and find the relationship is
> **not monotonic**: the 9B checker produced fewer false rejections (0.186) than the
> 30B one (0.255)."

**This is where we can claim something genuinely new.** Their result says "bigger is
better than self." Ours says "bigger is better than self, *up to a point, and then it
overshoots*." Our checkers are also all **freely-licensed and self-hosted**, where
theirs was a paid API — which matters for a thesis about not sending data off the
device.

### E3 & E4. The self-revision methods being critiqued

**Madaan et al., 2023 — "Self-Refine: Iterative Refinement with Self-Feedback."** NeurIPS 2023.
[arXiv:2303.17651](https://arxiv.org/abs/2303.17651)

**Shinn et al., 2023 — "Reflexion: Language Agents with Verbal Reinforcement Learning."** NeurIPS 2023.
[arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

The optimistic papers: a model critiques its own output and rewrites it, repeatedly.
Cite them as the approach, then cite Huang et al. (E1) as the correction. Our feedback
levels (none / short pointer / full explanation) are a controlled version of the "how
much feedback" question these papers raise but do not isolate.

---

## Group F — Why any of this matters: running models on ordinary hardware

### F1 & F2. The surveys

**Lu et al., 2024 — "A Survey of Small Language Models."**
[arXiv:2410.20011](https://arxiv.org/abs/2410.20011)

**"Demystifying Small Language Models for Edge Deployment,"** ACL 2025.
[ACL Anthology 2025.acl-long.718](https://aclanthology.org/2025.acl-long.718.pdf)

Use these for the introduction's framing. The second surveys 60+ small models including
the Gemma family we build on, and reports that compressed sub-7B models retain roughly
90–95% of full accuracy — which is the assumption our whole project rests on.

### F3. Concurrent work, and the closest thing to a competitor

**Bondarenko et al., 2026 — "Efficient Reasoning on the Edge."** Qualcomm AI Research.
[arXiv:2603.16867](https://arxiv.org/abs/2603.16867)

Published March 2026, revised June 2026 — while we were running our experiments. They
also use **LoRA adapters** and also use **parallel test-time scaling** (the same family
as our voting) on a **real mobile device**, with Qwen2.5-7B.

**Be upfront about this one.** It is close, and an examiner may know it. The honest
positioning:

| | Them | Us |
|---|---|---|
| Model size | 7B | **2B** |
| Extra accuracy from | more parallel samples | samples **plus an external checker** |
| Hardware | real phone | desktop GPU (we say so, and treat our timings as a floor) |
| Their focus | latency and adapter switching | **when escalating is worth it, and how good the checker must be** |

They optimise the *device side*. We ask a question they do not: what happens when the
device is allowed to ask for help, and how good does the helper need to be?

---

## Group G — Has anyone beaten our 68.5%? (Read this before the defence)

We searched specifically for work that (a) uses a roughly 2B model, (b) on GSM8K,
(c) with a checker, and (d) reports a higher number than ours. **Two things came
back. Neither invalidates the thesis, but both must be addressed out loud, because
an examiner who searches will find them in five minutes.**

### G1. The direct hit — a smaller system that scores higher

**Liu et al., 2023 — "TinyGSM: achieving >80% on GSM8K with small language models."** Microsoft Research.
[arXiv:2312.09241](https://arxiv.org/abs/2312.09241)

A **1.3B generator paired with a 1.3B verifier reaches 81.5% on GSM8K.** Both parts
are smaller than our 2B model, and the score is 13 points above ours. This is the
closest thing to a direct competitor that exists, and we should cite it ourselves
rather than wait to be shown it.

**Why it does not answer our research question:**

| | TinyGSM | Us |
|---|---|---|
| Base model | Phi-1.5 1.3B (**not** Qwen -- a common mix-up) | gemma-4-E2B-it, 2.3B effective |
| Training data | **12.3M synthetic problems** written by GPT-3.5 (~1.8B tokens) | the 7.5K real GSM8K training examples (~1,600x less) |
| Answer format | Python programs, **executed** to get the number | natural-language reasoning, `#### N` extracted |
| Verifier | **trained** on labelled data | **prompted**, off-the-shelf, no training |
| **Generations per question** | **48**, all scored by the verifier, best one kept | **3**, plus one retry on the ~30% that escalate |
| Reported metric | **verify48@1**, not pass@1 | greedy `#### N` exact match |
| Cost of escalation | not measured -- it is not a constraint they have | the entire point of our thesis |

**The 48 is the number to remember.** TinyGSM's headline is not a greedy answer; it
samples the model 48 times per question and has a trained verifier rank all 48. Our
whole system runs on 3 samples. So 81.5% versus 68.5% is not one system beating
another at equal cost -- it is **16x more local generation** on the exact axis this
thesis argues about. Quoting their number beside ours without that fact would
misrepresent both.

For the same reason their number is not comparable to ours at all: it is
`verify48@1` over executed Python, and ours is exact match on a greedily decoded
natural-language chain. Two different metrics on the same test set.

TinyGSM is a **data-scale** result: it shows that enough synthetic data plus code
execution makes a 1.3B model good at GSM8K. Ours is a **cost-constrained routing**
result: given a fixed model, how little help can you ask for and still improve? A
system that calls its verifier on 100% of questions has, by our framing, already
lost — that is the configuration we report as `hint-none, gate none` and use as an
upper bound, not as a proposal.

**One number from it we must quote honestly.** Before adding any verifier, their
fine-tuned 1.3B scored **68.2%** on GSM8K. That is a fine-tuning-only result higher
than our fine-tuning-only result (55.9%) — because of the synthetic data, not the
architecture. Say so plainly.

> **Say in the paper:** "TinyGSM (Liu et al., 2023) reaches 81.5% with a 1.3B
> generator and a trained 1.3B verifier, exceeding our result. Their gain comes from
> 12.3M synthetic training problems and program-aided solutions, with the verifier
> invoked on every question; we hold training data fixed to the 7.5K-example GSM8K
> train split and measure accuracy as a function of *how rarely* the verifier is
> called. The results are complementary rather than competing."

### G2. The uncomfortable one — a stock model that beats our whole pipeline

**Qwen2.5-1.5B-Instruct scores 73.2% on GSM8K** (4-shot, official Qwen2.5 Technical
Report, [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)). Qwen2.5-3B-Instruct is
reported around 86%.

That is a **smaller** model, with **no fine-tuning**, **no voting**, and **no
checker**, scoring above our complete 68.5% system. **Expect this question.** The
honest answers, in order of strength:

1. **Our claims are all paired comparisons on one base model.** Every number we
   report is the same 1,319 questions through the same model, differing only in the
   mechanism under test. Cross-model leaderboard numbers cannot establish that a gate
   or a checker works; only a controlled comparison can. Swapping in a stronger base
   model would raise every row in our table without changing a single conclusion.
2. **The mechanism is the contribution, not the score.** "Escalate the 30% where
   three samples disagree, and stack the checker on the vote" is a claim about
   *routing*, and it transfers to any base model — including Qwen.
3. **The protocols are not comparable.** Qwen's figure is self-reported at 4-shot in
   bfloat16 with lenient answer extraction. Ours is 4-bit quantised, zero-shot, on the
   full test set, requiring an exact `#### N` match. These are not the same
   measurement, and we should not pretend either direction of the gap is meaningful.
4. **Contamination is documented for GSM8K** (see G3) and Qwen2.5 was trained on very
   large maths corpora. Use this last and lightly — it is a real effect but it reads
   as an excuse if it is your first answer.

**This experiment was designed and built but deliberately not run** (decision
2026-09-11; see dossier §5.10). The code is committed and it executes with one
command, but no result exists and **none may be implied**. The thesis handles this by
stating the limitation openly (dossier §8.9): every result comes from one base model,
so the gate is demonstrated on that model rather than shown to be general.

**So answer the question with points 1-3 above, not with a result we do not have.**
If a reviewer presses, the honest position is: *"the mechanism is demonstrated; how
far it generalises across models is future work, and the experiment to settle it is
specified and implemented in our repository."* That is a normal scope boundary and
reads as one.

### G3. The benchmark itself is aging

GSM8K is close to saturated for frontier models (99%+) and has documented
contamination: removing contaminated test examples has been shown to drop some
models by up to 13 points. It remains a reasonable benchmark for *small and
fine-tuned* models, which is our regime, but we should state the limitation
ourselves and note that MATH-500 or a held-out set would strengthen the claim.

> **Say in the limitations:** "GSM8K is partially saturated and known to be affected
> by training-set contamination. We use it because it remains discriminative in the
> sub-3B regime and because it permits comparison with prior small-model work, but a
> contamination-controlled or more recent benchmark would strengthen these results."

### G4. Independent support for our strangest finding

**"JudgeBoard: Benchmarking and Enhancing Small Language Models for Reasoning Evaluation," 2025.**
[arXiv:2511.15958](https://arxiv.org/abs/2511.15958)

Reports that on the MATH dataset, **smaller judge models performed comparably to or
better than larger ones.** This is on a different dataset and a different judging
setup, so it is not the same result — but it means our "9B checker beat the 30B
checker" is consistent with an independent observation rather than an isolated
oddity. Cite it as corroboration; do not claim it as confirmation.

### G5. What nobody appears to have done

We did not find any published work that combines **all four** of these:

1. a fine-tuned sub-3B model as the answerer, **and**
2. an untrained, prompted external checker (not a trained verifier), **and**
3. a gate that sends only a fraction of questions to that checker, **and**
4. a self-consistency control reported beside every configuration.

TinyGSM has (1) and a verifier, but checks everything and trains it. Zhang et al.
(2024) has (1) and (2) but no gate and no cost accounting. The cascade papers
(Chen et al., Gupta et al.) have (3) but do not fine-tune a small model on the task.
**So the combination is unoccupied — but our accuracy number is not the best
published, and we should never imply that it is.** The contribution is the measured
trade-off curve and the stacking result, not the headline percentage.

---

## Where we sit — the one-paragraph version

Copy this into the paper and adapt it:

> Our system combines three established ideas: parameter-efficient fine-tuning of a
> small model (Dettmers et al., 2023), self-consistency decoding (Wang et al., 2023),
> and cost-aware cascading to a stronger model (Chen et al., 2023), with the stronger
> model acting as a verifier (Cobbe et al., 2021; Zheng et al., 2023) rather than a
> generator. Individually, none of these is new. Our contribution is threefold. First,
> we show that on a 2B model these components **interfere**: a cascade run in place of
> self-consistency is not statistically distinguishable from self-consistency alone,
> and only becomes so when layered on top of it — a comparison the cascade literature
> rarely reports. Second, we measure verifier strength as an explicit variable across a
> four-point ladder and find the relationship is **not monotonic**, refining the "small
> models need strong verifiers" result of Zhang et al. (2024). Third, we run the entire
> pipeline, verifier included, on **self-hosted open-weight models with no cloud service
> of any kind**, and report wall-clock latency alongside token cost.

---

## What is genuinely ours, and what is not

**Be honest here — examiners reward it and punish the opposite.**

| Element | New? | The honest position |
|---|---|---|
| Fine-tuning a small model on GSM8K | No | Standard practice. We cite QLoRA and move on. |
| Escalating hard questions to a bigger model | No | FrugalGPT, 2023. We are an instance of it. |
| Using sample disagreement as the difficulty signal | Partly | The signal is known (Aggarwal et al.). Using it as a *cascade gate* rather than a stopping rule is less common. |
| Combining agreement and confidence **with no tuned weight** | Likely new | The lexicographic tie-break is our own. Small, but defensible, and specifically immune to the "you tuned it on test data" attack. |
| **Showing the cascade must be stacked on voting, not substituted for it** | **Our main claim** | We did not find this negative-result-then-fix published in this form. It is the strongest thing in the thesis. |
| The non-monotonic verifier ladder (9B beats 30B) | Likely new | Extends Zhang et al. (2024) from two points to four. |
| Replicating "LLMs cannot self-correct" at 2B scale | Replication | Not novel, but a clean replication at a much smaller scale, and it *justifies our design*. Present it as evidence, not discovery. |

> **The one thing we must never claim:** that 68.5% is a record. It is not.
> TinyGSM reports 81.5% with two 1.3B models (§G1), and a stock Qwen2.5-1.5B-Instruct
> is reported at 73.2% with no cascade at all (§G2). Our contribution is the
> *trade-off curve* — accuracy as a function of how rarely the device asks for help —
> and the stacking result. Frame every claim that way and both comparisons become
> context rather than refutation.

---

## Questions an examiner will ask about the literature

**"Isn't this just FrugalGPT?"**
No — in FrugalGPT the expensive model writes the answer. In ours it only judges; the 2B
model writes every answer that ships. That is the difference between "the cloud answers
your question" and "the cloud checks your device's homework," and it is the whole
privacy argument.

**"Why didn't you train a verifier like Cobbe et al.?"**
Because training a verifier needs labelled data and a training run per verifier, and our
research question was *how strong does the verifier need to be* — which requires
swapping verifiers cheaply. Prompting an off-the-shelf model makes that a server restart
instead of a fine-tuning job. We name this as a limitation and as the obvious next
experiment.

**"Self-consistency is from 2022 — why is beating it interesting?"**
Because most cascade papers do not report it as a baseline, and when we did, our first
system **lost to it**. That is the finding. A system that beats a free alternative by a
statistically indistinguishable margin has not been shown to work, and the field would
be healthier if more papers checked.

**"Your checker is a 30B model — how is that 'edge'?"**
The checker is explicitly *not* on the device. The claim is about what fraction of
questions leave the device (30%) and how many tokens go with them, not about running 30B
locally. This is unchanged by our having self-hosted the checker for reproducibility.

**"Has nobody stacked a cascade on self-consistency before?"**
Verifier-weighted voting exists (arXiv:2410.12608). The difference is that they score
*every* candidate with the verifier; we take the free vote first and pay the verifier
only on the ~30% where the vote was split. Say this plainly — claiming nobody has ever
combined voting and verification would be wrong and easy to disprove.

---

## Ready-to-paste BibTeX

Also saved separately as `docs/references.bib`. Every arXiv identifier below was checked
against arxiv.org while compiling this file. **Verify page numbers and volume details
against the official proceedings before final submission** — arXiv versions and published
versions sometimes differ, and a few entries here cite the arXiv preprint where a
published version now exists.

```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168}, year={2021}
}
@inproceedings{wei2022cot,
  title={Chain-of-Thought Prompting Elicits Reasoning in Large Language Models},
  author={Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Ichter, Brian and Xia, Fei and Chi, Ed and Le, Quoc and Zhou, Denny},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2022}
}
@inproceedings{wang2023selfconsistency,
  title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author={Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc and Chi, Ed and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2023}
}
@inproceedings{hu2022lora,
  title={{LoRA}: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2022}
}
@inproceedings{dettmers2023qlora,
  title={{QLoRA}: Efficient Finetuning of Quantized {LLM}s},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2023}
}
@article{chen2023frugalgpt,
  title={{FrugalGPT}: How to Use Large Language Models While Reducing Cost and Improving Performance},
  author={Chen, Lingjiao and Zaharia, Matei and Zou, James},
  journal={arXiv preprint arXiv:2305.05176}, year={2023}
}
@article{gupta2024cascades,
  title={Language Model Cascades: Token-level Uncertainty and Beyond},
  author={Gupta, Neha and Narasimhan, Harikrishna and Jitkrittum, Wittawat and Rawat, Ankit Singh and Menon, Aditya Krishna and Kumar, Sanjiv},
  journal={arXiv preprint arXiv:2404.10136}, year={2024}
}
@article{zellinger2025abstention,
  title={Cost-Saving {LLM} Cascades with Early Abstention},
  author={Zellinger, Michael J and Liu, Rex and Thomson, Matt},
  journal={arXiv preprint arXiv:2502.09054}, year={2025}
}
@article{ong2024routellm,
  title={{RouteLLM}: Learning to Route {LLM}s with Preference Data},
  author={Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E and Kadous, M Waleed and Stoica, Ion},
  journal={arXiv preprint arXiv:2406.18665}, year={2024}
}
@inproceedings{zheng2023judge,
  title={Judging {LLM}-as-a-Judge with {MT-Bench} and Chatbot Arena},
  author={Zheng, Lianmin and Chiang, Wei-Lin and Sheng, Ying and Zhuang, Siyuan and Wu, Zhanghao and Zhuang, Yonghao and Lin, Zi and Li, Zhuohan and Li, Dacheng and Xing, Eric P and Zhang, Hao and Gonzalez, Joseph E and Stoica, Ion},
  booktitle={NeurIPS Datasets and Benchmarks Track}, year={2023}
}
@article{lightman2023verify,
  title={Let's Verify Step by Step},
  author={Lightman, Hunter and Kosaraju, Vineet and Burda, Yura and Edwards, Harri and Baker, Bowen and Lee, Teddy and Leike, Jan and Schulman, John and Sutskever, Ilya and Cobbe, Karl},
  journal={arXiv preprint arXiv:2305.20050}, year={2023}
}
@inproceedings{huang2024selfcorrect,
  title={Large Language Models Cannot Self-Correct Reasoning Yet},
  author={Huang, Jie and Chen, Xinyun and Mishra, Swaroop and Zheng, Huaixiu Steven and Yu, Adams Wei and Song, Xinying and Zhou, Denny},
  booktitle={International Conference on Learning Representations (ICLR)}, year={2024}
}
@inproceedings{zhang2024strongverifiers,
  title={Small Language Models Need Strong Verifiers to Self-Correct Reasoning},
  author={Zhang, Yunxiang and Khalifa, Muhammad and Logeswaran, Lajanugen and Kim, Jaekyeom and Lee, Moontae and Lee, Honglak and Wang, Lu},
  booktitle={Findings of the Association for Computational Linguistics (ACL)}, year={2024}
}
@inproceedings{madaan2023selfrefine,
  title={Self-Refine: Iterative Refinement with Self-Feedback},
  author={Madaan, Aman and Tandon, Niket and Gupta, Prakhar and Hallinan, Skyler and Gao, Luyu and Wiegreffe, Sarah and Alon, Uri and Dziri, Nouha and Prabhumoye, Shrimai and Yang, Yiming and Gupta, Shashank and Majumder, Bodhisattwa Prasad and Hermann, Katherine and Welleck, Sean and Yazdanbakhsh, Amir and Clark, Peter},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2023}
}
@inproceedings{shinn2023reflexion,
  title={Reflexion: Language Agents with Verbal Reinforcement Learning},
  author={Shinn, Noah and Cassano, Federico and Berman, Edward and Gopinath, Ashwin and Narasimhan, Karthik and Yao, Shunyu},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)}, year={2023}
}
@inproceedings{aggarwal2023adaptive,
  title={Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with {LLM}s},
  author={Aggarwal, Pranjal and Madaan, Aman and Yang, Yiming and Mausam},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)}, year={2023}
}
@article{snell2024testtime,
  title={Scaling {LLM} Test-Time Compute Optimally can be More Effective than Scaling Model Parameters},
  author={Snell, Charlie and Lee, Jaehoon and Xu, Kelvin and Kumar, Aviral},
  journal={arXiv preprint arXiv:2408.03314}, year={2024}
}
@article{kim2023bild,
  title={Speculative Decoding with Big Little Decoder},
  author={Kim, Sehoon and Mangalam, Karttikeya and Moon, Suhong and Malik, Jitendra and Mahoney, Michael W and Gholami, Amir and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2302.07863}, year={2023}
}
@article{lu2024smallsurvey,
  title={A Survey of Small Language Models},
  author={Lu, Zhenyan and Li, Xiang and Cai, Dongqi and Yi, Rongjie and Liu, Fangming and Zhang, Xiwen and Lane, Nicholas D and Xu, Mengwei},
  journal={arXiv preprint arXiv:2410.20011}, year={2024}
}
@article{liu2023tinygsm,
  title={{TinyGSM}: achieving >80\% on {GSM8K} with small language models},
  author={Liu, Bingbin and Bubeck, Sebastien and Eldan, Ronen and Kulkarni, Janardhan and Li, Yuanzhi and Nguyen, Anh and Ward, Rachel and Zhang, Yi},
  journal={arXiv preprint arXiv:2312.09241}, year={2023}
}
@article{qwen2024qwen25,
  title={Qwen2.5 Technical Report},
  author={{Qwen Team}},
  journal={arXiv preprint arXiv:2412.15115}, year={2024}
}
@article{bi2025judgeboard,
  title={{JudgeBoard}: Benchmarking and Enhancing Small Language Models for Reasoning Evaluation},
  author={Bi, Zhenyu and Srivastava, Gaurav and Li, Yang and Lu, Meng and Roy, Swastik and Ziyadi, Morteza and Wang, Xuan},
  journal={arXiv preprint arXiv:2511.15958}, year={2025}
}
@article{unified2026deployment,
  title={Unified Deployment-Aware Evaluation of Open Reasoning Language Models},
  note={Reports Gemma-4-E2B at 0.460 on GSM8K, few-shot CoT, 100 examples, bfloat16},
  journal={arXiv preprint arXiv:2604.07035}, year={2026}
}
@article{wang2025slmmux,
  title={{SLM-MUX}: Orchestrating Small Language Models for Reasoning},
  author={Wang, Chenyu and Wan, Zishen and Kang, Hao and Chen, Emma and Xie, Zhiqiang and Krishna, Tushar and Reddi, Vijay Janapa and Du, Yilun},
  journal={arXiv preprint arXiv:2510.05077}, year={2025}
}
@article{bondarenko2026edge,
  title={Efficient Reasoning on the Edge},
  author={Bondarenko, Yelysei and Hehn, Thomas and Hesselink, Rob and Lepert, Romain and Massoli, Fabio Valerio and Mironov, Evgeny and Mirvakhabova, Leyla and Orekondy, Tribhuvanesh and Stasis, Spyridon and Kuzmin, Andrey and Kuzina, Anna and Nagel, Markus and Nayak, Ankita and Rainone, Corrado and de Rooij, Ork and Whatmough, Paul N and Behboodi, Arash and Bejnordi, Babak Ehteshami},
  journal={arXiv preprint arXiv:2603.16867}, year={2026}
}
```

---

## Reading order, if someone only has two hours

1. **Wang et al. 2023** (self-consistency) — the baseline our story turns on.
2. **Huang et al. 2024** (cannot self-correct) — why the checker must be external.
3. **Zhang et al. 2024** (small models need strong verifiers) — the paper we extend.
4. **Chen et al. 2023** (FrugalGPT) — the family we belong to.

Everything else is context.
