# Diagrams for the paper — plain language

> **Two versions of every diagram exist. This is the plain-language one.**
>
> | Version | File | Pictures | Use it for |
> |---|---|---|---|
> | **Plain language** (this file) | `docs/DIAGRAMS.md` | `reports/figures/` | Teammates who are not engineers; explaining the system to anyone; a defence audience that includes non-specialists |
> | **Technical** | [`docs/diagrams-technical/DIAGRAMS_TECHNICAL.md`](diagrams-technical/DIAGRAMS_TECHNICAL.md) | `reports/figures/technical/` | The submitted paper, if your supervisor prefers standard terminology in figures |
>
> They show the **same system with the same measured numbers** — only the wording
> differs. The file names match across both folders, so you can swap one set for the
> other without touching your document's figure references.
>
> **If you change a number, change it in both files.** That is the one risk of
> keeping two versions, and nothing checks it for you.

**Who this is for:** whoever is assembling the report. You do not need to understand
the code. Every diagram below is written in ordinary words, with a caption you can
paste and a note on which section it belongs in.

**How to use these.** Each diagram is written in Mermaid, a plain-text way of
describing a picture. GitHub draws them automatically, so just scrolling this page in
your browser shows the finished images. Ready-to-insert files are in `reports/figures/`
as both `.svg` (stays sharp at any size — use this in Word if you can) and `.png`
(works everywhere).

**To change one:** edit the text in the grey box, then run
`.\scripts\render_diagrams.ps1` to redraw the pictures. You do not need design
software, and you should **not** redraw these by hand in Canva or PowerPoint — the
text here is the master copy, and a hand-drawn copy goes out of date silently the
moment a number changes.

**Every number in these diagrams is a real measurement.** If a number changes in
`THESIS_DOSSIER.md`, change it here too, in the same commit.

---

## Plain words, and the technical word for each

The diagrams use everyday language on purpose. When you write the *body* of the paper,
use the proper term from the right-hand column — examiners expect it. Introduce each
one once, in brackets, the first time you use it.

| What the diagrams say | The technical term | What it actually means |
|---|---|---|
| the small AI on your device | **the fine-tuned model** / the solver | The 2-billion-parameter model we trained. Small enough to run on ordinary hardware |
| the bigger AI that marks the work | **the supervisor** / the verifier / the checker | A larger model whose only job is to say whether the working looks right |
| asking for a second opinion | **escalation** | Sending a question to the bigger AI instead of answering it alone |
| about 3 questions in 10 | **a 30% escalation rate** | How often we are willing to ask for help |
| deciding which ones to send | **the gate** | The rule that picks which questions get a second opinion |
| the hard ones | **questions with high disagreement** | Ones where the three attempts all gave different answers |
| how sure the AI sounded | **confidence** (log-probability) | Read from the AI's own internal numbers. Costs nothing extra to obtain |
| taking the most common answer | **self-consistency** / majority voting | Ask three times, keep whichever answer appeared most |
| trained on 7,500 problems | **QLoRA fine-tuning** | A cheap way of teaching a model a new skill without retraining all of it |
| compressed to fit in memory | **4-bit quantisation** | Storing the AI's numbers roughly instead of exactly, so it fits on a small machine |
| doing both, in order | **stacking** | Take the most common answer first; only ask for help when the three disagree |

---

## Figure 1 — What the system is made of
<!-- figure: fig1_system_architecture -->

**Goes in:** Method, as the first figure. **Caption you can paste:**

> Figure 1: System architecture. The fine-tuned 2B model, the majority vote and the
> escalation gate all run on the local device. The supervisor is contacted only for
> questions the gate selects, and never receives the correct answer.

```mermaid
flowchart TB
    subgraph DEV["EVERYTHING HERE RUNS ON YOUR OWN MACHINE (an old laptop, an office PC, a phone)"]
        direction TB
        SOLVER["<b>The small AI</b><br/>We taught it using 7,500 school maths problems.<br/>It is squeezed down to fit on ordinary hardware,<br/>and it writes every answer you ever see."]
        VOTE["<b>Compare its three attempts</b><br/>If two or three attempts landed on the<br/>same answer, that is the answer we keep."]
        GATE["<b>Decide: does this one need help?</b><br/>Only if all three attempts disagree.<br/>Those are the questions it clearly<br/>found hard."]
        SOLVER --> VOTE
        SOLVER --> GATE
    end

    subgraph OFF["THIS PART IS NOT ON YOUR MACHINE (contacted for about 3 questions in 10)"]
        direction TB
        CHECK["<b>A bigger AI, acting as a marker</b><br/>It reads the question and the small AI's working,<br/>then says whether the working looks right.<br/><b>It is never shown the correct answer</b><br/>and it is never asked to answer the question."]
    end

    GATE -->|"only the ones it found hard"| CHECK
    CHECK -->|"a yes or no, plus one sentence<br/>saying where it went wrong<br/>(never the answer itself)"| SOLVER
    VOTE --> OUT["<b>The answer you see</b><br/>always written by the small AI<br/>on your own machine"]

    style DEV fill:#e8f4ea,stroke:#2d6a4f,stroke-width:2px
    style OFF fill:#fdf0e6,stroke:#b5651d,stroke-width:2px
    style OUT fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
```

**The point of this figure:** everything is inside the green box except one thing. The
orange box is the only part that needs an internet connection, and it is a *marker*,
not an answerer — it never writes an answer, it only says whether one looks right.
That is the difference between "the internet answers your question" and "the internet
checks your machine's homework."

---

## Figure 2 — What happens to one question
<!-- figure: fig2_per_question_data_flow -->

**Goes in:** Method, immediately after Figure 1. **This is the most important diagram
in the paper.** Caption you can paste:

> Figure 2: Per-question data flow across the full GSM8K test set (n = 1,319). All
> counts are measured. 923 questions (70%) are resolved entirely on the device; 396
> (30%) are escalated to the supervisor, of which 332 are rejected and retried.

```mermaid
flowchart TD
    Q["<b>1,319 maths questions</b><br/>the standard test set everyone uses"] --> GEN["<b>The small AI answers each one three times</b><br/>the same question, three separate attempts,<br/>allowed to try a different approach each time"]
    GEN --> CMP{"Did the three attempts<br/>land on the same answer?"}

    CMP -->|"all three matched<br/><b>427 questions</b>"| KEEP["<b>Keep that answer. Done.</b><br/>Nothing leaves your machine."]
    CMP -->|"two of the three matched<br/><b>400 questions</b>"| KEEP
    CMP -->|"all three were different<br/><b>492 questions</b>"| RANK["<b>These are the hard ones.</b><br/>There is no answer to trust, so we<br/>put them in order, starting with the<br/>ones the AI sounded least sure about."]

    RANK -->|"the other 96 — we only allow<br/>ourselves to send 3 in 10"| KEEP
    RANK -->|"send the <b>396</b> it<br/>sounded least sure about"| JUDGE["<b>The bigger AI marks the working</b><br/>It sees the question and the working.<br/>It never sees the correct answer."]

    JUDGE -->|"marked it as looking right<br/><b>64 questions</b>"| KEEP
    JUDGE -->|"marked it as wrong<br/><b>332 questions</b>"| RETRY["<b>The small AI tries the question again</b><br/>this time told which step went wrong,<br/>and how the question should be read.<br/><b>It is never told the answer.</b>"]

    RETRY --> FINAL["<b>The answer you see</b>"]
    KEEP --> FINAL

    style KEEP fill:#e8f4ea,stroke:#2d6a4f
    style JUDGE fill:#fdf0e6,stroke:#b5651d
    style FINAL fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
```

**Two things to say about this figure in the text:**

1. **923 of the 1,319 questions never leave the machine** — that is 427 + 400 where the
   attempts agreed, plus 96 hard ones we chose not to send because we had used up our
   allowance.
2. **The three attempts had to be made anyway**, in order to spot the hard ones. So
   comparing them and keeping the most common answer costs nothing extra. This is why
   the design in Figure 3 is free.

---

## Figure 3 — Why you need both halves
<!-- figure: fig3_why_stacking_works -->

**Goes in:** Discussion, as the figure for the main finding. **Caption:**

> Figure 3: The two mechanisms repair different questions. Majority voting corrects
> unlucky sampling; the supervisor corrects consistent misreadings. Because the two
> sets barely overlap, using the supervisor *instead of* the vote discards free
> accuracy — which is why every un-stacked configuration failed to beat
> self-consistency.

```mermaid
flowchart LR
    W["<b>The small AI got<br/>a question wrong.</b><br/>Why?"] --> T{"Which kind of<br/>mistake was it?"}

    T -->|"It does know how to do<br/>this. It was just careless<br/>on this attempt."| A["Its other two attempts<br/>got it right"]
    T -->|"It misunderstood what the<br/>question was asking, and<br/>would do so every time."| B["All three attempts were<br/>wrong in the same way"]

    A --> AV["<b>Keeping the most common answer<br/>already fixes this</b><br/>and it costs nothing"]
    B --> BV["<b>Only an outside opinion fixes this</b><br/>something has to point out<br/>the misunderstanding"]

    AV --> R["<b>So do both, in this order</b><br/>keep the most common answer first,<br/>and only ask for help on the<br/>questions where all three disagreed<br/><br/><b>59.4% → 62.6% correct,</b><br/>with no extra internet use at all"]
    BV --> R

    style AV fill:#e8f4ea,stroke:#2d6a4f
    style BV fill:#fdf0e6,stroke:#b5651d
    style R fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
```

**Say in the text:** used *instead of* comparing the three attempts, the marker gave no
reliable improvement at all. Used *on top of* it, every version improved. The two
halves repair different mistakes, so you need both.

---

## Figure 4 — How we choose which questions to send
<!-- figure: fig4_the_escalation_gate -->

**Goes in:** Method, where the selection rule is described. **Caption:**

> Figure 4: The combined gate. Questions are ordered by how much the three samples
> disagree; ties within a disagreement level are broken by the model's own confidence.
> Confidence reorders questions inside a tied group but can never outrank
> disagreement, so there is no weight to tune.

```mermaid
flowchart TD
    IN["One question, three attempts at it"] --> D["<b>First clue: did the attempts agree?</b><br/>There are only three possible answers<br/>to that, so it cannot separate<br/>questions finely enough on its own."]
    D --> G1["all three matched<br/>probably fine"]
    D --> G2["two of three matched<br/>probably fine"]
    D --> G3["<b>all three differed</b><br/>492 questions land here,<br/>and they all look equally hard"]

    G3 --> C["<b>Second clue: how sure did it sound?</b><br/>We can read this from the AI's own<br/>internal numbers, without asking it<br/>anything again — so it is free."]
    C --> ORD["<b>Put those 492 in order</b><br/>least sure first"]
    ORD --> PICK["<b>Work down the list, sending each one<br/>off, until the allowance runs out</b><br/>(we allow 3 questions in 10)"]

    G1 -.->|"never sent while<br/>harder ones are waiting"| PICK
    G2 -.-> PICK

    style G3 fill:#fdf0e6,stroke:#b5651d
    style PICK fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
```

**Why it is built this way.** The first clue can only ever say one of three things, so
it cannot tell you *which* of the 492 equally-hard questions to send. The second clue
breaks those ties. We tried letting the second clue overrule the first, and it worked
*worse* — so it is deliberately allowed only to decide the order within a tied group.

Both clues were already being worked out for other reasons, so combining them cost
nothing. Doing so raised the quality of the choice from **0.840 to 0.869** (1.0 would be
a perfect chooser, 0.5 would be random guessing), and caught **125 wrong answers
instead of 105** when only allowed to send 1 question in 10.

---

## Figure 5 — How the experiments were run
<!-- figure: fig5_experiment_pipeline -->

**Goes in:** an appendix, or the paragraph about repeatability. **Caption:**

> Figure 5: Experimental pipeline. Each box is a script in the repository; each arrow
> names the file it writes. The first attempt at every question is generated once and
> reused by every later experiment, so all comparisons are exactly paired.

```mermaid
flowchart TD
    HF[("The maths questions,<br/>downloaded from the internet")] --> PREP["<b>prepare_data.py</b><br/>puts them all in one tidy format"]
    PREP --> TRAIN["<b>train.py</b><br/>teaches the small AI<br/>using the practice questions<br/><i>takes 68 minutes</i>"]
    TRAIN --> GEN["<b>generate.py</b><br/>asks it all 1,319 test questions,<br/>three times each"]

    GEN --> CONF["<b>score_confidence.py</b><br/>works out how sure it sounded<br/>about each answer, without<br/>asking it anything again"]
    GEN -->|"the three sets of answers"| SUP
    CONF -->|"how sure it sounded"| SUP["<b>supervise.py</b><br/>runs the whole system:<br/>compare, decide, ask for help,<br/>try again"]

    SERVE["<b>serve_verifier.ps1</b><br/>starts the bigger AI up<br/>so it can be asked questions"] -.->|"on the same machine,<br/>in a separate window"| SUP
    SUP --> AN["<b>analyze_supervision.py</b><br/>counts everything up and<br/>builds the tables and charts"]
    AN --> REP[("The tables and figures<br/>that go in the paper")]

    style TRAIN fill:#e8f4ea,stroke:#2d6a4f
    style SUP fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
    style SERVE fill:#fdf0e6,stroke:#b5651d
```

**Worth one sentence in the text:** the small AI's first attempt at each question was
generated once and then reused by every experiment. That means every version of the
system was tested on *exactly* the same starting answers, so any difference between
them is caused by the thing we changed, not by the AI happening to have a better day.

---

## Figure 6 — How good does the marker need to be?
<!-- figure: fig6_checker_strength_ladder -->

**Goes in:** Results. **Caption:**

> Figure 6: Supervisor strength as a measured variable. Moving from 2B self-checking to
> a 9B supervisor sharply increases errors caught; moving from 9B to 30B makes false
> rejections worse. Supervisor capability is not monotonically beneficial.

```mermaid
flowchart LR
    L1["<b>The small AI marking<br/>its own work</b><br/>Catches only 22 of every<br/>100 wrong answers.<br/>It approves almost anything<br/>it wrote itself."]
    L2["<b>A medium AI as marker</b><br/>(9 billion)<br/>Catches 89 of every 100<br/>wrong answers.<br/>But wrongly fails 19 of every<br/>100 answers that were right.<br/><br/><b>BEST OVERALL</b><br/>and nearly twice as fast"]
    L3["<b>A large AI as marker</b><br/>(30 billion)<br/>Catches a few more errors,<br/>but wrongly fails <b>26</b> of every<br/>100 answers that were right.<br/><br/><b>WORSE, despite being bigger</b>"]
    L4["<b>A perfect marker</b><br/>one that is simply shown<br/>the answer key.<br/>Not a real option — it just<br/>shows the best score<br/>anyone could ever reach."]

    L1 -->|"a huge improvement"| L2
    L2 -->|"<b>goes backwards</b>"| L3
    L3 -.->|"for reference only"| L4

    style L2 fill:#e8f4ea,stroke:#2d6a4f,stroke-width:2px
    style L3 fill:#fdece9,stroke:#b03a2e
    style L4 fill:#eeeeee,stroke:#888,stroke-dasharray: 4 4
```

**The sentence this figure exists to support:** *there is a size of marker that fits the
job, and the 30-billion one overshoots it.* A marker that fails right answers too
readily destroys good work faster than it repairs bad work — so "use the biggest model
you can" turns out to be wrong advice here.

---

## What should NOT be a diagram

Results that are just *numbers* belong in a table or a chart, not in boxes and arrows:

- **Accuracy against internet cost** — already drawn, in `reports/figures/pareto.png`.
  Use it as the main results figure.
- **How many answers were turned from wrong to right, and right to wrong** — a small
  four-box table reads far better than any diagram.
- **The full list of results** — a table. Do not try to draw it.

---

## Redrawing the pictures

The `.svg` and `.png` files are generated from the Markdown. After editing any diagram
above, run:

```powershell
.\scripts\render_diagrams.ps1              # redraws BOTH wordings
.\scripts\render_diagrams.ps1 -Plain       # only this file's pictures
.\scripts\render_diagrams.ps1 -Technical   # only the technical ones
.\scripts\render_diagrams.ps1 -Only fig2   # one figure, both wordings
```

Running it with no options redraws both sets, which is the safe default — it keeps
the two versions from silently falling out of step.

Both wordings produce **the same six file names**, in different folders:
`reports/figures/` for these, `reports/figures/technical/` for the formal ones.

| File | What it shows |
|---|---|
| `fig1_system_architecture` | What the system is made of |
| `fig2_per_question_data_flow` | What happens to one question |
| `fig3_why_stacking_works` | Why you need both halves |
| `fig4_the_escalation_gate` | How we choose which questions to send |
| `fig5_experiment_pipeline` | How the experiments were run |
| `fig6_checker_strength_ladder` | How good the marker needs to be |

Adding a new `## Figure N — Title` section with a grey Mermaid box is enough; the
script will find it on its own, with no other changes needed.
