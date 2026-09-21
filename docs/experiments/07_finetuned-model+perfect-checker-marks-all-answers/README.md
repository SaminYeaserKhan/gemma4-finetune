# 07 — The answer key used as the marker (the upper limit)

**In standard terms:** Verdict-only pass with the exact-match oracle; no second model is loaded

**Only one model runs in this experiment: the small AI.** There is no second model. The 'marker' is a few lines of code that compare each answer with the answer key, which is why it is never wrong. Nothing leaves the machine and no tokens are exchanged. It is here for two reasons: it shows the best score any marker could possibly reach, so the real markers in experiments 05 and 06 have something to be measured against, and it confirms our scoring code is correct — an answer key that scored anything but 100% would mean a bug.

## What it is made of

![architecture](architecture.svg)

## What happens to the questions

![question flow](question_flow.svg)

## Results

| | |
|---|---|
| Right answers | **737 of 1,319 — 55.88%** |
| Compared with the trained AI answering once | +0.00 percentage points |
| Answers turned from wrong to right | 0 |
| Answers turned from right to wrong | 0 |
| AI attempts per question | 1.00 |
| Words the small AI writes per question (tokens) | 125.9 |
| Words the small AI reads per question (tokens) | 87.7 |
| Questions sent to the marker | 1,319 (100.0%) |
| Times the marker was asked | 1,319 |
| Tokens exchanged with the marker per question | 0.0  (in 0.0 / out 0.0) |
| Marker replies that could not be read (counted as 'right') | 0 |
| Small AI writing speed (measured) | 8.8 tokens/second |
| Marker time per check (measured) | instant (answer key) |
| **Time per question (estimated)** | 14.3 s |
| **Time for all 1,319 questions (estimated)** | 5 h 14 min |

## How good is this marker?

| | |
|---|---|
| Wrong answers it caught | **100.0%** |
| Right answers it wrongly failed | **0.0%** |
| When it said 'wrong', how often it was correct | 100.0% |

**Where these numbers come from:** `outputs/predictions/09_checker_perfect_oracle_marks_all.jsonl`. Every count, including the ones inside the
diagrams, is computed from that file by `scripts/build_experiment_catalogue.py`.
Time is estimated from measured speeds, because the runs did not record their own
duration — see the main table in [`../README.md`](../README.md).

<details><summary>Diagram source (edit the generator, not this)</summary>

```mermaid
flowchart TB
    subgraph DEV["RUNS ON YOUR OWN MACHINE"]
        direction TB
        SOLVER["<b>The small AI, trained by us</b><br/>on 7,500 school maths problems"]
        ONE["Answers each question <b>once</b>"]
        SOLVER --> ONE
        KEY["<b>The answer key</b><br/>a few lines of code compare each<br/>answer with the correct one<br/><b>no second AI is involved</b>"]
        ONE --> KEY
    end
    KEY --> SCORE["<b>A mark for every answer</b><br/>the answer is never changed:<br/>this measures the marking, not the system"]
    style DEV fill:#e8f4ea,stroke:#2d6a4f,stroke-width:2px
    style KEY fill:#eeeeee,stroke:#888,stroke-dasharray: 4 4
```

```mermaid
flowchart TD
    Q["<b>1,319 maths questions</b>"]
    Q --> A["The small AI answers each question once"]
    A --> M["<b>Each answer is compared with the answer key</b><br/>no second AI: this is a few lines of code"]
    M -->|"matches<br/><b>737</b>"| OK["Counted right"]
    M -->|"does not match<br/><b>582</b>"| NO["Counted wrong"]
    OK --> SAME["<b>No answer is changed.</b><br/>This experiment shows the best score any<br/>marker could reach, and checks that our<br/>scoring code is correct."]
    NO --> SAME
    SAME --> RES
    RES["<b>737 right</b> (55.9%)<br/>582 wrong"]
    style M fill:#eeeeee,stroke:#888,stroke-dasharray: 4 4
    style RES fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
```

</details>
