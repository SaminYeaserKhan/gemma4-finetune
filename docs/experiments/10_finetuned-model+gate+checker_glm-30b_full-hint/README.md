# 10 — Trained AI + selection of hard questions (30%) + large AI marker, full hint

**In standard terms:** disagreement gate at 30% escalation; GLM-4.7-Flash verifier; L2 feedback

The trained AI answers every question three times. On questions that are not sent for help, its first answer is kept. About 30% of questions are sent to large AI marker — the ones where the three attempts disagreed most. If the marker says the working is wrong, it replies with yes or no, which step went wrong, and how the question should be read, and the small AI tries again. The marker is never shown the correct answer.

## What it is made of

![architecture](architecture.svg)

## What happens to the questions

![question flow](question_flow.svg)

## Results

| | |
|---|---|
| Right answers | **784 of 1,319 — 59.44%** |
| Compared with the trained AI answering once | +3.56 percentage points |
| Answers turned from wrong to right | 70 |
| Answers turned from right to wrong | 23 |
| AI attempts per question | 3.45 |
| Words the small AI writes per question (tokens) | 466.2 |
| Words the small AI reads per question (tokens) | 415.1 |
| Questions sent to the marker | 396 (30.0%) |
| Times the marker was asked | 712 |
| Tokens exchanged with the marker per question | 296.0  (in 264.1 / out 31.8) |
| Marker replies that could not be read (counted as 'right') | 0 |
| Small AI writing speed (measured) | 8.8 tokens/second |
| Marker time per check (measured) | 2.07 s |
| **Time per question (estimated)** | 53.9 s |
| **Time for all 1,319 questions (estimated)** | 19 h 46 min |

**Where these numbers come from:** `outputs/predictions/12_pipeline_glm30b_hint_full.jsonl`. Every count, including the ones inside the
diagrams, is computed from that file by `scripts/build_experiment_catalogue.py`.
Time is estimated from measured speeds, because the runs did not record their own
duration — see the main table in [`../README.md`](../README.md).

<details><summary>Diagram source (edit the generator, not this)</summary>

```mermaid
flowchart TB
    subgraph DEV["RUNS ON YOUR OWN MACHINE"]
        direction TB
        SOLVER["<b>The small AI, trained by us</b><br/>on 7,500 school maths problems"]
        GATE["<b>Pick the hard questions</b><br/>the attempts disagreed most<br/>(about 30% of questions)"]
        SOLVER -->|three attempts| GATE
    end
    subgraph OFF["NOT ON YOUR MACHINE"]
        direction TB
        CHECK["<b>Marker</b><br/>a large AI marker (GLM-4.7-Flash, 30 billion)<br/><b>never shown the correct answer</b>"]
    end
    GATE -->|hard questions only| CHECK
    CHECK -->|"if wrong: it says which step went wrong<br/>and how to read the question<br/>(never the answer)"| SOLVER
    SOLVER --> OUT["<b>The answer you see</b><br/>always written by the small AI"]
    style DEV fill:#e8f4ea,stroke:#2d6a4f,stroke-width:2px
    style OFF fill:#fdf0e6,stroke:#b5651d,stroke-width:2px
```

```mermaid
flowchart TD
    Q["<b>1,319 maths questions</b>"]
    Q --> A["The small AI answers each question <b>three times</b>"]
    A --> H["<b>Pick the hard questions</b><br/>where the attempts disagreed most"]
    H -->|"not picked<br/><b>923</b>"| K["<b>Keep the first attempt</b><br/>nothing leaves the machine"]
    H -->|"picked<br/><b>396</b>"| M["<b>The marker checks the working</b><br/>never shown the correct answer"]
    M -->|"says right<br/><b>80</b>"| KEEP2["Keep that answer"]
    M -->|"says wrong<br/><b>316</b>"| R1["<b>The small AI tries again</b><br/>told which step went wrong<br/>and how to read the question"]
    R1 -->|"marker now says right<br/><b>32</b>"| KEEP2
    R1 -->|"still wrong<br/><b>284</b>"| R2["One last try,<br/>kept without checking"]
    K --> RES
    KEEP2 --> RES
    R2 --> RES
    RES["<b>784 right</b> (59.4%)<br/>535 wrong"]
    style K fill:#e8f4ea,stroke:#2d6a4f
    style M fill:#fdf0e6,stroke:#b5651d
    style RES fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
```

</details>
