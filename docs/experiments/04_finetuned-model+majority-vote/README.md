# 04 — Trained AI answers three times, most common answer kept

**In standard terms:** Fine-tuned model with self-consistency@3 (majority vote)

The trained AI answers each question three times and the most common answer is kept. No other AI is involved and nothing leaves the machine. This is the free alternative every system with a marker has to beat.

## What it is made of

![architecture](architecture.svg)

## What happens to the questions

![question flow](question_flow.svg)

## Results

| | |
|---|---|
| Right answers | **779 of 1,319 — 59.06%** |
| Compared with the trained AI answering once | +3.18 percentage points |
| Answers turned from wrong to right | 54 |
| Answers turned from right to wrong | 12 |
| AI attempts per question | 3.00 |
| Words the small AI writes per question (tokens) | 389.8 |
| Words the small AI reads per question (tokens) | 263.0 |
| Questions sent to the marker | 0 (0.0%) |
| Times the marker was asked | 0 |
| Tokens exchanged with the marker per question | 0.0  (in 0.0 / out 0.0) |
| Marker replies that could not be read (counted as 'right') | 0 |
| Small AI writing speed (measured) | 8.8 tokens/second |
| Marker time per check (measured) | — |
| **Time per question (estimated)** | 44.2 s |
| **Time for all 1,319 questions (estimated)** | 16 h 11 min |

**Where these numbers come from:** computed from `outputs/predictions/02_`, `03_` and `04_`. Every count, including the ones inside the
diagrams, is computed from that file by `scripts/build_experiment_catalogue.py`.
Time is estimated from measured speeds, because the runs did not record their own
duration — see the main table in [`../README.md`](../README.md).

<details><summary>Diagram source (edit the generator, not this)</summary>

```mermaid
flowchart TB
    subgraph DEV["RUNS ON YOUR OWN MACHINE"]
        direction TB
        SOLVER["<b>The small AI, trained by us</b><br/>on 7,500 school maths problems"]
        VOTE["<b>Compare three attempts</b><br/>keep the most common answer"]
        SOLVER -->|three attempts| VOTE
    end
    VOTE --> OUT["<b>The answer you see</b>"]
    style DEV fill:#e8f4ea,stroke:#2d6a4f,stroke-width:2px
```

```mermaid
flowchart TD
    Q["<b>1,319 maths questions</b>"]
    Q --> A["The small AI answers each question <b>three times</b>"]
    A --> D{"Did the attempts agree?"}
    D -->|"all three matched<br/><b>427</b>"| K["Keep the answer they agreed on"]
    D -->|"two of three matched<br/><b>400</b>"| K
    D -->|"all three differed<br/><b>492</b>"| F["No most-common answer,<br/>so keep the first attempt"]
    K --> RES
    F --> RES
    RES["<b>779 right</b> (59.1%)<br/>540 wrong"]
    style K fill:#e8f4ea,stroke:#2d6a4f
    style RES fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
```

</details>
