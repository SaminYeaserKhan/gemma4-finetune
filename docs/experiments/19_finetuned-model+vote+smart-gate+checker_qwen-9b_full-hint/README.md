# 19 — Trained AI + majority vote + smart selection of hard questions (30%) + medium AI marker, full hint — BEST RESULT

**In standard terms:** Stacked on self-consistency@3; combined gate (disagreement, confidence tie-break) at 30% escalation; Qwen3.5-9B verifier; L2 feedback

The trained AI answers every question three times. Where at least two attempts agree, that answer is kept straight away. About 30% of questions are sent to medium AI marker — the ones where all three attempts disagreed, taking first the ones the AI sounded least sure about. If the marker says the working is wrong, it replies with yes or no, which step went wrong, and how the question should be read, and the small AI tries again. The marker is never shown the correct answer.

## What it is made of

![architecture](architecture.svg)

## What happens to the questions

![question flow](question_flow.svg)

## Results

| | |
|---|---|
| Right answers | **903 of 1,319 — 68.46%** |
| Compared with the trained AI answering once | +12.59 percentage points |
| Answers turned from wrong to right | 190 |
| Answers turned from right to wrong | 24 |
| AI attempts per question | 3.44 |
| Words the small AI writes per question (tokens) | 464.1 |
| Words the small AI reads per question (tokens) | 442.8 |
| Questions sent to the marker | 396 (30.0%) |
| Times the marker was asked | 728 |
| Tokens exchanged with the marker per question | 345.8  (in 285.3 / out 60.5) |
| Marker replies that could not be read (counted as 'right') | 18 |
| Small AI writing speed (measured) | 8.8 tokens/second |
| Marker time per check (measured) | 1.00 s |
| **Time per question (estimated)** | 53.1 s |
| **Time for all 1,319 questions (estimated)** | 19 h 28 min |

> The smart selection also reads the first answer once more to measure how sure the AI sounded. That extra read was not timed and is not in the estimate.

**Where these numbers come from:** `outputs/predictions/15_pipeline_qwen9b_BEST_RESULT.jsonl`. Every count, including the ones inside the
diagrams, is computed from that file by `scripts/build_experiment_catalogue.py`.
Time is estimated from measured speeds, because the runs did not record their own
duration — see the main table in [`../README.md`](../README.md).

<details><summary>Diagram source (edit the generator, not this)</summary>

```mermaid
flowchart TB
    subgraph DEV["RUNS ON YOUR OWN MACHINE"]
        direction TB
        SOLVER["<b>The small AI, trained by us</b><br/>on 7,500 school maths problems"]
        VOTE["<b>Compare three attempts</b><br/>if two or three agree,<br/>keep that answer"]
        SOLVER -->|three attempts| VOTE
        GATE["<b>Pick the hard questions</b><br/>all three attempts disagreed;<br/>least sure-sounding first<br/>(about 30% of questions)"]
        SOLVER -->|three attempts| GATE
    end
    subgraph OFF["NOT ON YOUR MACHINE"]
        direction TB
        CHECK["<b>Marker</b><br/>a medium AI marker (Qwen3.5-9B, 9 billion)<br/><b>never shown the correct answer</b>"]
    end
    GATE -->|hard questions only| CHECK
    CHECK -->|"if wrong: it says which step went wrong<br/>and how to read the question<br/>(never the answer)"| SOLVER
    SOLVER --> OUT["<b>The answer you see</b><br/>always written by the small AI"]
    VOTE -->|agreed answers| OUT
    style DEV fill:#e8f4ea,stroke:#2d6a4f,stroke-width:2px
    style OFF fill:#fdf0e6,stroke:#b5651d,stroke-width:2px
```

```mermaid
flowchart TD
    Q["<b>1,319 maths questions</b>"]
    Q --> A["The small AI answers each question <b>three times</b>"]
    A --> D{"Did the attempts agree?"}
    D -->|"all or two matched<br/><b>827</b>"| K["<b>Keep the agreed answer</b><br/>nothing leaves the machine"]
    D -->|"all three differed<br/><b>492</b>"| H["<b>The hard ones</b><br/>order them, least sure first"]
    H -->|"<b>96</b> — over the<br/>30% allowance"| KF["Keep the first attempt"]
    KF --> RES
    H -->|"picked<br/><b>396</b>"| M["<b>The marker checks the working</b><br/>never shown the correct answer"]
    M -->|"says right<br/><b>64</b>"| KEEP2["Keep that answer"]
    M -->|"says wrong<br/><b>332</b>"| R1["<b>The small AI tries again</b><br/>told which step went wrong<br/>and how to read the question"]
    R1 -->|"marker now says right<br/><b>85</b>"| KEEP2
    R1 -->|"still wrong<br/><b>247</b>"| R2["One last try,<br/>kept without checking"]
    K --> RES
    KEEP2 --> RES
    R2 --> RES
    RES["<b>903 right</b> (68.5%)<br/>416 wrong"]
    style K fill:#e8f4ea,stroke:#2d6a4f
    style M fill:#fdf0e6,stroke:#b5651d
    style RES fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
```

</details>
