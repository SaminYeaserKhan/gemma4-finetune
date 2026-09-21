# 06 — A medium AI marks every answer (measuring the marker)

**In standard terms:** Verdict-only pass: Qwen3.5-9B judges every attempt-1 answer, no retry

The medium AI marker grades all 1,319 answers, with no retries, to measure how good it is at spotting wrong answers. Compare it with experiment 05.

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
| Tokens exchanged with the marker per question | 565.1  (in 490.5 / out 74.6) |
| Marker replies that could not be read (counted as 'right') | 50 |
| Small AI writing speed (measured) | 8.8 tokens/second |
| Marker time per check (measured) | 1.00 s |
| **Time per question (estimated)** | 15.3 s |
| **Time for all 1,319 questions (estimated)** | 5 h 35 min |

## How good is this marker?

| | |
|---|---|
| Wrong answers it caught | **88.8%** |
| Right answers it wrongly failed | **18.6%** |
| When it said 'wrong', how often it was correct | 79.1% |

**Where these numbers come from:** `outputs/predictions/08_checker_qwen9b_marks_all.jsonl`. Every count, including the ones inside the
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
    end
    subgraph OFF["NOT ON YOUR MACHINE"]
        direction TB
        CHECK["<b>Marker</b><br/>a medium AI marker (Qwen3.5-9B, 9 billion)<br/><b>never shown the correct answer</b>"]
    end
    ONE -->|every answer| CHECK
    CHECK --> SCORE["<b>A mark for every answer</b><br/>the answer is never changed:<br/>this measures the marking, not the system"]
    style DEV fill:#e8f4ea,stroke:#2d6a4f,stroke-width:2px
    style OFF fill:#fdf0e6,stroke:#b5651d,stroke-width:2px
```

```mermaid
flowchart TD
    Q["<b>1,319 maths questions</b>"]
    Q --> A["The small AI answers each question once"]
    A --> M["<b>The marker grades all 1,319 answers</b>"]
    M -->|"says right<br/><b>665</b>"| OK["Marked right"]
    M -->|"says wrong<br/><b>654</b>"| NO["Marked wrong"]
    NO --> NW["<b>517</b> really were wrong<br/>caught 517 of the 582 wrong answers"]
    NO --> NR["<b>137</b> were actually right<br/>a mistake by the marker"]
    OK --> SAME["<b>No answer is changed</b><br/>the score stays the same"]
    NW --> SAME
    NR --> SAME
    SAME --> RES
    RES["<b>737 right</b> (55.9%)<br/>582 wrong"]
    style M fill:#fdf0e6,stroke:#b5651d
    style NR fill:#fdece9,stroke:#b03a2e
    style RES fill:#e6eefc,stroke:#2b4c8c,stroke-width:2px
```

</details>
