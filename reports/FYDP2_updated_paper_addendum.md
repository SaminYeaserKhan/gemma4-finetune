# FYDP 2 Paper Addendum: Baseline, Few-Shot, and Fine-Tuned GSM8K Evaluation

This section is intended to be inserted into the updated FYDP 2 thesis paper after
the FYDP 1 background and proposed methodology. FYDP 1 described the plan: prepare a
small language model, fine-tune it on a reasoning dataset, compare it against the
original base model, and later add a supervisor model for verification. FYDP 2
completed the first major experimental part of that plan: the model was fine-tuned,
smoke-tested, and evaluated against base-model baselines.

## 1. Purpose of This FYDP 2 Experiment

The goal of this phase was to answer a practical question:

**Does fine-tuning a small model on worked math problems make it more reliable and
more token-efficient than using the base model directly?**

The experiment used the GSM8K dataset. GSM8K is a collection of grade-school math
word problems. Each problem contains a question and a worked solution. At the end of
each official solution, the final answer is written after a special marker:

```text
#### 72
```

In this project, that final number is treated as the gold answer. "Gold answer" means
the official correct answer supplied by the dataset.

The base model was `google/gemma-4-E2B-it`, a 2-billion-parameter instruction-tuned
Gemma 4 model. "Base model" means the original model before this project trained it
further. The fine-tuned model is the same base model with a trained LoRA adapter
added on top. The adapter is stored in:

```text
gemma4-gsm8k-final/
```

The full test set contains 1,319 GSM8K problems. The training set contains 7,473
examples.

## 2. Important Beginner-Friendly Definitions

### 2.1 What is fine-tuning?

Fine-tuning means taking a model that already knows general language patterns and
training it further on a smaller, specific dataset. In this project, the model was
shown many GSM8K examples where a question is followed by step-by-step reasoning and
a final answer. The purpose was not to teach the model English from scratch. The
purpose was to make it better at this particular behavior:

1. read a math word problem;
2. break it into steps;
3. perform the arithmetic;
4. finish with the final answer using the `#### <answer>` format.

### 2.2 What is QLoRA?

The project used QLoRA for fine-tuning. QLoRA is a memory-efficient fine-tuning
method. Instead of updating every parameter in the full model, it keeps the base
model mostly frozen and trains a much smaller set of extra weights called LoRA
adapter weights. This is useful because the local GPU has 16 GB of VRAM, which is
not enough for comfortable full-model training of a modern language model.

In simple terms:

- The original model remains mostly unchanged.
- A small trainable adapter learns the GSM8K reasoning behavior.
- During inference, the base model and adapter are used together.

### 2.3 What does "shot" mean?

The word "shot" can be confusing because it is used in two related ways.

In prompt engineering, a "shot" usually means an example placed inside the prompt.
For example, "8-shot prompting" means the model receives eight worked examples
before it receives the actual test question. The model can then imitate the pattern.

In smoke testing, people sometimes say "one-shot" informally to mean a quick test on
one example. This project includes both ideas:

- The base model was tested directly on a single-question/bare-prompt condition.
- The base model was also tested with 8 worked examples in the prompt.
- The fine-tuned model was smoke-tested on one example and then evaluated on the full
  test set without in-context examples.

For clarity, this section uses "single-question" for the quick direct prompt and
"8-shot" for the baseline that includes eight worked examples.

### 2.4 What is a smoke test?

A smoke test is a small test used to check whether the pipeline works at all. It is
not meant to prove final performance. For example, if a model answers one GSM8K
question correctly, that is useful evidence that loading, prompting, generation, and
answer extraction are working. However, one correct example does not prove that the
model is generally accurate. That is why the final results use the full 1,319-example
test set.

### 2.5 What is exact-match accuracy?

Exact-match accuracy means that the final answer produced by the model must match
the official final answer. The project extracts the final answer from the model's
text using the same rule for every run:

1. If the model writes a `####` marker, the number after the marker is used.
2. If there is no `####` marker, the last number in the model output is used.
3. The answer is normalized before comparison. For example, `$72`, `72.0`, and `72`
   can all be treated as the same numeric answer.

If the extracted answer is numerically different from the official answer, it is
counted as incorrect. If the model produces no extractable final number, it is also
counted as incorrect.

This metric is strict about the final answer. It does not fully judge whether the
reasoning is good or bad. For FYDP 2, the main measured result is final-answer
accuracy. In FYDP 3, the planned supervisor layer is meant to inspect reasoning more
carefully.

### 2.6 What are prompt tokens and generated tokens?

A token is a small piece of text used internally by the model. Tokens can be words,
parts of words, punctuation marks, or special control symbols.

This project measured two token quantities:

- **Prompt tokens:** the input tokens given to the model.
- **Generated tokens:** the output tokens produced by the model.

Total token cost per problem is:

```text
prompt tokens + generated tokens
```

This matters because token count affects runtime and cost. A model that needs a very
large prompt may be expensive even if it gives decent answers.

## 3. Experimental Conditions

Three major conditions are discussed here.

| Condition | Model | Prompt Style | Prediction File | Main Purpose |
|---|---|---|---|---|
| 1-shot/single-question base test | Original base model | Direct question, no worked examples | `outputs/predictions/baseline_bare_prompt_1.5pct.jsonl` | Shows that the base model performs poorly when it is not shown the expected GSM8K format |
| 8-shot base test | Original base model | Eight worked GSM8K examples before each test question | `outputs/predictions/baseline_gsm8k_test.jsonl` | Provides the fair base-model comparison |
| 1-shot/single-question fine-tuned test | Fine-tuned model | Direct question, no worked examples | `outputs/predictions/fine_tuned_gsm8k_test.jsonl` | Tests whether fine-tuning internalized the reasoning format |

The single-question fine-tuned smoke test also produced a correct answer for the
first GSM8K test example and is stored in:

```text
outputs/predictions/smoke_fine_tuned_clean.jsonl
```

The 8-shot base smoke test is stored in:

```text
outputs/predictions/smoke_fewshot_gsm8k_test.jsonl
```

## 4. Summary of Results

| Run | Correct / Total | Accuracy | Average Prompt Tokens | Average Generated Tokens | Average Total Tokens |
|---|---:|---:|---:|---:|---:|
| Base model, 1-shot/single-question bare prompt | 20 / 1319 | 1.5% | Not recorded in older run | 205.4 | Not recorded in older run |
| Base model, 8-shot prompt | 479 / 1319 | 36.3% | 1617.7 | 120.6 | 1738.3 |
| Fine-tuned model, 1-shot/single-question prompt | 737 / 1319 | 55.9% | 87.7 | 125.9 | 213.5 |

The fine-tuned model answered 258 more problems correctly than the 8-shot base
model:

```text
737 - 479 = 258 additional correct answers
```

The accuracy increased by 19.6 percentage points:

```text
55.9% - 36.3% = 19.6 percentage points
```

The fine-tuned model also used far fewer total tokens per problem. Its average total
token count was 213.5 tokens, while the 8-shot base model used 1738.3 tokens. This
means the fine-tuned model used about 12.3% as many tokens as the 8-shot baseline,
or roughly an 8.1x reduction in token usage.

## 5. 1-Shot/Single-Question Base Model Test

### 5.1 What was tested?

The base model was first tested with a direct GSM8K question and no examples in the
prompt. This is the simplest possible setup. The model receives the question and is
expected to answer.

This condition is useful as a diagnostic test, but it is not the fairest final
baseline. The reason is that GSM8K expects a particular answer style: step-by-step
reasoning followed by `#### <final answer>`. If the base model is not shown this
format, it may not know that the final answer should be written in that exact way.

### 5.2 Result

The base model scored:

```text
20 correct out of 1319
Accuracy = 1.5%
```

It also failed to produce an extractable final answer in 566 cases. These were
counted as incorrect because the evaluation code could not identify a valid final
answer from the output.

### 5.3 Example: Incorrect answer caused by repetition

Question:

```text
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
and bakes muffins for her friends every day with four. She sells the remainder
at the farmers' market daily for $2 per fresh duck egg. How much in dollars
does she make every day at the farmers' market?
```

Official solution:

```text
16 - 3 - 4 = 9 eggs sold
9 * 2 = 18 dollars
#### 18
```

The correct final answer is:

```text
18
```

The base model's bare-prompt output repeated the phrase "16 eggs per day" many times
and the extracted final number was:

```text
16
```

This was counted as incorrect because the extracted answer `16` does not match the
gold answer `18`.

### 5.4 Why did the base model perform badly here?

The direct base-model condition performed badly for several reasons.

First, the model was not given examples of the expected GSM8K format. It did not
reliably write a chain of calculations followed by `#### <answer>`.

Second, the model sometimes entered repetition loops. In the Janet example, it
repeated part of the question instead of solving the problem. Repetition is a common
failure mode in language generation when the model is uncertain or when the prompt
does not strongly guide the desired output.

Third, this test measured both reasoning ability and format-following ability at the
same time. A model might know some arithmetic but still fail the evaluation if it
does not produce an extractable final answer. Therefore, this single-question
bare-prompt result was recorded as evidence, but it was not used as the main fair
comparison.

## 6. 8-Shot Base Model Test

### 6.1 What was tested?

The base model was then tested using 8-shot chain-of-thought prompting. This means
that eight worked GSM8K examples from the training split were placed before each
test question. Each example demonstrated the expected behavior:

1. read the question;
2. reason step by step;
3. finish with `#### <final answer>`.

The base model itself was not trained or modified in this condition. It only learned
from the examples inside the prompt for that one request. This is called in-context
learning.

### 6.2 Result

The 8-shot base model scored:

```text
479 correct out of 1319
Accuracy = 36.3%
```

This is much better than the direct base-model condition. The improvement shows that
the base model can imitate the GSM8K answer format when it sees examples.

However, this approach is expensive in tokens:

```text
Average prompt tokens = 1617.7
Average generated tokens = 120.6
Average total tokens = 1738.3
```

The prompt is large because every question carries eight worked examples.

### 6.3 Example: Correct 8-shot answer

For the Janet duck-egg problem, the 8-shot base model answered correctly:

```text
16 - 3 - 4 = 9 eggs
9 * 2 = 18 dollars
#### 18
```

This was counted as correct because the extracted final answer `18` matched the gold
answer `18`.

### 6.4 Example: Incorrect 8-shot answer

Question:

```text
A robe takes 2 bolts of blue fiber and half that much white fiber.
How many bolts in total does it take?
```

Correct reasoning:

```text
Blue fiber = 2 bolts
White fiber = half of 2 = 1 bolt
Total = 2 + 1 = 3 bolts
#### 3
```

The 8-shot base model predicted:

```text
2 + 1/2 = 2.5
#### 2.5
```

This was counted as incorrect because the extracted answer `2.5` does not match the
gold answer `3`.

The error came from misunderstanding the phrase "half that much." In the question,
"that much" refers to the 2 bolts of blue fiber. Half of 2 is 1. The model instead
treated "half" as simply adding 0.5 bolt.

### 6.5 Why did 8-shot prompting perform better than the direct base test?

The 8-shot prompt performed better because it taught the model the pattern at
inference time. The model saw examples of how to structure a GSM8K answer and how to
end with the `####` marker. This reduced formatting failures and helped the model
produce more complete reasoning.

### 6.6 Why was it still worse than the fine-tuned model?

The 8-shot base model still made many reasoning mistakes because the model weights
were not changed. The eight examples helped with formatting and imitation, but they
did not permanently teach the model how to solve GSM8K problems.

The model also had to spend most of its input budget on examples. The average prompt
was about 1618 tokens before the model even started answering the actual question.
This made the baseline more expensive and less efficient.

## 7. 1-Shot/Single-Question Fine-Tuned Test

### 7.1 What was tested?

The fine-tuned model was tested with the same kind of direct question prompt, but now
the model had already been trained on GSM8K-style solutions. It did not need eight
examples in the prompt because the training process had already moved much of that
behavior into the adapter weights.

The quick smoke test on the first GSM8K test example passed. The fine-tuned model
answered the Janet duck-egg problem correctly:

```text
16 - 3 = 13
13 - 4 = 9
9 * 2 = 18
#### 18
```

This showed that the adapter could be loaded and used successfully.

### 7.2 Full test-set result

The fine-tuned model scored:

```text
737 correct out of 1319
Accuracy = 55.9%
```

Token usage was much lower than the 8-shot baseline:

```text
Average prompt tokens = 87.7
Average generated tokens = 125.9
Average total tokens = 213.5
```

The generated output length was similar to the 8-shot baseline, but the prompt was
much shorter. This is the main token-efficiency benefit of fine-tuning. Instead of
repeating eight examples for every question, the model uses the learned adapter
weights.

### 7.3 Example: Correct fine-tuned answer

For the Janet duck-egg problem, the fine-tuned model answered:

```text
She has 16-3 = 13 left after eating some of them for breakfast.
After baking, she is left with 13-4 = 9 to sell.
So she makes 9*2 = 18 a day from selling duck eggs.
#### 18
```

This was counted as correct because the extracted final answer `18` matched the gold
answer `18`.

### 7.4 Example: Incorrect fine-tuned answer

The fine-tuned model still made mistakes. On the robe problem, the correct answer was
`3`, but the fine-tuned model predicted:

```text
It takes twice as much white fiber so the white is 2*2 = 4 bolts.
The blue fibers are 2 bolts and there are also 4 white ones for a total of 6 bolts.
#### 6
```

This was counted as incorrect because the extracted answer `6` does not match the
gold answer `3`.

The mistake is different from the 8-shot base model's mistake. The 8-shot base model
treated "half that much" as 0.5. The fine-tuned model made an even stronger semantic
mistake: it interpreted "half that much" as "twice as much." This shows that
fine-tuning improves average performance but does not guarantee correct language
understanding on every problem.

### 7.5 Why did the fine-tuned model perform better?

The fine-tuned model performed better because it had repeatedly seen GSM8K-style
problems during training. It learned the expected structure of the answer and became
better at producing step-by-step reasoning followed by the final answer marker.

It also no longer needed eight examples in every prompt. This reduced the average
prompt length from 1617.7 tokens to 87.7 tokens.

### 7.6 Why did the fine-tuned model still make mistakes?

Fine-tuning does not make the model perfect. The model is still a small 2-billion
parameter model, and arithmetic word problems require several abilities at once:

1. understanding the wording of the problem;
2. identifying which numbers matter;
3. choosing the correct operation;
4. performing arithmetic correctly;
5. writing the final answer in the expected format.

A failure in any one of these steps can lead to a wrong answer. In the robe example,
the model failed at understanding the relationship described by "half that much."

## 8. What Counts as an Incorrect Answer?

An answer was considered incorrect if the final extracted answer did not match the
official GSM8K final answer.

The following cases were all counted as incorrect:

1. **Wrong final number.** Example: gold answer `18`, model answer `16`.
2. **Wrong arithmetic interpretation.** Example: gold answer `3`, model answer
   `2.5` or `6`.
3. **No extractable final answer.** If the model produced text but no usable final
   number, the answer was counted as incorrect.
4. **Repetition instead of solution.** If the model repeated the question or produced
   unrelated text and the extracted final number was wrong, it was counted as
   incorrect.

For FYDP 2, the evaluation was based on final-answer exact match. This means that a
model could be counted as correct if the final answer was right, even if the written
reasoning was not ideal. This limitation motivates the FYDP 3 supervisor stage,
where another model will judge whether the solution process itself is acceptable.

## 9. Main Findings for the Updated Paper

The FYDP 2 experiment supports three main findings.

First, the direct base model was not reliable on GSM8K. It scored only 1.5% in the
bare-prompt condition and often failed to produce a usable final answer.

Second, 8-shot prompting made the base model much better. Accuracy increased to
36.3% because the examples showed the model how to reason and format the output.
However, this required a very long prompt for every question.

Third, fine-tuning gave the best result among the tested conditions. The fine-tuned
model reached 55.9% accuracy and used far fewer prompt tokens than the 8-shot base
model. This means fine-tuning improved both reliability and token efficiency.

The result is important for the thesis because the project is not only trying to
increase correctness. It is also trying to reduce the cost of getting reliable
answers. The fine-tuned model answered more questions correctly while using about
one-eighth of the total tokens used by the 8-shot base baseline.

## 10. Limitations and Next Step

The FYDP 2 result is a strong baseline-vs-fine-tuned comparison, but it is not the
end of the project.

The main limitation is that GSM8K exact-match accuracy checks only the final answer.
It does not fully verify whether the reasoning is sound. A model may sometimes reach
the right number with weak reasoning, or produce convincing reasoning with a wrong
final answer.

The next phase, FYDP 3, should add the planned supervisor model. The supervisor will
read the question, the model's reasoning, and the final answer. It will return a
binary decision: accept or reject. If the answer is rejected, the fine-tuned model
will retry. This will test whether a supervised verification loop can reduce the
remaining errors after fine-tuning.
