# FYDP 3 — Supervised Cascade Results

Baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl` — 737/1319 correct.

## 1. Headline: accuracy vs. cloud cost

| condition | correct | total | accuracy | cloud_tokens_per_q | escalation_rate | supervisor_calls | local_tokens_per_q | avg_attempts | supervisor_errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local only (1 sample) | 737 | 1319 | 0.5588 | 0.0 |  |  |  |  |  |
| blind retry, take last | 658 | 1319 | 0.4989 | 0.0 |  |  |  |  |  |
| self-consistency@3 (majority) | 779 | 1319 | 0.5906 | 0.0 |  |  |  |  |  |
| pass@3 CEILING | 962 | 1319 | 0.7293 | 0.0 |  |  |  |  |  |
| glm_verdict | 737 | 1319 | 0.5588 | 499.1 | 1.0000 | 1319 | 213.5 | 1.00 | 0 |

## 2. Gate quality (can the device tell when it is wrong?)

| gate | auc | escalation_rate | escalated | errors_caught | recall | precision | retained_accuracy | ceiling_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| length (free) | 0.6757 | 0.0500 | 66 | 53 | 0.0911 | 0.8030 | 0.5778 | 0.5989 |
| length (free) | 0.6757 | 0.1000 | 132 | 99 | 0.1701 | 0.7500 | 0.5931 | 0.6338 |
| length (free) | 0.6757 | 0.2000 | 264 | 185 | 0.3179 | 0.7008 | 0.6237 | 0.6990 |
| length (free) | 0.6757 | 0.3000 | 396 | 249 | 0.4278 | 0.6288 | 0.6392 | 0.7475 |
| length (free) | 0.6757 | 0.4000 | 528 | 316 | 0.5430 | 0.5985 | 0.6637 | 0.7983 |
| length (free) | 0.6757 | 0.5000 | 660 | 380 | 0.6529 | 0.5758 | 0.6935 | 0.8469 |
| length (free) | 0.6757 | 0.6000 | 791 | 420 | 0.7216 | 0.5310 | 0.6932 | 0.8772 |
| length (free) | 0.6757 | 0.8000 | 1055 | 516 | 0.8866 | 0.4891 | 0.7500 | 0.9500 |
| length (free) | 0.6757 | 1.0000 | 1319 | 582 | 1.0000 | 0.4412 | nan | 1.0000 |
| disagreement (k samples) | 0.8404 | 0.0500 | 66 | 53 | 0.0911 | 0.8030 | 0.5778 | 0.5989 |
| disagreement (k samples) | 0.8404 | 0.1000 | 132 | 105 | 0.1804 | 0.7955 | 0.5981 | 0.6384 |
| disagreement (k samples) | 0.8404 | 0.2000 | 264 | 214 | 0.3677 | 0.8106 | 0.6512 | 0.7210 |
| disagreement (k samples) | 0.8404 | 0.3000 | 396 | 315 | 0.5412 | 0.7955 | 0.7107 | 0.7976 |
| disagreement (k samples) | 0.8404 | 0.4000 | 528 | 405 | 0.6959 | 0.7670 | 0.7762 | 0.8658 |
| disagreement (k samples) | 0.8404 | 0.5000 | 660 | 466 | 0.8007 | 0.7061 | 0.8240 | 0.9121 |
| disagreement (k samples) | 0.8404 | 0.6000 | 791 | 520 | 0.8935 | 0.6574 | 0.8826 | 0.9530 |
| disagreement (k samples) | 0.8404 | 0.8000 | 1055 | 566 | 0.9725 | 0.5365 | 0.9394 | 0.9879 |
| disagreement (k samples) | 0.8404 | 1.0000 | 1319 | 582 | 1.0000 | 0.4412 | nan | 1.0000 |

## 3. Verifier quality vs. GSM8K ground truth

| condition | judged | true_reject | false_reject | true_accept | false_accept | precision | recall | false_reject_rate | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glm_verdict | 1319 | 486 | 188 | 549 | 96 | 0.7211 | 0.8351 | 0.2551 | 0.7847 |

## 4. Flip analysis (did retrying break correct answers?)

| condition | wrong_to_right | right_to_wrong | net | stayed_right | stayed_wrong |
| --- | --- | --- | --- | --- | --- |
| glm_verdict | 0 | 0 | 0 | 737 | 582 |

## 5. Significance vs. the un-supervised baseline (exact McNemar)

| condition | baseline_only_right | cascade_only_right | p_value | significant_at_0.05 |
| --- | --- | --- | --- | --- |
| glm_verdict | 0 | 0 | 1.00000 | no |

## 6. Counterfactual cloud output cost per feedback level

**glm_verdict**

| feedback_level | cloud_output_tokens | per_question |
| --- | --- | --- |
| L0 | 10552 | 8.0 |
| L1 | 23522 | 17.8 |
| L2 | 37772 | 28.6 |

