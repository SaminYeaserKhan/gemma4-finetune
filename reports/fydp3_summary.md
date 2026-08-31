# FYDP 3 — Supervised Cascade Results

Baseline: `outputs/predictions/02_answers_finetuned_try1_main.jsonl` — 737/1319 correct.

## 1. Headline: accuracy vs. cloud cost

| condition | correct | total | accuracy | cloud_tokens_per_q | escalation_rate | supervisor_calls | local_tokens_per_q | avg_attempts | supervisor_errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local only (1 sample) | 737 | 1319 | 0.5588 | 0.0 |  |  |  |  |  |
| blind retry, take last | 658 | 1319 | 0.4989 | 0.0 |  |  |  |  |  |
| self-consistency@3 (majority) | 779 | 1319 | 0.5906 | 0.0 |  |  |  |  |  |
| pass@3 CEILING | 962 | 1319 | 0.7293 | 0.0 |  |  |  |  |  |
| hint-none | 756 | 1319 | 0.5732 | 300.9 | 0.3002 | 712 | 436.6 | 1.46 | 0 |
| hint-none + voting | 798 | 1319 | 0.6050 | 300.9 | 0.3002 | 712 | 436.6 | 1.46 | 0 |
| hint-short | 765 | 1319 | 0.5800 | 297.9 | 0.3002 | 712 | 436.8 | 1.46 | 0 |
| hint-short + voting | 807 | 1319 | 0.6118 | 297.9 | 0.3002 | 712 | 436.8 | 1.46 | 0 |
| hint-full | 784 | 1319 | 0.5944 | 296.0 | 0.3002 | 712 | 442.0 | 1.45 | 0 |
| hint-full + voting | 826 | 1319 | 0.6262 | 296.0 | 0.3002 | 712 | 442.0 | 1.45 | 0 |
| more-escalation | 801 | 1319 | 0.6073 | 369.0 | 0.3730 | 889 | 497.1 | 1.57 | 0 |
| more-escalation + voting | 843 | 1319 | 0.6391 | 369.0 | 0.3730 | 889 | 497.1 | 1.57 | 0 |
| smart-gate | 795 | 1319 | 0.6027 | 307.5 | 0.3002 | 728 | 459.6 | 1.48 | 0 |
| smart-gate + voting | 837 | 1319 | 0.6346 | 307.5 | 0.3002 | 728 | 459.6 | 1.48 | 0 |
| BEST-qwen9b | 861 | 1319 | 0.6528 | 345.8 | 0.3002 | 728 | 467.6 | 1.44 | 18 |
| BEST-qwen9b + voting | 903 | 1319 | 0.6846 | 345.8 | 0.3002 | 728 | 467.6 | 1.44 | 18 |

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
| confidence: mean logprob (1 forward pass) | 0.7087 | 0.0500 | 66 | 51 | 0.0876 | 0.7727 | 0.5762 | 0.5974 |
| confidence: mean logprob (1 forward pass) | 0.7087 | 0.1000 | 132 | 98 | 0.1684 | 0.7424 | 0.5922 | 0.6331 |
| confidence: mean logprob (1 forward pass) | 0.7087 | 0.2000 | 264 | 188 | 0.3230 | 0.7121 | 0.6265 | 0.7013 |
| confidence: mean logprob (1 forward pass) | 0.7087 | 0.3000 | 396 | 262 | 0.4502 | 0.6616 | 0.6533 | 0.7574 |
| confidence: mean logprob (1 forward pass) | 0.7087 | 0.4000 | 528 | 327 | 0.5619 | 0.6193 | 0.6776 | 0.8067 |
| confidence: mean logprob (1 forward pass) | 0.7087 | 0.5000 | 660 | 390 | 0.6701 | 0.5909 | 0.7086 | 0.8544 |
| confidence: mean logprob (1 forward pass) | 0.7087 | 0.6000 | 791 | 440 | 0.7560 | 0.5563 | 0.7311 | 0.8923 |
| confidence: mean logprob (1 forward pass) | 0.7087 | 0.8000 | 1055 | 530 | 0.9107 | 0.5024 | 0.8030 | 0.9606 |
| confidence: mean logprob (1 forward pass) | 0.7087 | 1.0000 | 1319 | 582 | 1.0000 | 0.4412 | nan | 1.0000 |
| confidence: min logprob (1 forward pass) | 0.6699 | 0.0500 | 66 | 45 | 0.0773 | 0.6818 | 0.5714 | 0.5929 |
| confidence: min logprob (1 forward pass) | 0.6699 | 0.1000 | 132 | 96 | 0.1649 | 0.7273 | 0.5906 | 0.6315 |
| confidence: min logprob (1 forward pass) | 0.6699 | 0.2000 | 264 | 172 | 0.2955 | 0.6515 | 0.6114 | 0.6892 |
| confidence: min logprob (1 forward pass) | 0.6699 | 0.3000 | 396 | 239 | 0.4107 | 0.6035 | 0.6284 | 0.7400 |
| confidence: min logprob (1 forward pass) | 0.6699 | 0.4000 | 528 | 305 | 0.5241 | 0.5777 | 0.6498 | 0.7900 |
| confidence: min logprob (1 forward pass) | 0.6699 | 0.5000 | 660 | 367 | 0.6306 | 0.5561 | 0.6737 | 0.8370 |
| confidence: min logprob (1 forward pass) | 0.6699 | 0.6000 | 791 | 427 | 0.7337 | 0.5398 | 0.7064 | 0.8825 |
| confidence: min logprob (1 forward pass) | 0.6699 | 0.8000 | 1055 | 524 | 0.9003 | 0.4967 | 0.7803 | 0.9560 |
| confidence: min logprob (1 forward pass) | 0.6699 | 1.0000 | 1319 | 582 | 1.0000 | 0.4412 | nan | 1.0000 |
| confidence: final-answer logprob (1 forward pass) | 0.7149 | 0.0500 | 66 | 53 | 0.0911 | 0.8030 | 0.5778 | 0.5989 |
| confidence: final-answer logprob (1 forward pass) | 0.7149 | 0.1000 | 132 | 99 | 0.1701 | 0.7500 | 0.5931 | 0.6338 |
| confidence: final-answer logprob (1 forward pass) | 0.7149 | 0.2000 | 264 | 194 | 0.3333 | 0.7348 | 0.6322 | 0.7058 |
| confidence: final-answer logprob (1 forward pass) | 0.7149 | 0.3000 | 396 | 267 | 0.4588 | 0.6742 | 0.6587 | 0.7612 |
| confidence: final-answer logprob (1 forward pass) | 0.7149 | 0.4000 | 528 | 338 | 0.5808 | 0.6402 | 0.6915 | 0.8150 |
| confidence: final-answer logprob (1 forward pass) | 0.7149 | 0.5000 | 660 | 393 | 0.6753 | 0.5955 | 0.7132 | 0.8567 |
| confidence: final-answer logprob (1 forward pass) | 0.7149 | 0.6000 | 791 | 444 | 0.7629 | 0.5613 | 0.7386 | 0.8954 |
| confidence: final-answer logprob (1 forward pass) | 0.7149 | 0.8000 | 1055 | 534 | 0.9175 | 0.5062 | 0.8182 | 0.9636 |
| confidence: final-answer logprob (1 forward pass) | 0.7149 | 1.0000 | 1319 | 582 | 1.0000 | 0.4412 | nan | 1.0000 |
| disagreement (k samples) | 0.8404 | 0.0500 | 66 | 53 | 0.0911 | 0.8030 | 0.5778 | 0.5989 |
| disagreement (k samples) | 0.8404 | 0.1000 | 132 | 105 | 0.1804 | 0.7955 | 0.5981 | 0.6384 |
| disagreement (k samples) | 0.8404 | 0.2000 | 264 | 214 | 0.3677 | 0.8106 | 0.6512 | 0.7210 |
| disagreement (k samples) | 0.8404 | 0.3000 | 396 | 315 | 0.5412 | 0.7955 | 0.7107 | 0.7976 |
| disagreement (k samples) | 0.8404 | 0.4000 | 528 | 405 | 0.6959 | 0.7670 | 0.7762 | 0.8658 |
| disagreement (k samples) | 0.8404 | 0.5000 | 660 | 466 | 0.8007 | 0.7061 | 0.8240 | 0.9121 |
| disagreement (k samples) | 0.8404 | 0.6000 | 791 | 520 | 0.8935 | 0.6574 | 0.8826 | 0.9530 |
| disagreement (k samples) | 0.8404 | 0.8000 | 1055 | 566 | 0.9725 | 0.5365 | 0.9394 | 0.9879 |
| disagreement (k samples) | 0.8404 | 1.0000 | 1319 | 582 | 1.0000 | 0.4412 | nan | 1.0000 |
| combined: disagreement, ties broken by confidence | 0.8685 | 0.0500 | 66 | 63 | 0.1082 | 0.9545 | 0.5858 | 0.6065 |
| combined: disagreement, ties broken by confidence | 0.8685 | 0.1000 | 132 | 125 | 0.2148 | 0.9470 | 0.6150 | 0.6535 |
| combined: disagreement, ties broken by confidence | 0.8685 | 0.2000 | 264 | 230 | 0.3952 | 0.8712 | 0.6664 | 0.7331 |
| combined: disagreement, ties broken by confidence | 0.8685 | 0.3000 | 396 | 325 | 0.5584 | 0.8207 | 0.7216 | 0.8052 |
| combined: disagreement, ties broken by confidence | 0.8685 | 0.4000 | 528 | 413 | 0.7096 | 0.7822 | 0.7863 | 0.8719 |
| combined: disagreement, ties broken by confidence | 0.8685 | 0.5000 | 660 | 474 | 0.8144 | 0.7182 | 0.8361 | 0.9181 |
| combined: disagreement, ties broken by confidence | 0.8685 | 0.6000 | 791 | 531 | 0.9124 | 0.6713 | 0.9034 | 0.9613 |
| combined: disagreement, ties broken by confidence | 0.8685 | 0.8000 | 1055 | 572 | 0.9828 | 0.5422 | 0.9621 | 0.9924 |
| combined: disagreement, ties broken by confidence | 0.8685 | 1.0000 | 1319 | 582 | 1.0000 | 0.4412 | nan | 1.0000 |

## 3. Verifier quality vs. GSM8K ground truth

| condition | judged | true_reject | false_reject | true_accept | false_accept | precision | recall | false_reject_rate | accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hint-none | 396 | 283 | 33 | 48 | 32 | 0.8956 | 0.8984 | 0.4074 | 0.8359 |
| hint-short | 396 | 283 | 33 | 48 | 32 | 0.8956 | 0.8984 | 0.4074 | 0.8359 |
| hint-full | 396 | 283 | 33 | 48 | 32 | 0.8956 | 0.8984 | 0.4074 | 0.8359 |
| more-escalation | 492 | 352 | 45 | 54 | 41 | 0.8866 | 0.8957 | 0.4545 | 0.8252 |
| smart-gate | 396 | 297 | 35 | 36 | 28 | 0.8946 | 0.9138 | 0.4930 | 0.8409 |
| BEST-qwen9b | 396 | 301 | 31 | 40 | 24 | 0.9066 | 0.9262 | 0.4366 | 0.8611 |

## 4. Flip analysis (did retrying break correct answers?)

| condition | wrong_to_right | right_to_wrong | net | stayed_right | stayed_wrong |
| --- | --- | --- | --- | --- | --- |
| hint-none | 46 | 27 | 19 | 710 | 536 |
| hint-short | 52 | 24 | 28 | 713 | 530 |
| hint-full | 70 | 23 | 47 | 714 | 512 |
| more-escalation | 94 | 30 | 64 | 707 | 488 |
| smart-gate | 80 | 22 | 58 | 715 | 502 |
| BEST-qwen9b | 136 | 12 | 124 | 725 | 446 |

## 5. Significance vs. the un-supervised baseline (exact McNemar)

| condition | baseline_only_right | cascade_only_right | p_value | significant_at_0.05 |
| --- | --- | --- | --- | --- |
| hint-none | 27 | 46 | 0.03442 | yes |
| hint-short | 24 | 52 | 0.00176 | yes |
| hint-full | 23 | 70 | 0.00000 | yes |
| more-escalation | 30 | 94 | 0.00000 | yes |
| smart-gate | 22 | 80 | 0.00000 | yes |
| BEST-qwen9b | 12 | 136 | 0.00000 | yes |
| hint-none + voting | 39 | 100 | 0.00000 | yes |
| hint-short + voting | 36 | 106 | 0.00000 | yes |
| hint-full + voting | 35 | 124 | 0.00000 | yes |
| more-escalation + voting | 42 | 148 | 0.00000 | yes |
| smart-gate + voting | 34 | 134 | 0.00000 | yes |
| BEST-qwen9b + voting | 24 | 190 | 0.00000 | yes |

## 5b. Significance vs. free self-consistency (the control that matters)

| condition | baseline_only_right | cascade_only_right | p_value | significant_at_0.05 |
| --- | --- | --- | --- | --- |
| hint-none | 81 | 58 | 0.06165 | no |
| hint-short | 78 | 64 | 0.27525 | no |
| hint-full | 77 | 82 | 0.75119 | no |
| more-escalation | 84 | 106 | 0.12741 | no |
| smart-gate | 76 | 92 | 0.24708 | no |
| BEST-qwen9b | 66 | 148 | 0.00000 | yes |
| hint-none + voting | 27 | 46 | 0.03442 | yes |
| hint-short + voting | 24 | 52 | 0.00176 | yes |
| hint-full + voting | 23 | 70 | 0.00000 | yes |
| more-escalation + voting | 30 | 94 | 0.00000 | yes |
| smart-gate + voting | 22 | 80 | 0.00000 | yes |
| BEST-qwen9b + voting | 12 | 136 | 0.00000 | yes |

## 6. Counterfactual cloud output cost per feedback level

**hint-none**

| feedback_level | cloud_output_tokens | per_question |
| --- | --- | --- |
| L0 | 5696 | 4.3 |
| L1 | 18129 | 13.7 |
| L2 | 31699 | 24.0 |

**hint-short**

| feedback_level | cloud_output_tokens | per_question |
| --- | --- | --- |
| L0 | 5696 | 4.3 |
| L1 | 18090 | 13.7 |
| L2 | 31279 | 23.7 |

**hint-full**

| feedback_level | cloud_output_tokens | per_question |
| --- | --- | --- |
| L0 | 5696 | 4.3 |
| L1 | 17949 | 13.6 |
| L2 | 31429 | 23.8 |

**more-escalation**

| feedback_level | cloud_output_tokens | per_question |
| --- | --- | --- |
| L0 | 7112 | 5.4 |
| L1 | 21991 | 16.7 |
| L2 | 38461 | 29.2 |

**smart-gate**

| feedback_level | cloud_output_tokens | per_question |
| --- | --- | --- |
| L0 | 5824 | 4.4 |
| L1 | 18788 | 14.2 |
| L2 | 32796 | 24.9 |

**BEST-qwen9b**

| feedback_level | cloud_output_tokens | per_question |
| --- | --- | --- |
| L0 | 5824 | 4.4 |
| L1 | 38885 | 29.5 |
| L2 | 67263 | 51.0 |

