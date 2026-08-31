# FYDP 3 — Supervised Cascade Results

Baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl` — 737/1319 correct.

## 1. Headline: accuracy vs. cloud cost

| condition | correct | total | accuracy | cloud_tokens_per_q |
| --- | --- | --- | --- | --- |
| local only (1 sample) | 737 | 1319 | 0.5588 | 0.0 |
| blind retry, take last | 658 | 1319 | 0.4989 | 0.0 |
| self-consistency@3 (majority) | 779 | 1319 | 0.5906 | 0.0 |
| pass@3 CEILING | 962 | 1319 | 0.7293 | 0.0 |

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

