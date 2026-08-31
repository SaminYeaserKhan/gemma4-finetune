## 2026-05-15T06:42:35Z - Trainer epoch-end evaluation caused CUDA OOM

- Type: `issue`
- step: `936`
- latest_valid_checkpoint: `checkpoint-900`
- issue: `Epoch-end token-loss eval attempted extra 4.38 GiB allocation on 16 GB GPU.`
- resolution: `Resumed from checkpoint-900 with --eval-strategy no; external generate/evaluate pipeline used instead.`

## 2026-05-15T06:42:35Z - Fine-tuned adapter smoke generation passed

- Type: `smoke_test`
- prediction_file: `outputs/predictions/smoke_fine_tuned_clean.jsonl`
- examples: `1`
- result: `Predicted GSM8K test example 0 final answer 18 correctly.`

## 2026-05-15T06:42:37Z - Finished Gemma GSM8K QLoRA training

- Type: `milestone`
- final_adapter: `gemma4-gsm8k-final`
- final_step: `1404`
- final_epoch: `3`
- train_runtime_seconds: `4080`
- train_loss: `0.09843`
- mean_token_accuracy: `0.924`
- note: `Training completed after disabling in-training eval; post-training exact-answer eval will be used.`
- environment: `{"python": "3.11.9", "platform": "Windows-10-10.0.26200-SP0", "torch": "2.6.0+cu124", "cuda_available": true, "cuda": "12.4", "gpu": "NVIDIA GeForce RTX 4080 SUPER", "vram_gb": 17.17}`

## 2026-05-15T11:16:43Z - Generated GSM8K predictions: smoke_tokens

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --adapter-dir gemma4-gsm8k-final --run-name smoke_tokens --limit 5`
- run_name: `smoke_tokens`
- task: `gsm8k`
- split: `test`
- examples: `5`
- adapter_dir: `gemma4-gsm8k-final`
- output: `outputs\predictions\smoke_tokens_gsm8k_test.jsonl`
- temperature: `0.0`
- max_new_tokens: `512`

## 2026-05-15T11:27:00Z - Generated GSM8K predictions: smoke_tokens

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --adapter-dir gemma4-gsm8k-final --run-name smoke_tokens --limit 5`
- run_name: `smoke_tokens`
- task: `gsm8k`
- split: `test`
- examples: `5`
- adapter_dir: `gemma4-gsm8k-final`
- output: `outputs\predictions\smoke_tokens_gsm8k_test.jsonl`
- temperature: `0.0`
- max_new_tokens: `512`

## 2026-05-15T22:36:10Z - Generated GSM8K predictions: baseline

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --run-name baseline`
- run_name: `baseline`
- task: `gsm8k`
- split: `test`
- examples: `1319`
- adapter_dir: ``
- output: `outputs\predictions\baseline_gsm8k_test.jsonl`
- temperature: `0.0`
- max_new_tokens: `512`

## 2026-05-15T23:27:35Z - Generated GSM8K predictions: smoke_fewshot

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --run-name smoke_fewshot --few-shot 8 --limit 5`
- run_name: `smoke_fewshot`
- task: `gsm8k`
- split: `test`
- examples: `5`
- adapter_dir: ``
- output: `outputs\predictions\smoke_fewshot_gsm8k_test.jsonl`
- temperature: `0.0`
- max_new_tokens: `512`
- few_shot: `8`

## 2026-05-16T02:46:57Z - Generated GSM8K predictions: baseline

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --run-name baseline --few-shot 8`
- run_name: `baseline`
- task: `gsm8k`
- split: `test`
- examples: `1319`
- adapter_dir: ``
- output: `outputs\predictions\baseline_gsm8k_test.jsonl`
- temperature: `0.0`
- max_new_tokens: `512`
- few_shot: `8`

## 2026-05-16T08:11:27Z - Generated GSM8K predictions: fine_tuned

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --adapter-dir gemma4-gsm8k-final --run-name fine_tuned`
- run_name: `fine_tuned`
- task: `gsm8k`
- split: `test`
- examples: `1319`
- adapter_dir: `gemma4-gsm8k-final`
- output: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- temperature: `0.0`
- max_new_tokens: `512`
- few_shot: `0`

## 2026-05-16T08:14:53Z - Evaluated gsm8k predictions

- Type: `evaluation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe evaluate.py --predictions baseline=outputs/predictions/baseline_gsm8k_test.jsonl --predictions fine_tuned=outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- task: `gsm8k`
- summaries: `[{"name": "baseline", "file": "outputs\\predictions\\baseline_gsm8k_test.jsonl", "total": 1319, "correct": 479, "accuracy": "0.3632", "avg_prompt_tokens": "1617.7", "avg_gen_tokens": "120.6", "avg_total_tokens": "1738.3", "supervisor_accept_rate": "0.0000", "avg_attempts": ""}, {"name": "fine_tuned", "file": "outputs\\predictions\\fine_tuned_gsm8k_test.jsonl", "total": 1319, "correct": 737, "accuracy": "0.5588", "avg_prompt_tokens": "87.7", "avg_gen_tokens": "125.9", "avg_total_tokens": "213.5", "supervisor_accept_rate": "0.0000", "avg_attempts": ""}]`
- csv_output: `reports/gsm8k_summary.csv`
- md_output: `reports/gsm8k_summary.md`

## 2026-05-16T08:18:23Z - Evaluated gsm8k predictions

- Type: `evaluation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe evaluate.py --predictions baseline=outputs/predictions/baseline_gsm8k_test.jsonl --predictions fine_tuned=outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- task: `gsm8k`
- summaries: `[{"name": "baseline", "file": "outputs\\predictions\\baseline_gsm8k_test.jsonl", "total": 1319, "correct": 479, "accuracy": "0.3632", "avg_prompt_tokens": "1617.7", "avg_gen_tokens": "120.6", "avg_total_tokens": "1738.3", "supervisor_accept_rate": "", "avg_attempts": ""}, {"name": "fine_tuned", "file": "outputs\\predictions\\fine_tuned_gsm8k_test.jsonl", "total": 1319, "correct": 737, "accuracy": "0.5588", "avg_prompt_tokens": "87.7", "avg_gen_tokens": "125.9", "avg_total_tokens": "213.5", "supervisor_accept_rate": "", "avg_attempts": ""}]`
- csv_output: `reports/gsm8k_summary.csv`
- md_output: `reports/gsm8k_summary.md`

## 2026-05-16T08:19:03Z - FYDP 2 complete: baseline vs fine-tuned GSM8K evaluation

- Type: `milestone`
- baseline_accuracy: `0.3632`
- fine_tuned_accuracy: `0.5588`
- accuracy_gain_pp: `19.6`
- baseline_prompt: `8-shot CoT`
- fine_tuned_prompt: `zero-shot`
- baseline_avg_total_tokens: `1738.3`
- fine_tuned_avg_total_tokens: `213.5`
- test_examples: `1319`
- report: `reports/gsm8k_summary.md`
- environment: `{"python": "3.11.9", "platform": "Windows-10-10.0.26200-SP0", "torch": "2.6.0+cu124", "cuda_available": true, "cuda": "12.4", "gpu": "NVIDIA GeForce RTX 4080 SUPER", "vram_gb": 17.17}`

## 2026-08-05T02:45:53Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --md-output reports\fydp3_summary.md --csv-output reports\fydp3_summary.csv`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- cascades: `[]`
- md_output: `reports\fydp3_summary.md`
- csv_output: `reports\fydp3_summary.csv`

## 2026-08-05T02:46:58Z - Cascade run: exact / none @ 100% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe supervise.py --provider exact --gate none --verdict-only --output outputs\predictions\smoke_verdict_exact.jsonl`
- provider: `exact`
- supervisor_model: `exact`
- gate: `none`
- escalation_rate: `1.0`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `1319`
- examples: `1319`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- output: `outputs\predictions\smoke_verdict_exact.jsonl`

## 2026-08-05T02:47:28Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --cascade oracle_verdict=outputs\predictions\smoke_verdict_exact.jsonl --md-output reports\smoke_fydp3.md --csv-output reports\smoke_fydp3.csv`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- cascades: `["oracle_verdict"]`
- md_output: `reports\smoke_fydp3.md`
- csv_output: `reports\smoke_fydp3.csv`

## 2026-08-05T02:49:12Z - Cascade run: exact / none @ 100% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe supervise.py --provider exact --gate none --limit 5 --feedback-level 1 --output outputs\predictions\smoke_cascade_exact.jsonl`
- provider: `exact`
- supervisor_model: `exact`
- gate: `none`
- escalation_rate: `1.0`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `5`
- examples: `5`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- output: `outputs\predictions\smoke_cascade_exact.jsonl`

## 2026-08-05T02:50:46Z - Cascade run: exact / none @ 100% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe supervise.py --provider exact --gate none --limit 5 --feedback-level 1 --resume --output outputs\predictions\smoke_cascade_exact.jsonl`
- provider: `exact`
- supervisor_model: `exact`
- gate: `none`
- escalation_rate: `1.0`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `5`
- examples: `5`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- output: `outputs\predictions\smoke_cascade_exact.jsonl`

## 2026-08-05T02:56:44Z - Generated GSM8K predictions: determinism_check

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --adapter-dir gemma4-gsm8k-final --run-name determinism_check --limit 20`
- run_name: `determinism_check`
- task: `gsm8k`
- split: `test`
- examples: `20`
- adapter_dir: `gemma4-gsm8k-final`
- output: `outputs\predictions\determinism_check_gsm8k_test.jsonl`
- temperature: `0.0`
- max_new_tokens: `512`
- few_shot: `0`

## 2026-08-17T01:03:00Z - Cascade run: exact / none @ 100% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe supervise.py --provider exact --gate none --limit 5 --output outputs/predictions/smoke_cache_check.jsonl --verdict-cache outputs/smoke_verdict_cache.jsonl`
- provider: `exact`
- supervisor_model: `exact`
- gate: `none`
- escalation_rate: `1.0`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `5`
- examples: `5`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- output: `outputs\predictions\smoke_cache_check.jsonl`
- verdict_cache: `outputs/smoke_verdict_cache.jsonl`
- supervisor_new_calls: `8`

## 2026-08-17T01:03:17Z - Cascade run: exact / none @ 100% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe supervise.py --provider exact --gate none --limit 20 --verdict-only --output outputs/predictions/smoke_vo.jsonl --verdict-cache outputs/smoke_verdict_cache.jsonl`
- provider: `exact`
- supervisor_model: `exact`
- gate: `none`
- escalation_rate: `1.0`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `20`
- examples: `20`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- output: `outputs\predictions\smoke_vo.jsonl`
- verdict_cache: `outputs/smoke_verdict_cache.jsonl`
- supervisor_new_calls: `20`

## 2026-08-17T01:03:18Z - Cascade run: exact / none @ 100% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe supervise.py --provider exact --gate none --limit 20 --verdict-only --output outputs/predictions/smoke_vo.jsonl --verdict-cache outputs/smoke_verdict_cache.jsonl`
- provider: `exact`
- supervisor_model: `exact`
- gate: `none`
- escalation_rate: `1.0`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `20`
- examples: `20`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- output: `outputs\predictions\smoke_vo.jsonl`
- verdict_cache: `outputs/smoke_verdict_cache.jsonl`
- supervisor_new_calls: `0`

## 2026-08-17T01:07:18Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --csv-output reports/verify_fydp3.csv --md-output reports/verify_fydp3.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- cascades: `[]`
- md_output: `reports/verify_fydp3.md`
- csv_output: `reports/verify_fydp3.csv`

## 2026-08-17T06:27:05Z - Generated GSM8K predictions: samplebank_s1

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --adapter-dir gemma4-gsm8k-final --run-name samplebank_s1 --temperature 0.7`
- run_name: `samplebank_s1`
- task: `gsm8k`
- split: `test`
- examples: `1319`
- adapter_dir: `gemma4-gsm8k-final`
- output: `outputs\predictions\samplebank_s1_gsm8k_test.jsonl`
- temperature: `0.7`
- max_new_tokens: `512`
- few_shot: `0`

## 2026-08-17T11:46:15Z - Generated GSM8K predictions: samplebank_s2

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --adapter-dir gemma4-gsm8k-final --run-name samplebank_s2 --temperature 0.7`
- run_name: `samplebank_s2`
- task: `gsm8k`
- split: `test`
- examples: `1319`
- adapter_dir: `gemma4-gsm8k-final`
- output: `outputs\predictions\samplebank_s2_gsm8k_test.jsonl`
- temperature: `0.7`
- max_new_tokens: `512`
- few_shot: `0`

## 2026-08-17T11:46:33Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl --csv-output reports/fydp3_samplebank.csv --md-output reports/fydp3_samplebank.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- cascades: `[]`
- md_output: `reports/fydp3_samplebank.md`
- csv_output: `reports/fydp3_samplebank.csv`

## 2026-08-17T12:21:35Z - Generated GSM8K predictions: promptval_train

- Type: `generation`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe generate.py --adapter-dir gemma4-gsm8k-final --run-name promptval_train --split train --limit 150 --temperature 0.0`
- run_name: `promptval_train`
- task: `gsm8k`
- split: `train`
- examples: `150`
- adapter_dir: `gemma4-gsm8k-final`
- output: `outputs\predictions\promptval_train_gsm8k_train.jsonl`
- temperature: `0.0`
- max_new_tokens: `512`
- few_shot: `0`

## 2026-08-17T12:56:44Z - Cascade run: llamacpp / none @ 100% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe supervise.py --provider llamacpp --gate none --verdict-only --resume`
- provider: `llamacpp`
- supervisor_model: `GLM-4.7-Flash-UD-Q4_K_XL`
- gate: `none`
- escalation_rate: `1.0`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `1319`
- examples: `1319`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- output: `outputs\predictions\cascade_llamacpp_none_L1_test.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `1319`

## 2026-08-17T12:57:02Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- cascades: `[]`
- md_output: `reports/fydp3_summary.md`
- csv_output: `reports/fydp3_summary.csv`

## 2026-08-17T12:57:10Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl --cascade glm_verdict=outputs/predictions/cascade_llamacpp_none_L1_test.jsonl --csv-output reports/fydp3_verdict.csv --md-output reports/fydp3_verdict.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- cascades: `["glm_verdict"]`
- md_output: `reports/fydp3_verdict.md`
- csv_output: `reports/fydp3_verdict.csv`

## 2026-08-28T02:43:55Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl --csv-output reports/fydp3_samplebank.csv --md-output reports/fydp3_samplebank.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- confidence: `outputs/predictions/confidence_gsm8k_test.jsonl`
- cascades: `[]`
- md_output: `reports/fydp3_samplebank.md`
- csv_output: `reports/fydp3_samplebank.csv`

## 2026-08-28T02:48:50Z - Cascade run: llamacpp / disagreement @ 30% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe supervise.py --provider llamacpp --gate disagreement --escalation-rate 0.3 --feedback-level 1 --limit 12 --output outputs/predictions/smoke_arm_check.jsonl --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl`
- provider: `llamacpp`
- supervisor_model: `GLM-4.7-Flash-UD-Q4_K_XL`
- gate: `disagreement`
- escalation_rate: `0.3`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `4`
- examples: `12`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- output: `outputs\predictions\smoke_arm_check.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `4`

## 2026-08-28T06:46:51Z - Cascade run: llamacpp / disagreement @ 30% / L0

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe C:\thesis\gemma4-finetune\supervise.py --provider llamacpp --supervisor-url http://127.0.0.1:8080 --gate disagreement --escalation-rate 0.3 --feedback-level 0 --resume --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl`
- provider: `llamacpp`
- supervisor_model: `GLM-4.7-Flash-UD-Q4_K_XL`
- gate: `disagreement`
- escalation_rate: `0.3`
- feedback_level: `0`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `396`
- examples: `1319`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- output: `outputs\predictions\cascade_llamacpp_disagreement_L0_test.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `316`

## 2026-08-28T10:14:53Z - Cascade run: llamacpp / disagreement @ 30% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe C:\thesis\gemma4-finetune\supervise.py --provider llamacpp --supervisor-url http://127.0.0.1:8080 --gate disagreement --escalation-rate 0.3 --feedback-level 1 --resume --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl`
- provider: `llamacpp`
- supervisor_model: `GLM-4.7-Flash-UD-Q4_K_XL`
- gate: `disagreement`
- escalation_rate: `0.3`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `396`
- examples: `1319`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- output: `outputs\predictions\cascade_llamacpp_disagreement_L1_test.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `316`

## 2026-08-28T13:31:08Z - Cascade run: llamacpp / disagreement @ 30% / L2

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe C:\thesis\gemma4-finetune\supervise.py --provider llamacpp --supervisor-url http://127.0.0.1:8080 --gate disagreement --escalation-rate 0.3 --feedback-level 2 --resume --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl`
- provider: `llamacpp`
- supervisor_model: `GLM-4.7-Flash-UD-Q4_K_XL`
- gate: `disagreement`
- escalation_rate: `0.3`
- feedback_level: `2`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `396`
- examples: `1319`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- output: `outputs\predictions\cascade_llamacpp_disagreement_L2_test.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `316`

## 2026-08-28T13:34:02Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl --cascade L0=outputs/predictions/cascade_llamacpp_disagreement_L0_test.jsonl --cascade L1=outputs/predictions/cascade_llamacpp_disagreement_L1_test.jsonl --cascade L2=outputs/predictions/cascade_llamacpp_disagreement_L2_test.jsonl --csv-output reports/fydp3_summary.csv --md-output reports/fydp3_summary.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- confidence: `outputs/predictions/confidence_gsm8k_test.jsonl`
- cascades: `["L0", "L1", "L2"]`
- md_output: `reports/fydp3_summary.md`
- csv_output: `reports/fydp3_summary.csv`

## 2026-08-29T00:54:34Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl --cascade L0=outputs/predictions/cascade_llamacpp_disagreement_L0_test.jsonl --cascade L1=outputs/predictions/cascade_llamacpp_disagreement_L1_test.jsonl --cascade L2=outputs/predictions/cascade_llamacpp_disagreement_L2_test.jsonl --csv-output reports/fydp3_summary.csv --md-output reports/fydp3_summary.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- confidence: `outputs/predictions/confidence_gsm8k_test.jsonl`
- cascades: `["L0", "L1", "L2"]`
- md_output: `reports/fydp3_summary.md`
- csv_output: `reports/fydp3_summary.csv`

## 2026-08-29T04:52:41Z - Cascade run: llamacpp / disagreement @ 37% / L2

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe C:\thesis\gemma4-finetune\supervise.py --provider llamacpp --gate disagreement --escalation-rate 0.373 --feedback-level 2 --resume --output C:\thesis\gemma4-finetune\outputs\predictions\cascade_llamacpp_disagreement_r37_L2_test.jsonl --sample-bank C:\thesis\gemma4-finetune\outputs\predictions\samplebank_s1_gsm8k_test.jsonl --sample-bank C:\thesis\gemma4-finetune\outputs\predictions\samplebank_s2_gsm8k_test.jsonl`
- provider: `llamacpp`
- supervisor_model: `GLM-4.7-Flash-UD-Q4_K_XL`
- gate: `disagreement`
- escalation_rate: `0.373`
- feedback_level: `2`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `492`
- examples: `1319`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["C:\\thesis\\gemma4-finetune\\outputs\\predictions\\samplebank_s1_gsm8k_test.jsonl", "C:\\thesis\\gemma4-finetune\\outputs\\predictions\\samplebank_s2_gsm8k_test.jsonl"]`
- output: `C:\thesis\gemma4-finetune\outputs\predictions\cascade_llamacpp_disagreement_r37_L2_test.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `397`

## 2026-08-29T04:58:30Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl --cascade L0=outputs/predictions/cascade_llamacpp_disagreement_L0_test.jsonl --cascade L1=outputs/predictions/cascade_llamacpp_disagreement_L1_test.jsonl --cascade L2=outputs/predictions/cascade_llamacpp_disagreement_L2_test.jsonl --cascade L2@37%=outputs/predictions/cascade_llamacpp_disagreement_r37_L2_test.jsonl --csv-output reports/fydp3_summary.csv --md-output reports/fydp3_summary.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- confidence: `outputs/predictions/confidence_gsm8k_test.jsonl`
- cascades: `["L0", "L1", "L2", "L2@37%"]`
- md_output: `reports/fydp3_summary.md`
- csv_output: `reports/fydp3_summary.csv`

## 2026-08-29T05:50:23Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl --cascade L0=outputs/predictions/cascade_llamacpp_disagreement_L0_test.jsonl --cascade L1=outputs/predictions/cascade_llamacpp_disagreement_L1_test.jsonl --cascade L2=outputs/predictions/cascade_llamacpp_disagreement_L2_test.jsonl --cascade L2@37%=outputs/predictions/cascade_llamacpp_disagreement_r37_L2_test.jsonl --csv-output reports/fydp3_summary.csv --md-output reports/fydp3_summary.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- confidence: `outputs/predictions/confidence_gsm8k_test.jsonl`
- cascades: `["L0", "L1", "L2", "L2@37%"]`
- md_output: `reports/fydp3_summary.md`
- csv_output: `reports/fydp3_summary.csv`

## 2026-08-29T09:45:02Z - Cascade run: llamacpp / combined @ 30% / L2

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe C:\thesis\gemma4-finetune\supervise.py --provider llamacpp --gate combined --escalation-rate 0.3 --feedback-level 2 --resume --output C:\thesis\gemma4-finetune\outputs\predictions\cascade_llamacpp_combined_r30_L2_test.jsonl --confidence C:\thesis\gemma4-finetune\outputs\predictions\confidence_gsm8k_test.jsonl --sample-bank C:\thesis\gemma4-finetune\outputs\predictions\samplebank_s1_gsm8k_test.jsonl --sample-bank C:\thesis\gemma4-finetune\outputs\predictions\samplebank_s2_gsm8k_test.jsonl`
- provider: `llamacpp`
- supervisor_model: `GLM-4.7-Flash-UD-Q4_K_XL`
- gate: `combined`
- escalation_rate: `0.3`
- feedback_level: `2`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `396`
- examples: `1319`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["C:\\thesis\\gemma4-finetune\\outputs\\predictions\\samplebank_s1_gsm8k_test.jsonl", "C:\\thesis\\gemma4-finetune\\outputs\\predictions\\samplebank_s2_gsm8k_test.jsonl"]`
- output: `C:\thesis\gemma4-finetune\outputs\predictions\cascade_llamacpp_combined_r30_L2_test.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `332`

## 2026-08-29T09:48:02Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl --cascade L2=outputs/predictions/cascade_llamacpp_disagreement_L2_test.jsonl --cascade L2@37%=outputs/predictions/cascade_llamacpp_disagreement_r37_L2_test.jsonl --cascade L2-combined=outputs/predictions/cascade_llamacpp_combined_r30_L2_test.jsonl --csv-output reports/fydp3_summary.csv --md-output reports/fydp3_summary.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- confidence: `outputs/predictions/confidence_gsm8k_test.jsonl`
- cascades: `["L2", "L2@37%", "L2-combined"]`
- md_output: `reports/fydp3_summary.md`
- csv_output: `reports/fydp3_summary.csv`

## 2026-08-29T11:24:40Z - Cascade run: llamacpp / none @ 100% / L1

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe C:\thesis\gemma4-finetune\supervise.py --provider llamacpp --supervisor-model Qwen3.5-9B-UD-Q4_K_XL --gate none --verdict-only --resume --output C:\thesis\gemma4-finetune\outputs\predictions\verdict_qwen_test.jsonl`
- provider: `llamacpp`
- supervisor_model: `Qwen3.5-9B-UD-Q4_K_XL`
- gate: `none`
- escalation_rate: `1.0`
- feedback_level: `1`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `1319`
- examples: `1319`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- output: `C:\thesis\gemma4-finetune\outputs\predictions\verdict_qwen_test.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `1319`

## 2026-08-29T11:25:09Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --cascade qwen-9B=outputs/predictions/verdict_qwen_test.jsonl --csv-output reports/fydp3_verdict_qwen.csv --md-output reports/fydp3_verdict_qwen.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `[]`
- confidence: `outputs/predictions/confidence_gsm8k_test.jsonl`
- cascades: `["qwen-9B"]`
- md_output: `reports/fydp3_verdict_qwen.md`
- csv_output: `reports/fydp3_verdict_qwen.csv`

## 2026-08-29T15:16:12Z - Cascade run: llamacpp / combined @ 30% / L2

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe C:\thesis\gemma4-finetune\supervise.py --provider llamacpp --gate combined --escalation-rate 0.3 --supervisor-model Qwen3.5-9B-UD-Q4_K_XL --feedback-level 2 --resume --output C:\thesis\gemma4-finetune\outputs\predictions\cascade_llamacpp_combined_r30_L2_qwen_test.jsonl --confidence C:\thesis\gemma4-finetune\outputs\predictions\confidence_gsm8k_test.jsonl --sample-bank C:\thesis\gemma4-finetune\outputs\predictions\samplebank_s1_gsm8k_test.jsonl --sample-bank C:\thesis\gemma4-finetune\outputs\predictions\samplebank_s2_gsm8k_test.jsonl`
- provider: `llamacpp`
- supervisor_model: `Qwen3.5-9B-UD-Q4_K_XL`
- gate: `combined`
- escalation_rate: `0.3`
- feedback_level: `2`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `396`
- examples: `1319`
- attempt1_cache: `outputs\predictions\fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["C:\\thesis\\gemma4-finetune\\outputs\\predictions\\samplebank_s1_gsm8k_test.jsonl", "C:\\thesis\\gemma4-finetune\\outputs\\predictions\\samplebank_s2_gsm8k_test.jsonl"]`
- output: `C:\thesis\gemma4-finetune\outputs\predictions\cascade_llamacpp_combined_r30_L2_qwen_test.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `350`

## 2026-08-29T15:16:31Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/samplebank_s1_gsm8k_test.jsonl --sample-bank outputs/predictions/samplebank_s2_gsm8k_test.jsonl --cascade GLM-combined=outputs/predictions/cascade_llamacpp_combined_r30_L2_test.jsonl --cascade Qwen-combined=outputs/predictions/cascade_llamacpp_combined_r30_L2_qwen_test.jsonl --cascade GLM@37%=outputs/predictions/cascade_llamacpp_disagreement_r37_L2_test.jsonl --csv-output reports/fydp3_summary.csv --md-output reports/fydp3_summary.md`
- baseline: `outputs/predictions/fine_tuned_gsm8k_test.jsonl`
- sample_bank: `["outputs/predictions/samplebank_s1_gsm8k_test.jsonl", "outputs/predictions/samplebank_s2_gsm8k_test.jsonl"]`
- confidence: `outputs/predictions/confidence_gsm8k_test.jsonl`
- cascades: `["GLM-combined", "Qwen-combined", "GLM@37%"]`
- md_output: `reports/fydp3_summary.md`
- csv_output: `reports/fydp3_summary.csv`

## 2026-08-29T15:35:49Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl --sample-bank outputs/predictions/04_answers_finetuned_try3.jsonl --cascade GLM-combined=outputs/predictions/14_pipeline_glm30b_hint_full_smart_gate.jsonl --cascade Qwen-combined=outputs/predictions/15_pipeline_qwen9b_BEST_RESULT.jsonl --cascade GLM@37%=outputs/predictions/13_pipeline_glm30b_hint_full_more_escalation.jsonl --csv-output reports/fydp3_summary.csv --md-output reports/fydp3_summary.md`
- baseline: `outputs/predictions/02_answers_finetuned_try1_main.jsonl`
- sample_bank: `["outputs/predictions/03_answers_finetuned_try2.jsonl", "outputs/predictions/04_answers_finetuned_try3.jsonl"]`
- confidence: `outputs/predictions/05_confidence_for_try1.jsonl`
- cascades: `["GLM-combined", "Qwen-combined", "GLM@37%"]`
- md_output: `reports/fydp3_summary.md`
- csv_output: `reports/fydp3_summary.csv`

## 2026-08-29T15:41:40Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl --sample-bank outputs/predictions/04_answers_finetuned_try3.jsonl --cascade hint-none=outputs/predictions/10_pipeline_glm30b_hint_none.jsonl --cascade hint-short=outputs/predictions/11_pipeline_glm30b_hint_short.jsonl --cascade hint-full=outputs/predictions/12_pipeline_glm30b_hint_full.jsonl --cascade more-escalation=outputs/predictions/13_pipeline_glm30b_hint_full_more_escalation.jsonl --cascade smart-gate=outputs/predictions/14_pipeline_glm30b_hint_full_smart_gate.jsonl --cascade BEST-qwen9b=outputs/predictions/15_pipeline_qwen9b_BEST_RESULT.jsonl --csv-output reports/fydp3_summary.csv --md-output reports/fydp3_summary.md`
- baseline: `outputs/predictions/02_answers_finetuned_try1_main.jsonl`
- sample_bank: `["outputs/predictions/03_answers_finetuned_try2.jsonl", "outputs/predictions/04_answers_finetuned_try3.jsonl"]`
- confidence: `outputs/predictions/05_confidence_for_try1.jsonl`
- cascades: `["hint-none", "hint-short", "hint-full", "more-escalation", "smart-gate", "BEST-qwen9b"]`
- md_output: `reports/fydp3_summary.md`
- csv_output: `reports/fydp3_summary.csv`

## 2026-08-29T15:41:41Z - Analyzed FYDP 3 cascade runs

- Type: `analysis`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe analyze_supervision.py --cascade qwen9b-checker=outputs/predictions/08_checker_qwen9b_marks_all.jsonl --csv-output reports/fydp3_verdict_qwen.csv --md-output reports/fydp3_verdict_qwen.md`
- baseline: `outputs/predictions/02_answers_finetuned_try1_main.jsonl`
- sample_bank: `[]`
- confidence: `outputs/predictions/05_confidence_for_try1.jsonl`
- cascades: `["qwen9b-checker"]`
- md_output: `reports/fydp3_verdict_qwen.md`
- csv_output: `reports/fydp3_verdict_qwen.csv`

## 2026-08-30T05:11:31Z - Cascade run: llamacpp / combined @ 100% / L2

- Type: `supervision`
- command: `C:\thesis\gemma4-finetune\venv\Scripts\python.exe supervise.py --provider llamacpp --supervisor-model Qwen3.5-9B-UD-Q4_K_XL --gate combined --escalation-rate 1.0 --feedback-level 2 --limit 3 --output outputs/predictions/checks/timing_smoke.jsonl --sample-bank outputs/predictions/03_answers_finetuned_try2.jsonl --sample-bank outputs/predictions/04_answers_finetuned_try3.jsonl`
- provider: `llamacpp`
- supervisor_model: `Qwen3.5-9B-UD-Q4_K_XL`
- gate: `combined`
- escalation_rate: `1.0`
- feedback_level: `2`
- retry_limit: `2`
- retry_temperature: `0.7`
- escalated: `3`
- examples: `3`
- attempt1_cache: `outputs\predictions\02_answers_finetuned_try1_main.jsonl`
- sample_bank: `["outputs/predictions/03_answers_finetuned_try2.jsonl", "outputs/predictions/04_answers_finetuned_try3.jsonl"]`
- output: `outputs\predictions\checks\timing_smoke.jsonl`
- verdict_cache: `outputs\verdict_cache.jsonl`
- supervisor_new_calls: `2`

