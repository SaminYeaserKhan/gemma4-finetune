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

