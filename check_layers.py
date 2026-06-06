# This script prints all layer names inside Gemma 4 E2B
# so we can identify exactly which ones to target with LoRA

import torch
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForImageTextToText.from_pretrained(
    "google/gemma-4-E2B-it",
    quantization_config=bnb_config,
    device_map="auto",
)

# Print every named module and its type
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")