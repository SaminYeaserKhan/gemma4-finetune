# Prints only the text transformer layers we can safely apply LoRA to
# Excludes the vision and audio encoder layers that cause errors

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

print("=== TEXT LAYERS SAFE FOR LORA ===")
for name, module in model.named_modules():
    # Skip vision and audio encoder layers entirely
    if "vision_tower" in name or "audio_tower" in name:
        continue
    # Only show Linear layers (these are the ones LoRA can target)
    if "Linear" in type(module).__name__:
        print(f"{name}: {type(module).__name__}")