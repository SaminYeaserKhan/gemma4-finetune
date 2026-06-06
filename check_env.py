from __future__ import annotations

import argparse

import accelerate
import bitsandbytes
import peft
import torch
import transformers
import trl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the local ML environment.")
    parser.add_argument(
        "--load-tokenizer",
        action="store_true",
        help="Also verify that the base tokenizer can be loaded.",
    )
    parser.add_argument("--model-name", default="google/gemma-4-E2B-it")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("=== Core ML Stack Versions ===")
    print(f"PyTorch:      {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"BitsAndBytes: {bitsandbytes.__version__}")
    print(f"PEFT:         {peft.__version__}")
    print(f"TRL:          {trl.__version__}")
    print(f"Accelerate:   {accelerate.__version__}")

    print("\n=== Hardware & CUDA ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version (PyTorch compiled with): {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {total_gb:.2f} GB")

    if args.load_tokenizer:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(args.model_name)
        print(f"\nTokenizer loaded: {args.model_name}")


if __name__ == "__main__":
    main()

