from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
]


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    checkpoints = [
        item
        for item in checkpoint_dir.iterdir()
        if item.is_dir() and checkpoint_step(item) >= 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=checkpoint_step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Hugging Face Trainer checkpoint.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also torch.load optimizer/scheduler/RNG/training_args files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(Path(args.checkpoint_dir))
    if checkpoint is None:
        raise SystemExit("No checkpoint-* directory found.")

    print(f"Checking checkpoint: {checkpoint}")
    missing = [name for name in REQUIRED_FILES if not (checkpoint / name).exists()]
    if missing:
        raise SystemExit(f"Missing required files: {', '.join(missing)}")

    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    print(f"global_step: {state.get('global_step')}")
    print(f"epoch: {state.get('epoch')}")
    print(f"max_steps: {state.get('max_steps')}")
    print(f"save_steps: {state.get('save_steps')}")
    print("required files: OK")

    if args.deep:
        # Import datasets before torch to match the stable import order used by train.py.
        from datasets import load_from_disk  # noqa: F401
        import torch

        for name in ["training_args.bin", "scheduler.pt", "rng_state.pth", "optimizer.pt"]:
            path = checkpoint / name
            print(f"torch.load {path.name} ...", flush=True)
            torch.load(path, map_location="cpu", weights_only=False)
        print("torch.load files: OK")


if __name__ == "__main__":
    main()
