#!/usr/bin/env python3
"""Train a small Qwen3.5-9B SMDD LoRA pilot on one explicitly selected GPU."""
import argparse
import json
from pathlib import Path

from dirmo.training.smdd_sft import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/LocalData1/uslee/checkpoints/Qwen3.5-9B"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--eval-generations", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=28)
    parser.add_argument("--memory-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2), flush=True)
    except Exception as exc:
        if args.output.exists():
            (args.output / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "reason": str(exc)}, indent=2) + "\n")
        raise
