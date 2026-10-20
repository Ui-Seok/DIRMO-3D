#!/usr/bin/env python3
"""Continue the completed SMDD adapter and evaluate every split in full."""
import argparse
import json
from pathlib import Path

from dirmo.io import write_json
from dirmo.training.smdd_full import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/LocalData1/uslee/checkpoints/Qwen3.5-9B"))
    parser.add_argument("--pilot", type=Path, default=Path("runs/smdd_lora_02"))
    parser.add_argument("--data", type=Path, default=Path("runs/smdd_lora_02/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-epochs", type=int, default=5)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[2e-5, 5e-5])
    parser.add_argument("--microbatch", type=int, default=8)
    parser.add_argument("--generation-batch", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--min-free-gib", type=float, default=36)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps({k: v for k, v in result.items() if k not in ["final", "history", "lr_trials"]}, indent=2))
    except Exception as exc:
        if args.output.exists():
            write_json(args.output / "failure.json", {"type": type(exc).__name__, "reason": str(exc)})
        raise
