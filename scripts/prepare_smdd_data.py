#!/usr/bin/env python3
"""Prepare reference→witness instruction pairs without cross-split leakage."""
import argparse
import json
from pathlib import Path

from dirmo.data.smdd import prepare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/LocalData1/uslee/datasets/smdd-bench/tasks"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output, args.seed), indent=2))
