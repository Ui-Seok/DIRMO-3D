#!/usr/bin/env python3
"""Separate strict response-format errors from molecular errors in saved outputs."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from dirmo.data.smdd import assess_product
from dirmo.training.smdd_sft import summarize_predictions


def analyze(data: Path, predictions: Path) -> dict:
    rows = {r["task_id"]: r for r in (json.loads(line) for line in data.read_text().splitlines() if line.strip())}
    strict, unwrapped = [], []
    failures = Counter()
    pairs = []
    seen = set()
    for line in predictions.read_text().splitlines():
        item = json.loads(line)
        if item["task_id"] in seen:
            raise ValueError(f"Duplicate prediction: {item['task_id']}")
        seen.add(item["task_id"])
        row = rows[item["task_id"]]
        raw = item["raw"]
        first = assess_product(raw, row["reference_smiles"], row["witness_smiles"])
        # Only remove a complete enclosing Markdown fence. Do not repair syntax,
        # chemistry, truncated molecules, or extra prose.
        match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", raw, flags=re.DOTALL)
        relaxed_raw = match.group(1).strip() if match else raw
        second = assess_product(relaxed_raw, row["reference_smiles"], row["witness_smiles"])
        strict.append({"assessment": first})
        unwrapped.append({"assessment": second})
        failures.update(k for k, v in second.get("checks", {}).items() if not v)
        pairs.append({"task_id": item["task_id"], "markdown_fence_removed": match is not None,
                      "strict": first, "fence_removed_diagnostic": second})
    return {"source": str(predictions), "strict": summarize_predictions(strict),
            "fence_removed_diagnostic": summarize_predictions(unwrapped),
            "failed_local_constraints_after_fence_removal": dict(failures), "rows": pairs,
            "note": "Fence removal is diagnostic only. No molecular repair or oracle goal evaluation was performed."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze(args.data, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
