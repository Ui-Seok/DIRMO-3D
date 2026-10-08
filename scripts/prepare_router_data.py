"""Select measured router data for one pocket/kernel and the configured steps."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import tempfile


def prepare(dataset: Path, output: Path, config: Path, kernel_id: str | None = None) -> dict:
    if dataset.resolve() == output.resolve():
        raise ValueError("Training output must differ from the raw collection dataset")
    settings = json.loads(config.read_text())
    steps = settings.get("fm", {}).get("parameters", {}).get("integration_steps", 64)
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 256:
        raise ValueError("Config integration_steps must be an integer in 1..256")
    eligible = []
    total = 0
    matching_statuses = Counter()
    failure_example = None
    for number, line in enumerate(dataset.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"Malformed JSON on dataset line {number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Dataset line {number} must be an object")
        total += 1
        if row.get("integration_steps") != steps:
            continue
        generation = row.get("generation_result", {})
        matching_statuses[generation.get("status", "unknown")] += 1
        if row.get("utility_change") is None:
            if failure_example is None and generation.get("reason"):
                failure_example = generation["reason"]
            continue
        value, identity = row["utility_change"], row.get("kernel_id")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Non-finite or invalid measured utility on dataset line {number}")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"Measured dataset line {number} is missing kernel_id")
        eligible.append(row)

    counts = Counter(row["kernel_id"] for row in eligible)
    if kernel_id is None:
        if not counts:
            detail = f" Matching generation statuses: {dict(matching_statuses)}."
            if failure_example:
                detail += f" First failure: {failure_example}"
            raise ValueError(f"No measured outcomes at configured integration_steps={steps}." + detail)
        if len(counts) > 1:
            detail = ", ".join(f"{identity}:{count}" for identity, count in sorted(counts.items()))
            raise ValueError(f"Multiple pocket/kernels at steps={steps}. "
                             f"Set DIRMO_KERNEL_ID or --kernel-id. Measured counts: {detail}")
        kernel_id = next(iter(counts))
    selected = [row for row in eligible if row["kernel_id"] == kernel_id]
    if len(selected) < 2:
        raise ValueError(f"Need at least two measured outcomes in the selected kernel at steps={steps}; "
                         f"found {len(selected)}. Inspect actions.jsonl/llm.jsonl and collect more real outcomes.")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=output.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            for row in selected:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"status": "prepared", "source": str(dataset), "output": str(output),
            "integration_steps": steps, "kernel_id": kernel_id, "source_rows": total,
            "training_rows": len(selected), "excluded_rows": total - len(selected),
            "note": "Original collection data is unchanged; no labels were invented."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--kernel-id")
    args = parser.parse_args()
    try:
        result = prepare(args.dataset, args.output, args.config, args.kernel_id)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
