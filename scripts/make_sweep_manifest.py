"""Build a JSONL case manifest from a processed CrossDocked split.

The objective and budgets are copied from a template case so they live in one place:
``train_router`` accepts only one ``objective_sha256`` per dataset, so every case in a
sweep must share the template's objective exactly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
import tempfile

CASE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def cases(directory: Path, template: dict, limit: int | None = None) -> list[dict]:
    found, skipped = [], []
    for pocket in sorted(directory.glob("*-pocket10.pdb")):
        stem = pocket.name[: -len("-pocket10.pdb")]
        receptor = directory / f"{stem}-pocket10.pdbqt"
        ligands = sorted(directory.glob(f"{stem}-pocket10_*.sdf"))
        if not receptor.is_file() or not ligands:
            skipped.append({"stem": stem, "reason": "missing receptor or ligand"})
            continue
        if not CASE_ID.fullmatch(stem):
            skipped.append({"stem": stem, "reason": "case_id pattern"})
            continue
        left, separator, right = stem.partition("-rec-")
        if not separator:
            skipped.append({"stem": stem, "reason": "no -rec- in name"})
            continue
        entry = {key: value for key, value in template.items()
                 if key in ("coordinate_units", "objective", "protected", "budgets")}
        entry.update(
            case_id=stem,
            # The grouping unit for grouped cross-validation: one protein per group.
            target_id=left.replace("-", "_"),
            series_id="-".join(right.split("-")[:2]),
            pocket_path=str(pocket.resolve()),
            initial_ligand_path=str(ligands[0].resolve()),
            receptor_path=str(receptor.resolve()),
            preparation_notes=template.get("preparation_notes", "As supplied; no receptor repair."),
        )
        found.append(entry)
        if limit and len(found) >= limit:
            break
    return found, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path,
                        help="Case JSON whose objective/protected/budgets every case reuses")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="Write only the first N cases (smoke tests)")
    args = parser.parse_args()
    try:
        if not args.dataset_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {args.dataset_dir}")
        template = json.loads(args.template.read_text())
        found, skipped = cases(args.dataset_dir, template, args.limit)
        if not found:
            raise ValueError("No complete pocket/receptor/ligand triples found")
        identifiers = [entry["case_id"] for entry in found]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate case_id")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=args.output.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            for entry in found:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
        temporary.replace(args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    print(json.dumps({"status": "written", "manifest": str(args.output), "cases": len(found),
                      "distinct_targets": len({entry["target_id"] for entry in found}),
                      "distinct_series": len({entry["series_id"] for entry in found}),
                      "skipped": skipped,
                      "objective_source": str(args.template),
                      "note": "All cases share the template objective; grouped CV groups on target_id."},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
