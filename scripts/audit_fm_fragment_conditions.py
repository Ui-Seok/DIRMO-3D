"""Audit recorded fragment masks and prior coordinates without running FLOWR."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem


def audit(run: Path, output: Path):
    study = json.loads((run / "study.json").read_text())
    references = {}
    for case in study["cases"]:
        mol = Chem.SDMolSupplier(case["truth_sdf"], removeHs=True)[0]
        if mol is None:
            raise ValueError(f"Cannot read reference for {case['case_id']}")
        references[case["case_id"]] = mol
    total, groups, records = Counter(), defaultdict(Counter), []
    for file in sorted(run.glob("draws/*_fragment/*/*/*/model-condition.json")):
        phase, case_id, condition, seed = file.parts[-5:-1]
        item = json.loads(file.read_text())
        mol = references[case_id]
        native = mol.GetConformer().GetPositions()
        prior = np.asarray(item["prior_coords_absolute"], dtype=float)
        held = np.asarray(item["actual_held_mask"], dtype=bool)
        maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
        expected = np.asarray([m not in item["expected_free_maps"] for m in maps])
        if not np.array_equal(held, expected) or prior.shape != native.shape:
            raise ValueError(f"Recorded condition mismatch: {file}")
        free = np.flatnonzero(~held)
        same = np.linalg.norm(prior[free] - native[free], axis=1) <= .001
        distances = np.linalg.norm(prior[free, None, :] - native[None, :, :], axis=2)
        matches = distances.min(axis=1) <= .001
        counts = {"draws": 1, "free_slots": len(free),
                  "draws_with_free_at_same_native_slot": int(same.any()),
                  "draws_with_free_at_any_native_atom": int(matches.any()),
                  "free_at_same_native_slot": int(same.sum()),
                  "free_at_any_native_atom": int(matches.sum())}
        total.update(counts)
        groups[(phase, condition)].update(counts)
        records.append({"file": str(file), "phase": phase, "case_id": case_id,
                        "condition": condition, "seed": int(seed),
                        "free_atom_maps": [maps[i] for i in free],
                        "free_at_same_native_slot": same.tolist(),
                        "free_at_any_native_atom": matches.tolist(),
                        "nearest_native_atom_maps": [maps[i] for i in distances.argmin(axis=1)],
                        "held_max_drift": float(np.linalg.norm(prior[held] - native[held], axis=1).max())})
    source = Path(__file__).resolve().parents[1] / "src/dirmo/generation/flowr.py"
    result = {"source": str(run), "coordinate_tolerance_angstrom": .001,
              "recorded_adapter_sha256": study["adapter_sha256"],
              "current_adapter_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "totals": dict(total),
              "groups": [{"phase": phase, "condition": condition, **counts}
                         for (phase, condition), counts in sorted(groups.items())],
              "interpretation": "The requested held mask is correct, but recorded free priors are not uniformly independent of reference coordinates. Coordinate matches do not establish which discrete features were present; these were not recorded.",
              "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "totals": result["totals"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.output.resolve())
