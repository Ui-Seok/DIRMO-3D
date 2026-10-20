"""Read-only forensic review of completed FM draws; original scores stay intact."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dirmo.chemistry.graphs import read_sdf, smiles
from dirmo.io import write_json, sha256_file


def coordinate_stereo(molecule):
    mol = Chem.Mol(molecule)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # MDL's crossed-double-bond marker survives replaceExistingTags=True and
    # masks a defined geometry. Clear annotations, then infer entirely from XYZ.
    for bond in mol.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def bootstrap(values):
    values = np.array(values, dtype=float)
    if not len(values):
        return None
    estimate = np.random.default_rng(2026).choice(values, (5000, len(values)), replace=True).mean(1)
    return {"mean": float(values.mean()), "95ci": np.quantile(estimate, [.025, .975]).tolist()}


def audit(root, output):
    plan = json.loads((root / "study.json").read_text())
    rows = [json.loads(line) for line in (root / "draws.jsonl").read_text().splitlines()]
    if len(rows) != plan["planned_fm_draws"]:
        raise ValueError("Review requires the completed generation dataset")
    case_by_id = {case["case_id"]: case for case in plan["cases"]}
    truths = {key: read_sdf(case["truth_sdf"]) for key, case in case_by_id.items()}
    groups, reviewed = defaultdict(list), []
    for row in rows:
        original = row["metrics"]
        item = {"draw_id": row["draw_id"], "case_id": row["case_id"], "target_id": row["target_id"],
                "seed": row["seed"], "checkpoint": row["checkpoint"], "condition": row["condition"],
                "directory": row["directory"], "mode": row["mode"], "original": original,
                "generation": row["generation"], "original_contract": original.get("contract_valid", False),
                "coordinate_contract": False, "coordinate_pose_success": False,
                "coordinate_changed_graph": False, "constitutional_changed_graph": False,
                "coordinate_valid_changed_fragment": False,
                "worker_vs_saved_mismatch": False}
        raw = row["generation"].get("sdf_path")
        if raw:
            try:
                molecule = read_sdf(raw)
                assigned = coordinate_stereo(molecule)
                item["coordinate_smiles"] = smiles(assigned)
                item["coordinate_changed_graph"] = smiles(assigned) != smiles(truths[row["case_id"]])
                left, right = Chem.Mol(assigned), Chem.Mol(truths[row["case_id"]])
                for mol in (left, right):
                    for atom in mol.GetAtoms():
                        atom.SetAtomMapNum(0)
                item["constitutional_changed_graph"] = (
                    Chem.MolToSmiles(left, isomericSmiles=False) != Chem.MolToSmiles(right, isomericSmiles=False))
                input_file = Path(row["directory"]) / "model-condition.json"
                if input_file.exists():
                    model_input = json.loads(input_file.read_text())
                    positions = molecule.GetConformer().GetPositions()
                    prior = np.array(model_input["prior_coords_absolute"])
                    item["raw_minus_prior_rmsd"] = float(np.sqrt(np.mean(np.sum((positions - prior) ** 2, axis=1))))
                if row["mode"] == "graph":
                    item["coordinate_contract"] = smiles(assigned) == smiles(truths[row["case_id"]])
                    item["worker_vs_saved_mismatch"] = bool(row["generation"].get("graph_check", {}).get("matches")
                                                             and not item["original_contract"])
                    item["stereoany_bond_count"] = sum(bond.GetStereo() == Chem.BondStereo.STEREOANY for bond in molecule.GetBonds())
                    if item["coordinate_contract"]:
                        reference = Chem.Mol(truths[row["case_id"]])
                        for atom in reference.GetAtoms():
                            atom.SetAtomMapNum(0)
                        item["coordinate_rmsd"] = float(rdMolAlign.CalcRMS(assigned, reference, maxMatches=4096))
                        item["coordinate_pose_success"] = bool(original.get("pb_valid", False) and item["coordinate_rmsd"] <= 2.)
                else:
                    item["coordinate_contract"] = item["original_contract"]
                    item["coordinate_valid_changed_fragment"] = bool(
                        item["coordinate_contract"] and original.get("pb_valid", False) and item["coordinate_changed_graph"])
            except Exception as exc:
                item["audit_error"] = f"{type(exc).__name__}: {exc}"
        full_file = Path(row["directory"]) / "full-protein-posebusters.json"
        if full_file.exists():
            full = json.loads(full_file.read_text())
            item["full"] = full
            criterion = (item.get("coordinate_rmsd", float("inf")) <= 2. if row["mode"] == "graph"
                         else item["coordinate_changed_graph"])
            item["coordinate_full_success"] = bool(item["coordinate_contract"] and full.get("passes", False) and criterion)
        reviewed.append(item)
        groups[(item["checkpoint"], item["condition"])].append(item)
    stats = []
    for key, group in sorted(groups.items()):
        prior_rmsd = [x["raw_minus_prior_rmsd"] for x in group if "raw_minus_prior_rmsd" in x]
        mismatch = [x for x in group if x["worker_vs_saved_mismatch"]]
        graph_differences = []
        geometry_fail, geometry_missing, energy_unavailable = Counter(), Counter(), 0
        duplicates = defaultdict(set)
        for item in group:
            o = item["original"]
            for name, passed in o.get("posebusters", {}).get("checks", {}).items():
                if passed is False:
                    geometry_fail[name] += 1
                elif passed is None:
                    geometry_missing[name] += 1
            continuous = o.get("posebusters", {}).get("continuous", {})
            if o.get("posebusters", {}).get("checks", {}).get("internal_energy") is False and continuous.get("energy_ratio") is None:
                energy_unavailable += 1
            path = item["generation"].get("sdf_path")
            if path:
                molecule = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)[0]
                if molecule is not None:
                    duplicates[item["case_id"]].add(tuple(molecule.GetConformer().GetPositions().round(4).ravel()))
            if item["worker_vs_saved_mismatch"]:
                graph_differences.append({"case_id": item["case_id"], "seed": item["seed"],
                                          "coordinate_recovers_contract": item["coordinate_contract"],
                                          "stereoany_bonds": item.get("stereoany_bond_count", 0)})
        stats.append({"checkpoint": key[0], "condition": key[1], "attempts": len(group),
                      "build_valid": sum(x["original"].get("build_valid", False) for x in group),
                      "legacy_contract": sum(x["original_contract"] for x in group),
                      "coordinate_contract": sum(x["coordinate_contract"] for x in group),
                      "pocket_pb": sum(x["original"].get("pb_valid", False) for x in group),
                      "legacy_pose_success": sum(x["original"].get("pose_success", False) for x in group),
                      "coordinate_pose_success": sum(x["coordinate_pose_success"] for x in group),
                      "changed_valid_fragment": sum(x["original"].get("valid_changed_fragment", False) for x in group),
                      "coordinate_changed_valid_fragment": sum(x["coordinate_valid_changed_fragment"] for x in group),
                      "valid_constitutional_changes": sum(x["coordinate_contract"] and x["original"].get("pb_valid", False)
                                                        and x["constitutional_changed_graph"] for x in group)
                                                     if group[0]["mode"] == "fragment" else None,
                      "serialization_only_changed_valid_fragments": sum(x["original"].get("valid_changed_fragment", False)
                                                                        and not x["coordinate_changed_graph"] for x in group),
                      "full_evaluated": sum("full" in x for x in group),
                      "coordinate_full_success": sum(x.get("coordinate_full_success", False) for x in group),
                      "worker_vs_saved_mismatch": len(mismatch),
                      "mismatches_coordinate_recovered": sum(x["coordinate_contract"] for x in mismatch),
                      "mismatch_examples": graph_differences,
                      "geometry_failed_checks": dict(geometry_fail), "geometry_missing_checks": dict(geometry_missing),
                      "energy_unavailable": energy_unavailable,
                      "median_raw_prior_rmsd": float(np.median(prior_rmsd)) if prior_rmsd else None,
                      "unique_raw_coordinate_counts": {k: len(v) for k, v in duplicates.items()}})
    comparisons = []
    for condition in ("fragment_small", "fragment_large"):
        lookup = {(x["checkpoint"], x["target_id"], x["seed"]): x for x in reviewed if x["condition"] == condition}
        targets, corrected_targets = defaultdict(list), defaultdict(list)
        for item in groups[("base", condition)]:
            other = lookup[("v2", item["target_id"], item["seed"])]
            targets[item["target_id"]].append(int(other["original"].get("valid_changed_fragment", False)) -
                                               int(item["original"].get("valid_changed_fragment", False)))
            corrected_targets[item["target_id"]].append(int(other["coordinate_valid_changed_fragment"]) -
                                                         int(item["coordinate_valid_changed_fragment"]))
        comparisons.append({"condition": condition, "targets": len(targets),
                            "v2_minus_base": bootstrap([np.mean(v) for v in targets.values()]),
                            "coordinate_v2_minus_base": bootstrap([np.mean(v) for v in corrected_targets.values()])})
    report = {"generated_attempts": len(rows), "raw_draws": sum(x["generation"]["raw_draws"] for x in reviewed),
              "nfe": sum(x["generation"]["nfe"] for x in reviewed), "groups": stats, "paired_fragment": comparisons,
              "original_dataset_sha256": sha256_file(root / "draws.jsonl"),
              "method": "Original metrics retained. On a copy clear MDL bond stereo annotations and derive atom/bond stereo exclusively from XYZ; compare exact canonical identity, then compute unaligned symmetry RMSD.",
              "full_protein_files_present": sum("full" in x for x in reviewed),
              "full_protein_files_expected": sum(bool(x["generation"].get("sdf_path")) for x in reviewed)}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "forensic-summary.json", report)
    with (output / "coordinate-stereo-audit.jsonl").open("w") as file:
        for item in reviewed:
            file.write(json.dumps(item, allow_nan=False) + "\n")
    print(json.dumps({"attempts": len(rows), "output": str(output), "full_present": report["full_protein_files_present"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve(), args.output.resolve())
