"""Audit saved PB repairs and score unchanged poses with CPU AutoDock Vina."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from rdkit import Chem
from dirmo.chemistry.graphs import graph_check, read_sdf, smiles
from dirmo.chemistry.properties import graph_properties
from dirmo.io import sha256_file, write_json


def read(path):
    return json.loads(Path(path).read_text())


def csv_save(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def describe(path):
    mol = read_sdf(path)
    return {"smiles": smiles(mol), **graph_properties(smiles(mol))}


def score_case(task):
    from Bio.PDB import PDBParser
    from dirmo.evaluation.fm_quality import prepare_receptor, vina_for_mol
    case, items, destination = task
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / "scores.json"
    identity = {"protein_sha256": sha256_file(case["protein_path"]),
                "native_sha256": sha256_file(case["native_sdf"]),
                "inputs": {key: sha256_file(path) for key, path in items.items()},
                "protocol": "Meeko standard amino acids; HIE histidines; raw heavy XYZ; Vina score only; 40 A native-centered box"}
    if result_path.exists():
        saved = read(result_path)
        if saved.get("identity") == identity:
            return saved
    protein = PDBParser(QUIET=True).get_structure(case["case_id"], case["protein_path"])
    receptor, status = prepare_receptor(protein, destination)
    native = read_sdf(case["native_sdf"])
    center = native.GetConformer().GetPositions().mean(axis=0).tolist()
    scoring_case = {"receptor_path": receptor, "box_center": center, "box_size": [40., 40., 40.]}
    scores = {}
    for key, path in items.items():
        try:
            value = vina_for_mol(read_sdf(path), scoring_case)
        except Exception as exc:
            value = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        scores[key] = {"sdf": path, **value}
    result = {"case_id": case["case_id"], "identity": identity, "receptor_status": status,
              "receptor": receptor, "box_center": center, "box_size": scoring_case["box_size"],
              "scores": scores}
    write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = read(run / "study.json")
    case_by_id = {c["case_id"]: c for c in plan["cases"]}
    sources = [read(p) for p in sorted((run / "cases").glob("*/sources/*/source.json"))]
    audits, paired, tasks, property_rows = [], [], [], []
    failure_reasons, initial_failures = Counter(), Counter()
    for case in plan["cases"]:
        initial = read(run / "evaluations/initial" / f'{case["case_id"]}.json')["draws"]
        for key, item in initial.items():
            if not item["pb"]["passes"] and not item["pb"].get("missing"):
                initial_failures.update(item["pb"].get("failed_checks", []))
                if not item["repair_eligible"]:
                    failure_reasons[item.get("mask", {}).get("reason", item.get("ineligible_reason", "unknown"))] += 1
        case_sources = [s for s in sources if s["case_id"] == case["case_id"]]
        if not case_sources:
            continue
        repairs = read(run / "evaluations/repairs" / f'{case["case_id"]}.json')["sources"]
        items = {"native": case["native_sdf"]}
        for source in case_sources:
            draw = source["draw"]
            source_key = f"source/{draw:03d}"
            items[source_key] = source["original_sdf"]
            original = read_sdf(source["original_sdf"])
            before = describe(source["original_sdf"])
            property_rows.append({"case_id": case["case_id"], "draw": draw, "arm": "source", "candidate": "", **before})
            for arm, result in repairs[f"{draw:03d}"].items():
                free = source["mask"][f'{"targeted" if arm == "UFF_targeted" else arm}_indices']
                held = [i for i in range(original.GetNumAtoms()) if i not in free]
                selection = result["selected_candidate"]
                for candidate in result["candidates"]:
                    index = candidate["candidate"]
                    key = f"{arm}/{draw:03d}/{index:02d}"
                    path = candidate["sdf"]
                    items[key] = path
                    actual = read_sdf(path)
                    xyz_delta = actual.GetConformer().GetPositions() - original.GetConformer().GetPositions()
                    drift = np.linalg.norm(xyz_delta[held], axis=1) if held else np.array([])
                    free_rmsd = float(np.sqrt(np.mean(np.sum(xyz_delta[free] ** 2, axis=1))))
                    graph = graph_check(original, actual, derive_stereo=True)
                    after = describe(path)
                    deltas = {f"delta_{name}": after[name] - before[name] for name in before if name != "smiles"}
                    audit = {"case_id": case["case_id"], "draw": draw, "arm": arm, "candidate": index,
                             "selected": index == selection, "pb_valid": candidate["pb"]["passes"],
                             "graph_preserved": graph["matches"], "held_atoms": len(held), "free_atoms": len(free),
                             "held_max_drift_angstrom": float(drift.max()) if len(drift) else None,
                             "held_rmsd_angstrom": float(np.sqrt(np.mean(drift ** 2))) if len(drift) else None,
                             "free_rmsd_angstrom": free_rmsd,
                             "original_smiles": before["smiles"], "output_smiles": after["smiles"], **deltas}
                    audits.append(audit)
                    property_rows.append({"case_id": case["case_id"], "draw": draw, "arm": arm, "candidate": index, **after})
                    if index == selection:
                        paired.append({"case_id": case["case_id"], "draw": draw, "arm": arm,
                                       "source_key": source_key, "output_key": key,
                                       "pb_before": False, "pb_after": candidate["pb"]["passes"],
                                       "graph_preserved": graph["matches"],
                                       "held_max_drift_angstrom": audit["held_max_drift_angstrom"],
                                       "free_rmsd_angstrom": free_rmsd,
                                       "resolved_checks": ";".join(result["resolved_checks"]),
                                       "new_failed_checks": ";".join(result["new_failed_checks"]),
                                       "before_failed_checks": ";".join(source["pb_before"]["failed_checks"]),
                                       "after_failed_checks": ";".join(candidate["pb"]["failed_checks"]),
                                       **{f"before_{k}": v for k, v in before.items()},
                                       **{f"after_{k}": v for k, v in after.items()}, **deltas})
        tasks.append((case, items, str(output / "scoring" / case["case_id"])))
    csv_save(output / "preservation_all_candidates.csv", audits)
    csv_save(output / "chemical_properties.csv", property_rows)
    print(json.dumps({"audited_candidates": len(audits), "paired_selected": len(paired),
                      "scoring_pockets": len(tasks), "scoring_ligands": sum(len(x[1]) for x in tasks)}), flush=True)
    scored = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(score_case, task) for task in tasks]):
            result = future.result()
            scored[result["case_id"]] = result
            print(json.dumps({"scored_case": result["case_id"], "receptor_status": result["receptor_status"],
                              "ligand_statuses": dict(Counter(x["status"] for x in result["scores"].values()))}), flush=True)
    all_scores = []
    for cid, result in scored.items():
        for key, value in result["scores"].items():
            all_scores.append({"case_id": cid, "key": key, **value})
    csv_save(output / "vina_all_ligands.csv", all_scores)
    for row in paired:
        scores = scored[row["case_id"]]["scores"]
        before, after = scores[row.pop("source_key")], scores[row.pop("output_key")]
        row.update(vina_before_status=before["status"], vina_after_status=after["status"],
                   vina_before=before.get("score"), vina_after=after.get("score"), delta_vina=None)
        if before["status"] == after["status"] == "ok":
            row["delta_vina"] = after["score"] - before["score"]
    csv_save(output / "paired_selected_metrics.csv", paired)
    summary = {"cases": len(plan["cases"]), "repair_inputs": len(sources), "repair_pockets": len(tasks),
               "candidate_count": len(audits), "failure_selection_reasons": dict(failure_reasons),
               "first_generated_failed_checks": dict(initial_failures),
               "random_equals_targeted": sum(s["mask"]["random_equals_targeted"] for s in sources),
               "receptor_success": sum(r["receptor_status"] == 0 for r in scored.values()),
               "receptor_failed_cases": [cid for cid, r in scored.items() if r["receptor_status"] != 0],
               "arms": {}}
    for arm in plan["arms"]:
        all_arm = [r for r in audits if r["arm"] == arm]
        selected = [r for r in paired if r["arm"] == arm]
        affinity = [r for r in selected if r["delta_vina"] is not None]
        pb_affinity = [r for r in affinity if r["pb_after"]]
        delta = [r["delta_vina"] for r in affinity]
        summary["arms"][arm] = {
            "candidates": len(all_arm), "graph_preserved": sum(r["graph_preserved"] for r in all_arm),
            "held_candidates": sum(r["held_atoms"] > 0 for r in all_arm),
            "held_max_drift_angstrom": max((r["held_max_drift_angstrom"] for r in all_arm if r["held_atoms"]), default=None),
            "selected_graph_preserved": sum(r["graph_preserved"] for r in selected),
            "selected_max_descriptor_delta": {name: max(abs(r[f"delta_{name}"]) for r in selected)
                                               for name in graph_properties(selected[0]["before_smiles"])},
            "pb_recovered": sum(r["pb_after"] for r in selected),
            "new_failures_in_selected": dict(Counter(k for r in selected for k in r["new_failed_checks"].split(";") if k)),
            "remaining_failures_in_selected": dict(Counter(k for r in selected for k in r["after_failed_checks"].split(";") if k)),
            "vina_pairs": len(affinity), "vina_before_median": float(np.median([r["vina_before"] for r in affinity])) if affinity else None,
            "vina_after_median": float(np.median([r["vina_after"] for r in affinity])) if affinity else None,
            "paired_delta_vina_median": float(np.median(delta)) if delta else None,
            "paired_delta_vina_mean": float(np.mean(delta)) if delta else None,
            "vina_improved": sum(x < -1e-6 for x in delta), "vina_worsened": sum(x > 1e-6 for x in delta),
            "vina_unchanged": sum(abs(x) <= 1e-6 for x in delta),
            "pb_valid_vina_pairs": len(pb_affinity),
            "pb_valid_delta_vina_median": float(np.median([r["delta_vina"] for r in pb_affinity])) if pb_affinity else None,
            "pb_valid_and_vina_improved": sum(r["delta_vina"] < -1e-6 for r in pb_affinity)}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
