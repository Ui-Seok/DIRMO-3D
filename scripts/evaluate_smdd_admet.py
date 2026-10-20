#!/usr/bin/env python3
"""CPU ADMET-AI v1.4 evaluation after the complete checkpoint/final split run."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from collections import Counter
from pathlib import Path

import yaml

from dirmo.evaluation.smdd_admet import DATASET_PROPERTIES, conditions, oracle_smiles
from dirmo.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("runs/smdd_lora_02/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if json.loads((args.experiment / "report.json").read_text())["status"] != "complete":
        raise ValueError("Finish checkpoint selection and all split evaluations first")
    datasets = {name: {r["task_id"]: r for r in (json.loads(l) for l in (args.data / f"{name}.jsonl").read_text().splitlines())}
                for name in ["train", "validation", "test"]}
    model_names = ["base", "adapter"]
    last_report = args.experiment / "final/last_adapter/report.json"
    if last_report.exists():
        if json.loads(last_report.read_text())["status"] != "complete":
            raise ValueError("Finish last-checkpoint evaluation first")
        model_names.append("last_adapter")
    records = {}
    witnesses = {}
    molecules = set()
    for name, rows in datasets.items():
        for row in rows.values():
            molecules.add(row["reference_smiles"])
            witness = (Path(row["source"]) / "witness.smi").read_text().split()[0]
            witnesses[row["task_id"]] = witness
            molecules.add(witness)
        for model in model_names:
            path = args.experiment / "final" / model / name / "predictions.jsonl"
            predictions = [json.loads(l) for l in path.read_text().splitlines()]
            if len(predictions) != len(rows) or {r["task_id"] for r in predictions} != set(rows):
                raise ValueError(f"Incomplete or duplicate coverage: {model}/{name}")
            records[(model, name)] = predictions
            for prediction in predictions:
                assessed = prediction["fence_removed_diagnostic"]
                if assessed["smiles_valid"]:
                    molecules.add(oracle_smiles(prediction))
    args.output.mkdir(parents=True)
    import torch
    from admet_ai import ADMETModel
    from admet_ai.constants import DEFAULT_MODELS_DIR
    if importlib.metadata.version("admet_ai") != "1.4.0":
        raise ValueError("SMDD baseline requires ADMET-AI v1.4.0, not v2")
    model = ADMETModel(num_workers=0, drugbank_path=None, fingerprint_multiprocessing_min=10**9)
    if model.device.type != "cpu":
        raise ValueError("Run this oracle in its separate CPU environment")
    source = {"packages": {p: importlib.metadata.version(p) for p in ["admet_ai", "chemprop", "torch", "rdkit", "numpy"]},
              "device": str(model.device), "models_sha256": {str(p.relative_to(DEFAULT_MODELS_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(DEFAULT_MODELS_DIR).rglob("*.pt")}, "unique_molecules": len(molecules),
              "selection": str(args.experiment / "selection.json"), "used_for_checkpoint_selection": False}
    source["oracle_smiles_policy"] = "Original output product_smiles, first SMILES token; no canonicalization. Original witness.smi used for label audit."
    write_json(args.output / "identity.json", source)
    cache = {}
    ordered = sorted(molecules)
    start = time.monotonic()
    for offset in range(0, len(ordered), 100):
        batch = ordered[offset:offset + 100]
        predicted = model.predict(smiles=batch)
        for smiles in batch:
            cache[smiles] = {prop: float(predicted.loc[smiles, dataset]) for dataset, prop in DATASET_PROPERTIES.items()}
        write_json(args.output / "predictions_by_smiles.json", cache)
        print(json.dumps({"event": "admet_progress", "done": len(cache), "rows": len(ordered)}), flush=True)
    audit = []
    pending = []
    summary = {}
    for name, rows in datasets.items():
        for row in rows.values():
            published = yaml.safe_load((Path(row["source"]) / "task.yaml").read_text())["baseline_values"]
            differences = {p: cache[row["reference_smiles"]][p] - published[p] for p in DATASET_PROPERTIES.values()}
            audit.append({"task_id": row["task_id"], "split": name, "reference_deltas_from_published_baseline": differences,
                          "witness_admet_conditions": conditions(row, cache[witnesses[row["task_id"]]])})
        for model_name in model_names:
            output = []
            counts = Counter()
            per_goal = {}
            for record in records[(model_name, name)]:
                row = rows[record["task_id"]]
                assessed = record["fence_removed_diagnostic"]
                entry = {"task_id": row["task_id"], "smiles_valid": assessed["smiles_valid"],
                         "local_hard_constraints_passed": assessed["local_hard_constraints_passed"],
                         "changed": assessed.get("changed", False), "properties": None, "conditions": None,
                         "full_benchmark_success": False if not assessed["local_hard_constraints_passed"] else None}
                if assessed["smiles_valid"]:
                    smiles = oracle_smiles(record)
                    values = cache[smiles]
                    checked = conditions(row, values)
                    entry.update(product_smiles=smiles, graph_canonical_smiles=assessed["product_smiles"], properties=values, conditions=checked)
                    counts["valid_smiles"] += 1
                    if checked["admet_goal_count"] > 0:
                        counts["eligible_admet_goals"] += 1
                        counts["all_admet_goals_passed"] += checked["all_admet_goals_passed"]
                    counts["all_admet_holds_passed"] += checked["all_admet_holds_passed"]
                    for goal in checked["goals"]:
                        if goal["passed"] is not None:
                            bucket = per_goal.setdefault(goal["property"], {"measured": 0, "passed": 0})
                            bucket["measured"] += 1
                            bucket["passed"] += goal["passed"]
                    if not checked["all_admet_goals_passed"] or not checked["all_admet_holds_passed"]:
                        entry["full_benchmark_success"] = False
                    if assessed["local_hard_constraints_passed"] and checked["all_admet_goals_passed"] and checked["all_admet_holds_passed"]:
                        counts["pending_boltz"] += 1
                        pending.append({"model": model_name, "split": name, "task_id": row["task_id"],
                                        "product_smiles": smiles, "source": row["source"],
                                        "changed": entry["changed"]})
                output.append(entry)
            path = args.output / model_name / name
            path.mkdir(parents=True)
            with (path / "conditions.jsonl").open("w") as handle:
                for entry in output:
                    handle.write(json.dumps(entry, allow_nan=False) + "\n")
            summary.setdefault(name, {})[model_name] = {"rows": len(output), "counts": dict(counts), "per_admet_goal": per_goal,
                "full_success_known_false": sum(e["full_benchmark_success"] is False for e in output),
                "full_success_pending_boltz": sum(e["full_benchmark_success"] is None for e in output)}
    write_json(args.output / "reference_and_witness_audit.json", audit)
    write_json(args.output / "pending_boltz.json", pending)
    report = {"status": "complete", "splits": summary, "seconds": time.monotonic() - start,
              "baseline_reference_max_abs_delta": max(abs(v) for r in audit for v in r["reference_deltas_from_published_baseline"].values()),
              "pending_boltz_candidates": len(pending), "limitations": [
                  "ADMET values are model predictions, not experimental measurements.",
                  "Binding affinity objectives/holds and binding probability require Boltz evaluation.",
                  "Published task baselines are used by the official improvement/hold definitions; reference re-predictions are audited separately."]}
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
