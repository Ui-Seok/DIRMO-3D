"""SMDD task-4 co-fold inputs and checks; preserve unrounded affinity values."""
import io
import math
import re

from Bio import SeqIO
from rdkit import Chem


def evaluation_config(task):
    entry = next(e for e in task["evaluation"] if e["function"] == "assess_boltz_task4")
    return {v["arg"]: v["value"] for v in entry["inputs"] if v["source"] == "const"}


def cofold_input(fasta, smiles, config):
    proteins = []
    for record in SeqIO.parse(io.StringIO(fasta), "fasta"):
        match = re.search(r"\|Chains? ([^|]+)\|", record.description)
        chains = match.group(1).split(",") if match else [record.id]
        for chain in chains:
            auth = re.search(r"\[auth\s+(\S+)\]", chain)
            proteins.append({"protein": {"id": auth.group(1) if auth else chain.strip(),
                                          "sequence": str(record.seq)}})
    if not proteins:
        raise ValueError("No protein sequences in FASTA")
    chains = {p["protein"]["id"]: len(p["protein"]["sequence"]) for p in proteins}
    for chain, position in config["pocket_contacts"]:
        if chain not in chains or not 1 <= position <= chains[chain]:
            raise ValueError("Pocket position does not match FASTA chains")
    sequences = proteins + [{"ligand": {"id": "LIG", "smiles": smiles}}]
    seen = {Chem.CanonSmiles(smiles)}
    for ligand in config.get("co_ligands") or []:
        canonical = Chem.CanonSmiles(ligand["smiles"])
        if canonical not in seen:
            seen.add(canonical)
            sequences.append({"ligand": ligand})
    return {"version": 1, "sequences": sequences,
            "constraints": [{"pocket": {"binder": "LIG", "contacts": config["pocket_contacts"],
                                         "max_distance": 6.0, "force": True}}],
            "properties": [{"affinity": {"binder": "LIG"}}]}


def mean_affinity(result, key):
    # SMDD averages numeric per-model suffixes, falling back to the ensemble field.
    values = [float(value) for name, value in result.items()
              if name.startswith(key) and name[len(key):].isdigit()]
    if not values:
        values = [float(result[key])]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite affinity prediction")
    return sum(values) / len(values)


def binding_conditions(result, config):
    probability = mean_affinity(result, "affinity_probability_binary")
    affinity = mean_affinity(result, "affinity_pred_value")
    baseline = config["baseline_affinity"]
    objective, hold = config.get("affinity_objective"), config.get("affinity_hold_constant")
    improvement = baseline - affinity
    goal_passed = objective is None or improvement >= objective["threshold"]
    hold_passed = hold is None or affinity - baseline <= hold["tolerance"]
    return {"binding_probability": probability, "affinity_pred_value": affinity,
            "baseline_affinity": baseline, "affinity_improvement": improvement,
            "binding_passed": probability > 0.7, "affinity_goal_passed": goal_passed,
            "affinity_hold_passed": hold_passed,
            "all_passed": probability > 0.7 and goal_passed and hold_passed}
