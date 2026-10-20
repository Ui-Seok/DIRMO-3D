"""SMDD ADMET conditions, keeping unmeasured Boltz conditions explicit."""
import math
import json
import re


# Official SMDD-Bench metrics/lead_optimization.py property names.
DATASET_PROPERTIES = {"hERG": "herg", "BBB_Martins": "bbb", "Solubility_AqSolDB": "solubility",
                      "CYP3A4_Veith": "cyp3a4", "Clearance_Hepatocyte_AZ": "clearance",
                      "Caco2_Wang": "caco2", "PPBR_AZ": "ppb", "AMES": "ames"}


def oracle_smiles(record):
    """Preserve the actual output string, as SMDD's solution.smi reader does.

    Legacy Chemprop features include local ChiralTag, so chemically equivalent
    SMILES serializations can yield different predictions. Do not silently
    substitute a graph-canonicalized string before this oracle.
    """
    if not record["fence_removed_diagnostic"]["smiles_valid"]:
        return None
    raw = record["raw"]
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", raw, flags=re.DOTALL)
    obj = json.loads(match.group(1).strip() if match else raw)
    return obj["product_smiles"].strip().split()[0].replace('"', '').replace("'", "")


def conditions(row, values):
    goals, holds = [], []
    for goal in row["goals"]:
        item = {**goal, "value": values.get(goal["property"]), "passed": None}
        if goal["property"] != "binding_affinity" and item["value"] is not None and math.isfinite(item["value"]):
            improvement = (goal["baseline"] - item["value"]) if goal["direction"] == "minimize" else (item["value"] - goal["baseline"])
            item.update(improvement=improvement, passed=improvement >= goal["threshold"])
        goals.append(item)
    for hold in row["hold"]:
        item = {**hold, "value": values.get(hold["property"]), "passed": None}
        if hold["property"] != "binding_affinity" and item["value"] is not None and math.isfinite(item["value"]):
            delta = item["value"] - hold["baseline"]
            bad_change = delta if hold["direction"] == "increase" else -delta if hold["direction"] == "decrease" else abs(delta)
            item.update(delta=delta, passed=bad_change <= hold["tolerance"])
        holds.append(item)
    admet_goals = [g for g in goals if g["property"] != "binding_affinity"]
    admet_holds = [h for h in holds if h["property"] != "binding_affinity"]
    return {"goals": goals, "holds": holds, "admet_goal_count": len(admet_goals),
            "all_admet_goals_passed": all(g["passed"] is True for g in admet_goals),
            "all_admet_holds_passed": all(h["passed"] is True for h in admet_holds),
            "binding_affinity_measured": False, "boltz_binding_probability": None}
