"""Frozen-LLM editing with factual tool feedback; no witness or training labels."""
from __future__ import annotations

import json
import math

from rdkit import Chem
from rdkit.Chem import Crippen

from ..data.smdd import CONSTRAINTS, assess_product, json_text, properties
from .smdd_admet import conditions


SYSTEM = """You are a medicinal chemistry editor. Propose one small, chemically valid
edit of the current ligand while preserving similarity to the ORIGINAL ligand.
Goals and hold tolerances are always relative to the published ORIGINAL baseline,
never to a preceding candidate. Input contains actual RDKit calculations and ADMET
model predictions; these are not experiments. Atom IDs are local to each displayed
mapped SMILES. Crippen atom contributions describe a logP calculation, not physical
causality or binding evidence. Pocket distances belong to the reference structure;
they do not locate candidate atoms or prove candidate interactions.
Use a brief edit hypothesis (one sentence), not a long reasoning narrative. Choose
editable atom IDs from current_molecule. Change the molecule to address failed
conditions if feedback is present, keeping already passing conditions. Avoid
repeating failed products. Do not invent measured ADMET or binding values. No fixed
candidate menu is supplied. Binding affinity and probability require Boltz and may
be unmeasured; do not assert they pass.
Return ONLY strict JSON with exactly these keys:
{"focus_atom_maps":[1],"edit_hypothesis":"replace group at atom 1 ...",
 "product_smiles":"complete single-component isomeric SMILES WITHOUT atom maps"}.
focus_atom_maps must contain 1-8 distinct integer IDs in current_molecule. Keep
edit_hypothesis under 400 characters. product_smiles must be a single SMILES token,
with no names, coordinates, Markdown, atom maps, or other text. Preserve specified
stereochemistry unless it is part of your explicit edit hypothesis."""

_GROUPS = {
    "amide": "[NX3][CX3](=[OX1])", "ester": "[CX3](=[OX1])[OX2][#6]",
    "hydroxyl": "[OX2H]", "amine": "[NX3;!$(N[C,S,P]=O)]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])[NX3]",
    "halogen": "[F,Cl,Br,I]", "nitrile": "[CX2]#[NX1]",
}


def molecule_analysis(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Cannot analyze invalid molecule")
    for i, atom in enumerate(mol.GetAtoms(), 1):
        atom.SetAtomMapNum(i)
    # Attribute explicit-H Crippen contributions to their heavy atom. Thus their
    # sum matches MolLogP, instead of omitting implicit-H contributions.
    hydrogenated = Chem.AddHs(mol)
    parts = Crippen._GetAtomContribs(hydrogenated)
    contrib = [p[0] for p in parts[:mol.GetNumAtoms()]]
    for atom in hydrogenated.GetAtoms():
        if atom.GetIdx() >= mol.GetNumAtoms():
            owner = atom.GetNeighbors()[0].GetIdx()
            contrib[owner] += parts[atom.GetIdx()][0]
    values = properties(mol)
    if not math.isclose(sum(contrib), values["logp"], abs_tol=1e-6):
        raise ValueError("Crippen attribution does not sum to MolLogP")
    return {
        "mapped_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "rdkit_properties": values,
        "atom_fields": ["map", "element", "aromatic", "charge", "neighbors", "crippen_logp_with_H"],
        "atoms": [[a.GetAtomMapNum(), a.GetSymbol(), int(a.GetIsAromatic()),
                   a.GetFormalCharge(), [n.GetAtomMapNum() for n in a.GetNeighbors()],
                   round(contrib[a.GetIdx()], 4)] for a in mol.GetAtoms()],
        "functional_groups": {name: [[i + 1 for i in match] for match in
                              mol.GetSubstructMatches(Chem.MolFromSmarts(smarts))]
                              for name, smarts in _GROUPS.items()},
    }


def messages(row, current_smiles, reference_admet, current_admet, round_number, feedback=None):
    # Explicit whitelist: no row['messages'], witness, answer, task ID, or paths.
    content = {
        "proposal_number": round_number,
        "original_molecule": {"smiles": row["reference_smiles"],
                              "rdkit_properties": properties(Chem.MolFromSmiles(row["reference_smiles"])),
                              "admet_reprediction": reference_admet},
        "current_molecule": {**molecule_analysis(current_smiles),
                             "admet_prediction": current_admet},
        "goals": row["goals"], "hold": row["hold"],
        "hard_constraints": CONSTRAINTS, "reference_pocket": row["pocket"],
        "notes": "Published goal/hold baselines stay authoritative even if reference re-prediction differs.",
    }
    if feedback is not None:
        content["tool_feedback"] = feedback
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json_text(content)}]


def assess_plan(raw: str, current_smiles: str, original_smiles: str) -> dict:
    result = {"plan_valid": False, "local_hard_constraints_passed": False,
              "pre_boltz_passed": False, "full_benchmark_success": False}
    try:
        plan = json.loads(raw)
        if not isinstance(plan, dict) or set(plan) != {"focus_atom_maps", "edit_hypothesis", "product_smiles"}:
            raise ValueError("Expected exactly focus_atom_maps, edit_hypothesis, product_smiles")
        maps = plan["focus_atom_maps"]
        count = Chem.MolFromSmiles(current_smiles).GetNumAtoms()
        if (not isinstance(maps, list) or not 1 <= len(maps) <= 8 or
                any(type(i) is not int or not 1 <= i <= count for i in maps) or len(set(maps)) != len(maps)):
            raise ValueError("focus_atom_maps must be 1-8 distinct current molecule atom IDs")
        if not isinstance(plan["edit_hypothesis"], str) or not 1 <= len(plan["edit_hypothesis"].strip()) <= 400:
            raise ValueError("edit_hypothesis must be a nonempty sentence under 400 characters")
        smiles = plan["product_smiles"]
        if not isinstance(smiles, str) or not smiles or smiles != smiles.strip() or len(smiles.split()) != 1:
            raise ValueError("product_smiles must be exactly one SMILES token")
        result.update(plan_valid=True, plan=plan, oracle_smiles=smiles,
                      edit_location_tool_verified=False)
        # Canonicalization is for graph checks only. Preserve original output for
        # Chemprop/ADMET inference: equivalent serializations can differ there.
        result.update(assess_product(json_text({"product_smiles": smiles}), original_smiles))
    except (ValueError, TypeError, RuntimeError) as exc:
        result["error"] = str(exc)
    return result


def attach_admet(assessment, row, values):
    checked = conditions(row, values)
    assessment.update(admet_properties=values, admet_conditions=checked)
    passed = (assessment["local_hard_constraints_passed"] and assessment.get("changed", False)
              and checked["all_admet_goals_passed"] and checked["all_admet_holds_passed"])
    assessment.update(pre_boltz_passed=passed, full_benchmark_success=None if passed else False)
    return assessment


def tool_feedback(assessment, tried_smiles):
    failures = []
    if not assessment["plan_valid"]:
        failures.append({"stage": "output_format", "reason": assessment.get("error", "Invalid JSON plan")})
    elif not assessment.get("smiles_valid"):
        failures.append({"stage": "rdkit_parse", "reason": "Invalid molecular graph or SMILES"})
    else:
        failures.extend({"stage": "rdkit", "condition": name, "passed": False}
                        for name, passed in assessment.get("checks", {}).items() if not passed)
        if not assessment.get("changed"):
            failures.append({"stage": "edit", "condition": "Must change the original ligand"})
        if "admet_conditions" in assessment:
            for kind in ["goals", "holds"]:
                failures.extend({"stage": "admet", "kind": kind, **item}
                                for item in assessment["admet_conditions"][kind] if item["passed"] is False)
    return {
        "failed_conditions": failures,
        "last_product_smiles": assessment.get("oracle_smiles"),
        "last_rdkit_properties": assessment.get("metrics"),
        "last_tanimoto_to_original": assessment.get("tanimoto"),
        "admet_checks": assessment.get("admet_conditions"),
        "pre_boltz_passed": assessment.get("pre_boltz_passed", False),
        "binding_status": "unmeasured; passing RDKit/ADMET is not full success",
        "previous_products": tried_smiles,
        "instruction": "Repair failed conditions with a small edit; if none failed, propose another plausible candidate.",
    }


def next_state(original, assessment):
    # A syntactically valid but disconnected, mapped or radical product must not
    # become the next editing substrate. Original similarity remains the anchor.
    safe = assessment.get("smiles_valid") and all(assessment.get("checks", {}).get(k, False)
            for k in ["single_component", "no_radicals", "no_atom_maps"])
    return assessment["oracle_smiles"] if safe else original


def select_attempt(attempts):
    """Lexicographic diagnostics, not an invented scalar utility or witness rank."""
    def rank(item):
        a = item["assessment"]
        admet = a.get("admet_conditions", {})
        failed = sum(v["passed"] is False for group in ["goals", "holds"] for v in admet.get(group, []))
        return (a.get("full_benchmark_success") is True, a.get("pre_boltz_passed", False),
                a.get("local_hard_constraints_passed", False), a.get("smiles_valid", False),
                a.get("changed", False), -failed, -item["round"])
    return max(attempts, key=rank) if attempts else None
