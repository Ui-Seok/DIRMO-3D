"""SMDD-Bench task-4 pairs, with grouped splits and reference-only prompts.

This is a separate instruction-tuning pilot. It does not change the DIRMO planner
response schema, and does not substitute RDKit checks for the benchmark oracles.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from Bio.SeqUtils import seq1
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Contrib.SA_Score import sascorer


SYSTEM_PROMPT = (
    "You are a molecular medicinal chemistry editor. Given a reference ligand, "
    "its predicted properties, optimization goals, hold conditions, and pocket context, "
    "propose one closely related molecule. Goals specify minimum changes from baseline. "
    "Hold directions specify the change that must not exceed tolerance: increase, "
    "decrease, or absolute change. Pocket distances are measured reference contacts, "
    "not guaranteed hydrogen bonds. Return only a JSON object with exactly one key "
    '"product_smiles": a valid single-component isomeric SMILES without atom maps. '
    "Do not include explanations or claim unmeasured properties."
)

CONSTRAINTS = {
    "mw_lt": 600, "logp_range": [-1, 5], "tpsa_lt": 140,
    "hbd_le": 5, "hba_le": 10, "rotatable_bonds_le": 10,
    "formal_charge_range": [-2, 2], "sa_lt": 4.5,
    "no_radicals": True, "single_component": True,
    "filters": ["PAINS", "BRENK"], "morgan_r2_2048_tanimoto_ge": 0.7,
    "boltz_binding_probability_gt": 0.7,
}


def json_text(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def properties(mol) -> dict:
    return {"qed": QED.qed(mol), "sa": sascorer.calculateScore(mol),
            "mw": Descriptors.MolWt(mol), "logp": Crippen.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol), "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "formal_charge": Chem.GetFormalCharge(mol)}


_FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_FILTERS = {}


def assess_product(raw: str, reference: str, witness: str | None = None) -> dict:
    """Strict output validation and deterministic subset of task-4 constraints."""
    result = {"json_valid": False, "smiles_valid": False,
              "local_hard_constraints_passed": False,
              "oracle_goal_success": None, "oracle_hold_success": None,
              "boltz_binding_probability": None}
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or set(obj) != {"product_smiles"} or not isinstance(obj["product_smiles"], str):
            return result
        result["json_valid"] = True
        product = Chem.MolFromSmiles(obj["product_smiles"])
        if product is None or product.GetNumAtoms() == 0:
            return result
        result["smiles_valid"] = True
        ref = Chem.MolFromSmiles(reference)
        if ref is None:
            raise ValueError("Invalid reference")
        values = properties(product)
        similarity = DataStructs.TanimotoSimilarity(_FP.GetFingerprint(ref), _FP.GetFingerprint(product))
        for name in ["PAINS", "BRENK"]:
            if name not in _FILTERS:
                params = FilterCatalog.FilterCatalogParams()
                params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, name))
                _FILTERS[name] = FilterCatalog.FilterCatalog(params)
        checks = {"single_component": len(Chem.GetMolFrags(product)) == 1,
                  "no_radicals": not any(a.GetNumRadicalElectrons() for a in product.GetAtoms()),
                  "no_atom_maps": not any(a.GetAtomMapNum() for a in product.GetAtoms()),
                  "mw": values["mw"] < 600, "logp": -1 <= values["logp"] <= 5,
                  "tpsa": values["tpsa"] < 140, "hbd": values["hbd"] <= 5,
                  "hba": values["hba"] <= 10, "rotatable_bonds": values["rotatable_bonds"] <= 10,
                  "formal_charge": -2 <= values["formal_charge"] <= 2,
                  "sa": values["sa"] < 4.5, "tanimoto": similarity >= 0.7,
                  **{name.lower(): not _FILTERS[name].HasMatch(product) for name in _FILTERS}}
        product_smiles = canonical(obj["product_smiles"])
        result.update(product_smiles=product_smiles, metrics=values, checks=checks,
                      tanimoto=similarity, changed=product_smiles != canonical(reference),
                      local_hard_constraints_passed=all(checks.values()))
        if witness is not None:
            result["witness_exact_match"] = product_smiles == canonical(witness)
    except (ValueError, TypeError, RuntimeError) as exc:
        result["error"] = str(exc)
    return result


def read_fasta(path: Path) -> dict[str, str]:
    records = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            records.append([line, ""])
        elif line.strip():
            if not records:
                raise ValueError(f"Missing FASTA header: {path}")
            records[-1][1] += line.strip().upper()
    chains = {}
    for header, sequence in records:
        match = re.search(r"\|Chains? ([^|]+)\|", header)
        if match is None:
            raise ValueError(f"Missing chain label: {header}")
        for chain in match.group(1).split(","):
            # PDB FASTA can include author-chain annotations after the label.
            label = chain.strip().split()[0]
            if label in chains and chains[label] != sequence:
                raise ValueError(f"Conflicting FASTA chain {label}: {path}")
            chains[label] = sequence
    return chains


def pocket_context(folder: Path, task: dict, sequences: dict[str, str]) -> dict:
    """SEQRES positions are never silently treated as arbitrary PDB residue IDs.

    Boltz files normally use sequence numbering. Geometry is included only when
    the PDB residue identity at that number agrees with the FASTA identity.
    """
    atoms = defaultdict(list)
    names = {}
    ligand = []
    target = task["target_ligand"]
    for line in (folder / "complex.pdb").read_text().splitlines():
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        if line[76:78].strip() == "H":
            continue
        key = (line[21], int(line[22:26]))
        xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        if key == (target["chain"], target["residue_number"]) and line[17:20].strip() == target["resname"]:
            ligand.append(xyz)
        elif line.startswith("ATOM"):
            atoms[key].append(xyz)
            names[key] = seq1(line[17:20].strip())
    residues = []
    unavailable = 0
    for chain, position in task["pocket_residues"]:
        sequence = sequences.get(chain, "")
        if not 1 <= position <= len(sequence):
            raise ValueError(f"Unsupported SEQRES mapping: {folder.name} {chain}:{position}")
        aa = sequence[position - 1]
        distance = None
        if ligand and (chain, position) in atoms and names[(chain, position)] == aa:
            delta = np.asarray(atoms[(chain, position)])[:, None, :] - np.asarray(ligand)[None, :, :]
            distance = round(float(np.linalg.norm(delta, axis=-1).min()), 1)
        else:
            unavailable += 1
        residues.append([chain, position, aa, distance])
    return {"residue_fields": ["chain", "seqres_position", "aa", "reference_min_heavy_distance_A"],
            "residues": residues, "source": "reference complex.pdb",
            "unavailable_distances": unavailable}


def make_messages(row: dict) -> list[dict]:
    # Whitelist input fields. Witness, witness metrics, paths and task IDs cannot
    # enter the prompt even if the stored row later receives extra annotations.
    content = {"reference_mapped_smiles": row["reference_mapped_smiles"],
               "goals": row["goals"], "hold": row["hold"],
               "hard_constraints": CONSTRAINTS, "pocket": row["pocket"]}
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json_text(content)}]


def load_pairs(root: Path) -> list[dict]:
    rows = []
    for folder in sorted(root.glob("smdd_004_*")):
        task = yaml.safe_load((folder / "task.yaml").read_text())
        mols = [Chem.SDMolSupplier(str(folder / name), removeHs=True)[0]
                for name in ["reference.sdf", "witness.sdf"]]
        if any(m is None for m in mols):
            raise ValueError(f"Unparseable pair: {folder}")
        reference = canonical(Chem.MolToSmiles(mols[0]))
        sdf_witness = canonical(Chem.MolToSmiles(mols[1]))
        # Predicted 3D SDFs can introduce stereocenters absent from the published
        # chemical solution. Use the official solution SMILES as the label.
        witness = canonical((folder / "witness.smi").read_text().split()[0])
        published = Chem.MolFromSmiles(witness)
        if Chem.MolToSmiles(published, isomericSmiles=False) != Chem.MolToSmiles(Chem.MolFromSmiles(sdf_witness), isomericSmiles=False):
            raise ValueError(f"Witness SMILES/SDF connectivity differs: {folder.name}")
        mol = Chem.MolFromSmiles(reference)
        for i, atom in enumerate(mol.GetAtoms(), start=1):
            atom.SetAtomMapNum(i)
        sequences = read_fasta(folder / "protein.fasta")
        target_sequences = sorted({sequences[chain] for chain, _ in task["pocket_residues"]})
        baseline = task["baseline_values"]
        row = {"task_id": task["id"], "source": str(folder.resolve()),
               "protein_key": hashlib.sha256(json_text(target_sequences).encode()).hexdigest(),
               "reference_smiles": reference, "witness_smiles": witness,
               "witness_label_source": "witness.smi", "witness_sdf_smiles": sdf_witness,
               "reference_mapped_smiles": Chem.MolToSmiles(mol),
               "goals": [{**g, "baseline": baseline[g["property"]]} for g in task["optimization_objectives"]],
               "hold": [{**g, "baseline": baseline[g["property"]]} for g in task["hold_constant"]],
               "pocket": pocket_context(folder, task, sequences),
               "reference_metrics": properties(mols[0]), "witness_metrics": properties(published),
               "answer": json_text({"product_smiles": witness})}
        row["messages"] = make_messages(row)
        rows.append(row)
    if not rows:
        raise ValueError(f"No task-4 pairs under {root}")
    return rows


def leakage_keys(row: dict) -> set[tuple[str, str]]:
    keys = {("protein", row["protein_key"])}
    for name in ["reference_smiles", "witness_smiles"]:
        smiles = canonical(row[name])
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        keys.add(("molecule", smiles))
        if scaffold:  # Empty scaffolds would merge every acyclic molecule.
            keys.add(("scaffold", scaffold))
    return keys


def split_pairs(rows: list[dict], seed: int = 42) -> tuple[dict, dict]:
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in sorted(leakage_keys(row)):
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(rows[i])
    components = list(groups.values())
    if len(components) < 3:
        raise ValueError("At least three independent protein/molecule/scaffold components required")
    random.Random(seed).shuffle(components)

    def take_subset(available, target):
        # Closest attainable whole-group size, with deterministic seeded tie order.
        attainable = {0: []}
        for i, group in enumerate(available):
            for count, indices in list(attainable.items()):
                attainable.setdefault(count + len(group), indices + [i])
        candidates = [n for n, ids in attainable.items() if ids and len(ids) <= len(available) - 2]
        if not candidates:
            raise ValueError("Cannot create nonempty independent held-out splits")
        count = min(candidates, key=lambda n: (abs(n - target), n))
        chosen = set(attainable[count])
        return [g for i, g in enumerate(available) if i in chosen], [g for i, g in enumerate(available) if i not in chosen]

    validation, remaining = take_subset(components, max(1, round(len(rows) * 0.1)))
    # Leave at least one training group after choosing the second holdout.
    if len(remaining) == 2:
        test, train = remaining[:1], remaining[1:]
    else:
        test, train = take_subset(remaining, max(1, round(len(rows) * 0.1)))
    splits = {name: sorted([r for g in gs for r in g], key=lambda r: r["task_id"])
              for name, gs in [("train", train), ("validation", validation), ("test", test)]}
    sets = {name: set().union(*(leakage_keys(r) for r in values)) for name, values in splits.items()}
    overlaps = {}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlaps[f"{a}:{b}"] = len(sets[a] & sets[b])
    if any(overlaps.values()):
        raise AssertionError(f"Split leakage: {overlaps}")
    report = {"seed": seed, "rows": len(rows), "connected_groups": len(components),
              "group_sizes": sorted(map(len, components)), "overlaps": overlaps,
              "split_rule": "connected components sharing protein sequence, either molecule, or nonempty Bemis-Murcko scaffold",
              "splits": {name: {"rows": len(values), "proteins": len({r["protein_key"] for r in values}),
                                 "goals": dict(Counter(g["property"] for r in values for g in r["goals"]))}
                         for name, values in splits.items()}}
    return splits, report


def prepare(root: Path, output: Path, seed: int = 42) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    rows = load_pairs(root)
    splits, report = split_pairs(rows, seed)
    assessments = [assess_product(r["answer"], r["reference_smiles"]) for r in rows]
    report["witness_local_checks"] = {
        "passed": sum(r["local_hard_constraints_passed"] for r in assessments),
        "failed_checks": dict(Counter(k for a in assessments for k, passed in a.get("checks", {}).items() if not passed))}
    report["missing_pocket_distances"] = sum(r["pocket"]["unavailable_distances"] for r in rows)
    report["witness_sdf_stereochemistry_differences"] = sum(r["witness_sdf_smiles"] != r["witness_smiles"] for r in rows)
    report["witness_label_source"] = "Official witness.smi; SDF used only to cross-check connectivity"
    report["limitations"] = [
        "Witnesses are benchmark-provided solutions, not necessarily unique or experimental optima.",
        "No witness property or future outcome appears in prompts.",
        "RDKit checks omit ADMET-AI/Boltz objectives, holds, and binding probability.",
        "This split is for tuning; the full public benchmark is no longer an untouched test set.",
    ]
    output.mkdir(parents=True)
    for name, values in splits.items():
        (output / f"{name}.jsonl").write_text("".join(json_text(r) + "\n" for r in values))
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
