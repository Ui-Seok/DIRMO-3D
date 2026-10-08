from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from rdkit import Chem

from ..chemistry.graphs import graph_id, map_indices, protected_graph, smiles
from ..chemistry.properties import graph_properties, pharmacophore
from ..config import EvaluationConfig
from ..io import digest, version
from ..schemas import Case, EvaluationRecord, Observation
from .pocket import protein_atoms


def geometry(mol, pocket_path: str, tolerance: float) -> dict:
    pos = mol.GetConformer().GetPositions()
    pocket, elements, _ = protein_atoms(pocket_path)
    table = Chem.GetPeriodicTable()
    lr = np.array([table.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])
    pr = np.array([table.GetRvdw(table.GetAtomicNumber(e)) for e in elements])
    d = np.linalg.norm(pos[:, None] - pocket[None, :], axis=-1)
    margin = d - tolerance * (lr[:, None] + pr[None, :])
    bad_bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = np.linalg.norm(pos[i] - pos[j])
        covalent_sum = sum(table.GetRcovalent(mol.GetAtomWithIdx(k).GetAtomicNum()) for k in (i, j))
        if not 0.55 * covalent_sum <= length <= 1.35 * covalent_sum:
            bad_bonds.append({"atoms": [i, j], "length": float(length)})
    pair_dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    topology = Chem.GetDistanceMatrix(mol)
    internal_margin = pair_dist - 0.6 * (lr[:, None] + lr[None, :])
    nonbonded = np.triu(topology > 2, 1)
    internal_clashes = int(((internal_margin < 0) & nonbonded).sum())
    return {"passes": not bad_bonds and internal_clashes == 0 and bool(np.all(margin >= 0)),
            "ligand_bond_geometry_pass": not bad_bonds, "internal_clash_count": internal_clashes,
            "protein_overlap_count": int((margin < 0).sum()), "bad_bonds": bad_bonds,
            "minimum_contact_distance_angstrom": float(d.min()),
            "minimum_contact_margin_angstrom": float(margin.min()),
            "clash_severity_angstrom": float(max(0, -margin.min())),
            "definition": "DIRMO covalent-radius bond screen and scaled VDW distance screen; not PoseBusters"}


def posebusters(mol, pocket: str) -> dict:
    from posebusters import PoseBusters
    checker = PoseBusters(config="dock")
    report = checker.bust(mol_pred=mol, mol_true=None, mol_cond=Path(pocket), full_report=True)
    row = report.iloc[0]
    # PoseBusters renames outputs and then lowercases/spaces-to-underscores the columns.
    # Modules share raw names (no_clashes, flatness_passes), so raw-name lookup is wrong.
    keys = []
    for module in checker.config["modules"]:
        for name in module.get("chosen_binary_test_output", []):
            renamed = module.get("rename_outputs", {}).get(name, name + module.get("rename_suffix", ""))
            keys.append(renamed.lower().replace(" ", "_"))
    import pandas as pd
    checks = {key: bool(row[key]) if key in row and not pd.isna(row[key]) else None for key in keys}
    continuous = {str(k): (float(v) if np.isfinite(v) else None) for k, v in row.items()
                  if isinstance(v, (float, np.floating)) and k not in checks}
    return {"passes": bool(checks) and all(v is True for v in checks.values()), "checks": checks,
            "continuous": continuous,
            "missing_reasons": {key: "Upstream returned missing check" for key, value in checks.items() if value is None},
            "preset": "dock", "reference_ligand": None, "reference_protein": pocket,
            "omitted_checks": {"identity": "edited ligand is not a reference identity task",
                               "rmsd": "generated pose is evaluated in the fixed protein frame"}}


def vina_score(mol, case: Case) -> dict:
    if not case.receptor_path or not Path(case.receptor_path).is_file():
        raise FileNotFoundError("Prepared receptor PDBQT is required for Vina score-only")
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina import Vina
    ligand = Chem.AddHs(mol, addCoords=True)
    setups = MoleculePreparation().prepare(ligand)
    if len(setups) != 1:
        raise ValueError("Meeko returned multiple preparations; choose protonation explicitly")
    pdbqt, success, reason = PDBQTWriterLegacy.write_string(setups[0])
    if not success:
        raise ValueError(f"Meeko PDBQT preparation failed: {reason}")
    scorer = Vina(sf_name="vina", cpu=1, seed=0, verbosity=0)
    scorer.set_receptor(case.receptor_path)
    scorer.set_ligand_from_string(pdbqt)
    scorer.compute_vina_maps(center=list(case.box_center), box_size=list(case.box_size))
    values = scorer.score()
    return {"vina": float(values[0]), "components": [float(v) for v in values], "mode": "score_only",
            "box_center": case.box_center, "box_size": case.box_size, "receptor": case.receptor_path,
            "ligand_preparation": "RDKit added H coordinates; Meeko Gasteiger; no docking/minimization"}


def interactions(mol, protein_path: str) -> dict:
    import MDAnalysis as mda
    import prolif as plf
    protein_universe = mda.Universe(protein_path)
    # ProLIF donor/acceptor roles require explicit H. Do not invent protonation.
    if not any(a.element == "H" for a in protein_universe.atoms):
        raise ValueError("ProLIF requires a separately prepared protein with explicit hydrogens")
    protein = plf.Molecule.from_mda(protein_universe)
    ligand = plf.Molecule.from_rdkit(Chem.AddHs(mol, addCoords=True))
    fingerprint = plf.Fingerprint(count=True)
    data = fingerprint.generate(ligand, protein, metadata=True)
    records = []
    for (_, residue), roles in data.items():
        for role, occurrences in roles.items():
            records.append({"residue_id": f"{residue.chain}:{residue.name}:{residue.number}", "role": role,
                            "count": len(occurrences)})
    return {"interactions": records, "source": "ProLIF", "definition": "ProLIF interaction roles"}


def preservation(original, mol, case: Case, interaction_record: EvaluationRecord) -> dict:
    protected_graph(original, mol, case.protected)
    indices = map_indices(mol)
    reference = map_indices(original)
    features = pharmacophore(mol)
    results = []
    for constraint in case.protected.features:
        found = [f for f in features if f["family"] == constraint.family and
                 set(f["atom_maps"]) == set(constraint.atom_maps)]
        distance = min((float(np.linalg.norm(np.array(f["position"]) - constraint.reference_position))
                        for f in found), default=None)
        results.append({"kind": "pharmacophore", "family": constraint.family, "atom_maps": constraint.atom_maps,
                        "hard": constraint.hard, "distance_angstrom": distance,
                        "passes": distance is not None and distance <= constraint.max_distance})
    if case.protected.max_atom_displacement is not None:
        for atom_map in case.protected.atom_maps:
            delta = np.linalg.norm(mol.GetConformer().GetPositions()[indices[atom_map]] -
                                   original.GetConformer().GetPositions()[reference[atom_map]])
            results.append({"kind": "pocket_relative_displacement", "atom_map": atom_map, "hard": True,
                            "distance_angstrom": float(delta),
                            "passes": bool(delta <= case.protected.max_atom_displacement)})
    measured = interaction_record.values.get("interactions", [])
    for constraint in case.protected.interactions:
        results.append({"kind": "interaction", "residue_id": constraint.residue_id, "role": constraint.role,
                        "hard": constraint.hard, "passes": interaction_record.status == "ok" and any(
                            r["residue_id"] == constraint.residue_id and r["role"] == constraint.role
                            for r in measured)})
    return {"hard_pass": all(r["passes"] for r in results if r["hard"]), "checks": results,
            "features": features, "frame": "fixed protein; no ligand alignment"}


def measure(fn, package: str | None = None) -> EvaluationRecord:
    start = time.monotonic()
    try:
        values = fn()
        return EvaluationRecord(status="ok", values=values, version=version(package) if package else None,
                                latency_seconds=time.monotonic() - start)
    except (ImportError, FileNotFoundError) as exc:
        return EvaluationRecord(status="missing", reason=str(exc), version=version(package) if package else None,
                                latency_seconds=time.monotonic() - start)
    except Exception as exc:
        return EvaluationRecord(status="failed", reason=f"{type(exc).__name__}: {exc}",
                                version=version(package) if package else None,
                                latency_seconds=time.monotonic() - start)


class Evaluator:
    def __init__(self, config: EvaluationConfig, case: Case, fixed_pocket: str, original):
        self.config, self.case, self.pocket, self.original = config, case, fixed_pocket, original
        self.baseline: Observation | None = None

    def evaluate(self, mol, matches: bool = True) -> Observation:
        cfg, case = self.config, self.case
        records = {"properties": measure(lambda: {**graph_properties(smiles(mol)),
                         "sa_convention": "RDKit contrib SA_Score; lower is better"}, "rdkit"),
                   "geometry": measure(lambda: geometry(mol, self.pocket, cfg.clash_tolerance)),
                   "posebusters": measure(lambda: posebusters(mol, self.pocket), "posebusters") if cfg.posebusters
                         else EvaluationRecord(status="disabled", reason="Explicit config disable"),
                   "vina": measure(lambda: vina_score(mol, case), "vina") if cfg.vina else
                         EvaluationRecord(status="disabled", reason="Explicit config disable"),
                   "interactions": measure(lambda: interactions(mol, case.protein_path or self.pocket), "prolif")
                         if cfg.interactions else EvaluationRecord(status="disabled", reason="Explicit config disable")}
        records["preservation"] = measure(lambda: preservation(self.original, mol, case, records["interactions"]))
        metrics = {k: float(v) for name in ("properties", "vina") if records[name].status == "ok"
                   for k, v in records[name].values.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        missing = [name for name in case.objective.required_evaluations if records[name].status != "ok"]
        failures = [f"{name}:{records[name].status}" for name in missing]
        for name in ("geometry", "posebusters"):
            if records[name].status == "ok" and not records[name].values.get("passes", False):
                failures.append(f"{name}:invalid")
        hard_pass = records["preservation"].status == "ok" and records["preservation"].values.get("hard_pass", False)
        if not hard_pass:
            failures.append("protected_constraint_failed")
        if not matches:
            failures.append("graph_mismatch")
        metric_complete = all(k in metrics for k in case.objective.metrics)
        if not metric_complete:
            failures.append("objective_metric_missing")
        targets = metric_complete and all((metrics[k] >= g.target if g.direction == "maximize" else
                      metrics[k] <= g.target) for k, g in case.objective.metrics.items())
        utility = sum((1 if g.direction == "maximize" else -1) * g.weight * metrics[k]
                      for k, g in case.objective.metrics.items()) if metric_complete else None
        if self.baseline:
            for k, goal in case.objective.metrics.items():
                if goal.max_loss is not None:
                    if k not in metrics or k not in self.baseline.metrics:
                        failures.append(f"baseline_metric_missing:{k}")
                    else:
                        loss = (self.baseline.metrics[k] - metrics[k]) * (
                            1 if goal.direction == "maximize" else -1)
                        if loss > goal.max_loss:
                            failures.append(f"maximum_loss_exceeded:{k}")
        payload = {"graph_id": graph_id(mol), "metrics": metrics, "pose": mol.GetConformer().GetPositions().tolist()}
        return Observation(observation_id=digest(payload), graph_id=graph_id(mol), evaluations=records,
                           metrics=metrics, graph_matches=matches, hard_constraints_pass=hard_pass,
                           feasible=not failures, target_met=bool(targets), utility=utility, failures=failures,
                           incomplete_evaluation=bool(missing or not metric_complete or
                               any(records[n].status == "disabled" for n in ("posebusters", "vina"))))
