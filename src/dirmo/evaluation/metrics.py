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


HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "PHE", "MET", "TRP", "PRO", "GLY", "CYS"}
CHARGED = {"ASP", "GLU", "LYS", "ARG"}
POLAR_ELEMENTS = {"N", "O"}
# A new heavy atom sits roughly a bond length out; scan beyond it to size the opening.
GROWTH_BOND = 1.5
GROWTH_SCAN = np.arange(1.5, 4.01, 0.25)
# A cut pocket ends about 10 A out, so a ray aimed at the truncation hits nothing and reads
# as infinite room. Free space only counts where the probe is still enclosed by protein.
BURIAL_RADIUS = 8.0
BURIAL_MINIMUM = 12


def _directions(count: int = 32) -> np.ndarray:
    """Roughly even directions on the sphere (Fibonacci), for probing free space."""
    index = np.arange(count) + 0.5
    phi = np.arccos(1 - 2 * index / count)
    theta = np.pi * (1 + 5 ** 0.5) * index
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def opportunity(pos, pocket, ligand_radii, pocket_radii, tolerance, residues, maps):
    """Per-atom free space: where could a substituent grow, and what faces it?

    The clash screen already computes every ligand-to-pocket distance and then keeps only
    the worst one. The same matrix says which directions are EMPTY, which is what a plan
    needs: an interaction fingerprint reports what the ligand already does, not what the
    pocket is offering that it does not use yet. Nothing here is predicted; it is geometry.
    """
    table = Chem.GetPeriodicTable()
    probe = table.GetRvdw(6)                       # a new carbon
    rays = _directions()
    out = []
    for i in range(len(pos)):
        candidates = pos[i] + GROWTH_SCAN[:, None, None] * rays[None, :, :]   # (R, D, 3)
        flat = candidates.reshape(-1, 3)
        to_pocket = np.linalg.norm(flat[:, None] - pocket[None, :], axis=-1)
        clear_pocket = to_pocket >= tolerance * (probe + pocket_radii[None, :])
        others = np.delete(np.arange(len(pos)), i)
        to_ligand = np.linalg.norm(flat[:, None] - pos[others][None, :], axis=-1)
        clear_ligand = to_ligand >= 0.6 * (probe + ligand_radii[others][None, :])
        enclosed = (to_pocket <= BURIAL_RADIUS).sum(1) >= BURIAL_MINIMUM
        clear = (clear_pocket.all(1) & clear_ligand.all(1) & enclosed).reshape(
            len(GROWTH_SCAN), len(rays))
        # Longest unobstructed run outward along each direction, starting at a bond length.
        reach = np.zeros(len(rays))
        for d in range(len(rays)):
            run = 0.0
            for step, ok in zip(GROWTH_SCAN, clear[:, d]):
                if not ok:
                    break
                run = float(step)
            reach[d] = run
        best = int(np.argmax(reach))
        facing = residues[int(np.argmin(np.linalg.norm(pos[i] - pocket, axis=1)))]
        resname = facing.split(":")[1] if ":" in facing else ""
        out.append({"atom_map": maps[i], "facing_residue": facing,
                    "attachable": bool(reach[best] >= GROWTH_BOND),
                    "free_extension_angstrom": float(reach[best]),
                    "free_direction": [float(v) for v in rays[best]],
                    "facing_hydrophobic": resname in HYDROPHOBIC,
                    "facing_charged": resname in CHARGED})
    return out


def geometry(mol, pocket_path: str, tolerance: float) -> dict:
    pos = mol.GetConformer().GetPositions()
    pocket, elements, residues = protein_atoms(pocket_path)
    table = Chem.GetPeriodicTable()
    lr = np.array([table.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])
    pr = np.array([table.GetRvdw(table.GetAtomicNumber(e)) for e in elements])
    d = np.linalg.norm(pos[:, None] - pocket[None, :], axis=-1)
    margin = d - tolerance * (lr[:, None] + pr[None, :])
    maps = [atom.GetAtomMapNum() or index + 1 for index, atom in enumerate(mol.GetAtoms())]
    bad_bonds = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = np.linalg.norm(pos[i] - pos[j])
        covalent_sum = sum(table.GetRcovalent(mol.GetAtomWithIdx(k).GetAtomicNum()) for k in (i, j))
        if not 0.55 * covalent_sum <= length <= 1.35 * covalent_sum:
            bad_bonds.append({"atoms": [i, j], "atom_maps": [maps[i], maps[j]],
                              "length": float(length), "expected": float(covalent_sum)})
    pair_dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    topology = Chem.GetDistanceMatrix(mol)
    internal_margin = pair_dist - 0.6 * (lr[:, None] + lr[None, :])
    nonbonded = np.triu(topology > 2, 1)
    internal_clashes = int(((internal_margin < 0) & nonbonded).sum())
    # Per-atom diagnosis. The margins are already computed for the pass/fail screen; keeping
    # only their minimum throws away WHICH atom is in trouble, which is what a policy needs
    # to decide between editing that atom and regenerating its position.
    symmetric = nonbonded | nonbonded.T
    internal_worst = np.where(symmetric.any(1),
                              np.where(symmetric, internal_margin, np.inf).min(1), np.inf)
    per_atom = [{"atom_map": maps[i], "element": atom.GetSymbol(),
                 "pocket_margin_angstrom": float(margin[i].min()),
                 "pocket_distance_angstrom": float(d[i].min()),
                 "internal_margin_angstrom": (None if not np.isfinite(internal_worst[i])
                                              else float(internal_worst[i])),
                 "bond_strained": any(i in bond["atoms"] for bond in bad_bonds)}
                for i, atom in enumerate(mol.GetAtoms())]
    room = opportunity(pos, pocket, lr, pr, tolerance, residues, maps)
    # Group by the pocket residue each atom faces: four to eight units per ligand is an
    # input a plan can be written over, and unlike a 2D fragment split it comes from the
    # pose, so two atoms of one ring land in different units when they face different walls.
    grouped = {}
    for atom, space in zip(per_atom, room):
        unit = grouped.setdefault(space["facing_residue"], {
            "facing_residue": space["facing_residue"], "atom_maps": [],
            "facing_hydrophobic": space["facing_hydrophobic"],
            "facing_charged": space["facing_charged"]})
        unit["atom_maps"].append(atom["atom_map"])
        unit["min_pocket_margin_angstrom"] = min(
            unit.get("min_pocket_margin_angstrom", float("inf")), atom["pocket_margin_angstrom"])
        unit["max_free_extension_angstrom"] = max(
            unit.get("max_free_extension_angstrom", 0.0), space["free_extension_angstrom"])
        unit["attachable_atom_maps"] = unit.get("attachable_atom_maps", []) + (
            [atom["atom_map"]] if space["attachable"] else [])
        polar = atom["element"] in POLAR_ELEMENTS
        unit["polarity_mismatch"] = unit.get("polarity_mismatch", False) or (
            (polar and space["facing_hydrophobic"]) or (not polar and space["facing_charged"]))
    units = sorted(grouped.values(), key=lambda u: -u["max_free_extension_angstrom"])
    return {"passes": not bad_bonds and internal_clashes == 0 and bool(np.all(margin >= 0)),
            "ligand_bond_geometry_pass": not bad_bonds, "internal_clash_count": internal_clashes,
            "protein_overlap_count": int((margin < 0).sum()), "bad_bonds": bad_bonds,
            "minimum_contact_distance_angstrom": float(d.min()),
            "minimum_contact_margin_angstrom": float(margin.min()),
            "clash_severity_angstrom": float(max(0, -margin.min())),
            "per_atom": per_atom,
            "per_atom_opportunity": room,
            "units": units,
            "clashing_atom_maps": [a["atom_map"] for a in per_atom if a["pocket_margin_angstrom"] < 0],
            "strained_atom_maps": sorted({a["atom_map"] for a in per_atom
                                          if a["bond_strained"] or (
                                              a["internal_margin_angstrom"] is not None
                                              and a["internal_margin_angstrom"] < 0)}),
            "definition": "DIRMO covalent-radius bond screen and scaled VDW distance screen; not PoseBusters"}


def posebusters(mol, pocket: str, *, conditioning_mol=None) -> dict:
    from posebusters import PoseBusters
    checker = PoseBusters(config="dock")
    report = checker.bust(mol_pred=mol, mol_true=None,
                          mol_cond=conditioning_mol if conditioning_mol is not None else Path(pocket),
                          full_report=True)
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
