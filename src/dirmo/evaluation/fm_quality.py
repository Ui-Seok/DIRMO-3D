"""Reproducible raw-generation study, separate from DIRMO optimization policies.

GPU phases run in separate processes with independent model/RNG state. The
explicit phase concurrency is recorded; CPU evaluation never imports the model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import copy
from functools import lru_cache
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..chemistry.graphs import graph_id, mapped, read_sdf, smiles, write_sdf
from ..chemistry.properties import graph_properties
from ..evaluation.pocket import prepare_fixed_pocket, protein_atoms, structure
from ..gpu import inventory, resolve_device, worker_environment
from ..io import digest, environment, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS = {
    "base": ("flowr_root_base_spindr.ckpt", "78ef9ebe4459df1ce2668685785602a37288594db83d2bb5d970c8bc06e1ce9d"),
    "v2": ("flowr_root_v2.ckpt", "f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0"),
}
POSE_CONDITIONS = [("random", "random", 0.), ("reference0", "reference", 0.),
                   ("reference03", "reference", .3)]
PHASES = [f"{checkpoint}_{mode}" for checkpoint in CHECKPOINTS for mode in ("graph", "fragment")]


def emit(**values):
    print(json.dumps(values, allow_nan=False), flush=True)


def fragment_regions(mol):
    """Predetermined small tip and a larger connected region, without outcome data."""
    count = mol.GetNumAtoms()
    pieces = []
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        copy = Chem.RWMol(mol)
        copy.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for fragment in Chem.GetMolFrags(copy.GetMol(), sanitizeFrags=False):
            if 0 < len(fragment) < count / 2:
                pieces.append(tuple(sorted(fragment)))
    if not pieces:
        small = [min(range(count), key=lambda i: (mol.GetAtomWithIdx(i).GetDegree(), i))]
    else:
        choices = [p for p in pieces if len(p) <= 3] or pieces
        small = list(min(choices, key=lambda p: (-min(len(p), 3), len(p), p)))
        small = small[:3]
    wanted = min(count - 1, max(len(small) + 1, round(count * .2)))
    large = set(small)
    while len(large) < wanted:
        boundary = sorted({a.GetIdx() for i in large for a in mol.GetAtomWithIdx(i).GetNeighbors()} - large)
        if not boundary:
            break
        large.add(boundary[0])
    return {"small": [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in small],
            "large": [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in sorted(large)]}


def prepare_receptor(protein, destination):
    """Explicit, repeatable scoring receptor; never alter the FM pocket."""
    from Bio.PDB import PDBIO, Select, is_aa
    histidines = []
    omitted = []
    for chain in next(protein.get_models()):
        for residue in chain:
            if not is_aa(residue, standard=True):
                omitted.append(f"{chain.id}:{residue.id[1]}:{residue.resname}")
            elif residue.resname == "HIS":
                histidines.append(f"{chain.id.strip()}:{residue.id[1]}{residue.id[2].strip()}")

    class StandardProtein(Select):
        def accept_model(self, model):
            return model.id == next(protein.get_models()).id

        def accept_residue(self, residue):
            return is_aa(residue, standard=True)

        def accept_atom(self, atom):
            return (atom.element not in ("H", "D") and
                    (not atom.is_disordered() or atom.get_altloc() in ("A", " ")))

    file = destination / "score-protein.pdb"
    writer = PDBIO()
    writer.set_structure(protein)
    writer.save(str(file), StandardProtein())
    receptor = destination / "receptor.pdbqt"
    command = [str(Path(sys.executable).parent / "mk_prepare_receptor.py"),
               "--read_pdb", str(file), "--write_pdbqt", str(receptor), "--default_altloc", "A"]
    if histidines:
        command.extend(["--set_template", ",".join(histidines) + "=HIE"])
    with (destination / "receptor-preparation.log").open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=log, timeout=120)
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = 124
    ready = status == 0 and receptor.exists()
    write_json(destination / "receptor-preparation.json", {
        "command": command, "status": status, "omitted_nonstandard_residues": omitted,
        "protonation": "Meeko residue templates; explicitly neutral HIE for every HIS; not pH optimized",
        "coordinates": "original heavy coordinates; no relaxation; FM pocket unchanged"})
    return str(receptor) if ready else None, status


def prepare(args):
    output = args.output.resolve()
    if (output / "study.json").exists():
        if not args.resume:
            raise FileExistsError("Study exists; use --resume with its original settings")
        plan = json.loads((output / "study.json").read_text())
        for key, value in {"cases_requested": args.cases, "seeds": args.seeds,
                           "seed": args.seed, "steps": args.steps}.items():
            if plan[key] != value:
                raise ValueError(f"Resume setting differs: {key}")
        return plan
    output.mkdir(parents=True, exist_ok=True)
    candidates = sorted(p for p in args.dataset.iterdir() if p.is_dir())
    np.random.default_rng(args.seed).shuffle(candidates)
    cases, exclusions, seen_sequences, seen_scaffolds = [], [], set(), set()
    from Bio.SeqUtils import seq1
    for directory in candidates:
        identifier = directory.name
        ligand_path = directory / f"{identifier}_ligand.sdf"
        protein_path = directory / f"{identifier}_protein.pdb"
        try:
            source = read_sdf(ligand_path)
            hydrogen_options = Chem.RemoveHsParameters()
            hydrogen_options.removeWithQuery = True
            source = mapped(Chem.RemoveHs(source, hydrogen_options))
            if any(a.GetAtomicNum() == 1 for a in source.GetAtoms()):
                raise ValueError("Unsupported retained isotope/hydrogen in heavy-atom study")
            # PDBbind molfile query annotations can prevent normal H removal.
            # Rebuild the same typed graph and carry each mapped heavy coordinate,
            # removing only query annotations, never changing element/bond/stereo.
            clean = Chem.MolFromSmiles(smiles(source, True))
            clean_order = {a.GetAtomMapNum(): a.GetIdx() for a in clean.GetAtoms()}
            truth = Chem.RenumberAtoms(clean, [clean_order[a.GetAtomMapNum()] for a in source.GetAtoms()])
            conformer = Chem.Conformer(truth.GetNumAtoms())
            conformer.Set3D(True)
            for index, xyz in enumerate(source.GetConformer().GetPositions()):
                conformer.SetAtomPosition(index, xyz)
            truth.AddConformer(conformer)
            if smiles(truth) != smiles(source):
                raise ValueError("Removing query annotations changed typed graph")
            if not 10 <= truth.GetNumAtoms() <= 64:
                raise ValueError("Heavy atom count outside study eligibility [10,64]")
            protein = structure(str(protein_path))
            sequences = tuple(sorted("".join(seq1(r.resname) for r in chain.get_residues()
                                             if r.id[0] == " ") for chain in next(protein.get_models())))
            sequence_id = digest(sequences)
            if sequence_id in seen_sequences:
                raise ValueError("Duplicate exact protein sequence in pilot")
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles(truth)))
            if scaffold and scaffold in seen_scaffolds:
                raise ValueError("Duplicate exact Murcko scaffold in pilot")
            destination = output / "cases" / identifier
            destination.mkdir(parents=True, exist_ok=True)
            truth_file = destination / "ligand_truth.sdf"
            write_sdf(truth_file, truth)
            # Use the persisted SDF representation, including MDL stereo handling,
            # for the typed request, rather than the in-memory pre-write molecule.
            truth = read_sdf(truth_file)
            if graph_id(truth) != graph_id(source):
                raise ValueError("Native typed stereo graph changes during SDF roundtrip")
            pocket_file = destination / "pocket.pdb"
            prep = prepare_fixed_pocket(str(protein_path), str(truth_file), str(pocket_file), 6.)
            positions, _, _ = protein_atoms(str(pocket_file))
            if not 20 <= len(positions) <= 1000:
                raise ValueError("Pocket heavy atom count outside study eligibility [20,1000]")
            # Only connectivity/stereo are used for embedding. Native coordinates never
            # enter this conformer's initialization or alignment.
            embed = Chem.AddHs(Chem.MolFromSmiles(smiles(truth, True)))
            parameters = AllChem.ETKDGv3()
            parameters.randomSeed = args.seed + len(cases)
            if AllChem.EmbedMolecule(embed, parameters) != 0:
                raise ValueError("ETKDG embedding failed")
            embed = Chem.RemoveHs(embed)
            map_order = {a.GetAtomMapNum(): a.GetIdx() for a in embed.GetAtoms()}
            embed = Chem.RenumberAtoms(embed, [map_order[a.GetAtomMapNum()] for a in truth.GetAtoms()])
            coords = embed.GetConformer().GetPositions()
            random = np.random.default_rng(args.seed + len(cases))
            rotation, _ = np.linalg.qr(random.normal(size=(3, 3)))
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            center = (positions.min(0) + positions.max(0)) / 2
            coords = (coords - coords.mean(0)) @ rotation + center
            for i, xyz in enumerate(coords):
                embed.GetConformer().SetAtomPosition(i, xyz)
            input_file = destination / "ligand_input.sdf"
            write_sdf(input_file, embed)
            persisted_input = read_sdf(input_file)
            if graph_id(persisted_input) != graph_id(truth):
                raise ValueError("Embedded graph/stereo differs from crystal ligand")
            if smiles(persisted_input, True) != smiles(truth, True):
                raise ValueError("Mapped stereo serialization differs between input and native ligand")
            # Meeko prepares a separate score-only receptor. No residue deletion or
            # geometry relaxation is allowed if preparation cannot match templates.
            receptor, receptor_status = prepare_receptor(protein, destination)
            case = {"case_id": identifier, "target_id": sequence_id, "scaffold": scaffold,
                    "protein_path": str(protein_path), "pocket_path": str(pocket_file),
                    "input_sdf": str(input_file), "truth_sdf": str(truth_file),
                    "graph_smiles": smiles(truth, True), "graph_id": graph_id(truth),
                    "atoms": truth.GetNumAtoms(), "pocket_atoms": len(positions),
                    "formal_charge": Chem.GetFormalCharge(truth),
                    "stereocenters": len(Chem.FindMolChiralCenters(truth, includeUnassigned=False)),
                    "regions": fragment_regions(truth), "pocket_preparation": prep,
                    "box_center": center.tolist(), "box_size": [30., 30., 30.],
                    "receptor_path": receptor, "receptor_preparation_status": receptor_status,
                    "initial_rmsd": float(rdMolAlign.CalcRMS(embed, truth, maxMatches=4096)),
                    "source_hashes": {"ligand": sha256_file(ligand_path), "protein": sha256_file(protein_path)},
                    "prepared_hashes": {"input": sha256_file(input_file), "truth": sha256_file(truth_file),
                                        "pocket": sha256_file(pocket_file)}}
            write_json(destination / "case.json", case)
            cases.append(case)
            seen_sequences.add(sequence_id)
            if scaffold:
                seen_scaffolds.add(scaffold)
            emit(status="prepared", case=identifier, count=len(cases), atoms=case["atoms"],
                 pocket_atoms=case["pocket_atoms"], receptor_ready=receptor is not None)
            if len(cases) == args.cases:
                break
        except Exception as exc:
            exclusions.append({"case_id": identifier, "reason": f"{type(exc).__name__}: {exc}"})
            if len(exclusions) <= 3:
                emit(status="eligibility_exclusion", **exclusions[-1])
    write_json(output / "eligibility-exclusions.json", exclusions)
    if len(cases) != args.cases:
        raise ValueError(f"Only {len(cases)} eligible complexes found")
    plan = {"schema_version": 1, "cases_requested": args.cases, "cases": cases,
            "seeds": args.seeds, "seed": args.seed, "steps": args.steps,
            "dataset": str(args.dataset), "gpu": args.gpu,
            "source_revision": "739a33c2aab74d074dd24605b9a2e34dc809ae9c",
            "checkpoints": CHECKPOINTS, "environment": environment(),
            "code_sha256": sha256_file(__file__),
            "adapter_sha256": sha256_file(ROOT / "src/dirmo/generation/flowr.py"),
            "planned_fm_draws": args.cases * args.seeds * 5 * len(CHECKPOINTS),
            "truth_access": "Evaluator only for graph tasks; native held core allowed for fragment tasks",
            "selection": "Seed-shuffled eligibility; unique exact protein sequences and Murcko scaffolds",
            "pretraining_overlap": "unknown; not an unseen-target generalization benchmark",
            "geometry_pocket": "6 A residue pocket, protein only; no water/cofactor validation",
            "pose_initialization": "ETKDG from graph/stereo, seeded rotation, pocket bounding-box center; no ligand alignment",
            "evaluation": "raw; no repair/relaxation/filtering; score-only Vina where receptor prep succeeds",
            "primary_pose_endpoint": "contract valid AND PB dock valid AND in-place symmetry RMSD <= 2 A"}
    write_json(output / "study.json", plan)
    return plan


def cpu_initializer():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"


@lru_cache(maxsize=4)
def score_engine(receptor, center, box):
    from vina import Vina
    scorer = Vina(sf_name="vina", cpu=1, seed=0, verbosity=0)
    scorer.set_receptor(receptor)
    scorer.compute_vina_maps(center=list(center), box_size=list(box))
    return scorer


def vina_for_mol(mol, case):
    if not case["receptor_path"]:
        return {"status": "missing", "reason": "Receptor preparation failed; see receptor-preparation.log"}
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    setups = MoleculePreparation().prepare(Chem.AddHs(mol, addCoords=True))
    if len(setups) != 1:
        raise ValueError("Multiple ligand preparations")
    text, success, reason = PDBQTWriterLegacy.write_string(setups[0])
    if not success:
        raise ValueError(reason)
    scorer = score_engine(case["receptor_path"], tuple(case["box_center"]), tuple(case["box_size"]))
    scorer.set_ligand_from_string(text)
    return {"status": "ok", "score": float(scorer.score()[0]), "mode": "score_only"}


def held_contract(reference, actual, free_maps):
    if reference.GetNumAtoms() != actual.GetNumAtoms():
        return {"passes": False, "reason": "Atom count differs"}
    held = [i for i, a in enumerate(reference.GetAtoms()) if a.GetAtomMapNum() not in free_maps]
    mismatches = []
    for i in held:
        left, right = reference.GetAtomWithIdx(i), actual.GetAtomWithIdx(i)
        if (left.GetAtomicNum(), left.GetFormalCharge(), left.GetIsotope()) != (
                right.GetAtomicNum(), right.GetFormalCharge(), right.GetIsotope()):
            mismatches.append({"atom_index": i, "kind": "held_atom"})
    for position, i in enumerate(held):
        for j in held[position + 1:]:
            a, b = reference.GetBondBetweenAtoms(i, j), actual.GetBondBetweenAtoms(i, j)
            signature = lambda bond: None if bond is None else (str(bond.GetBondType()), bond.GetIsAromatic())
            if signature(a) != signature(b):
                mismatches.append({"atom_indices": [i, j], "kind": "held_bond"})
    distances = np.linalg.norm(reference.GetConformer().GetPositions()[held] -
                               actual.GetConformer().GetPositions()[held], axis=1)
    drift = float(distances.max())
    return {"passes": not mismatches and drift <= .001, "held_count": len(held),
            "held_max_drift": drift, "mismatches": mismatches,
            "coordinate_tolerance_angstrom": .001, "mapping": "verified upstream reference atom order"}


def evaluate_molecule(path, case, mode="graph", free_maps=()):
    started = time.monotonic()
    from .metrics import posebusters
    result = {"build_valid": False, "contract_valid": False, "pb_valid": False,
              "pose_success": False, "evaluation_errors": {}}
    try:
        molecule = read_sdf(path)
        truth = read_sdf(case["truth_sdf"])
        result["build_valid"] = True
        result["smiles"] = smiles(molecule)
        result["changed_graph"] = graph_id(molecule) != case["graph_id"]
        result["properties"] = graph_properties(result["smiles"])
        if mode == "fragment":
            result["held_contract"] = held_contract(truth, molecule, free_maps)
            result["contract_valid"] = result["held_contract"]["passes"]
        else:
            # Check actual stereochemistry from coordinates, not only SDF metadata.
            from ..chemistry.graphs import graph_check
            result["graph_check"] = graph_check(truth, molecule, derive_stereo=True)
            result["contract_valid"] = result["graph_check"]["matches"]
        try:
            pb = posebusters(molecule, case["pocket_path"])
            result["posebusters"] = pb
            result["pb_valid"] = pb["passes"]
        except Exception as exc:
            result["evaluation_errors"]["posebusters"] = f"{type(exc).__name__}: {exc}"
        if mode == "graph" and result["contract_valid"]:
            left, right = Chem.Mol(molecule), Chem.Mol(truth)
            for mol in (left, right):
                for atom in mol.GetAtoms():
                    atom.SetAtomMapNum(0)
            result["rmsd"] = float(rdMolAlign.CalcRMS(left, right, maxMatches=4096))
            result["pose_success"] = result["pb_valid"] and result["rmsd"] <= 2.
        result["valid_changed_fragment"] = (mode == "fragment" and result["contract_valid"]
                                            and result["pb_valid"] and result["changed_graph"])
        try:
            result["vina"] = vina_for_mol(molecule, case)
        except Exception as exc:
            result["vina"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        result["evaluation_errors"]["molecule"] = f"{type(exc).__name__}: {exc}"
    result["evaluation_seconds"] = time.monotonic() - started
    return result


def evaluate_draw(row, case):
    generation = row["generation"]
    path = generation.get("sdf_path")
    metrics = evaluate_molecule(path, case, row["mode"], row.get("free_maps", [])) if path else {
        "build_valid": False, "contract_valid": False, "pb_valid": False, "pose_success": False,
        "valid_changed_fragment": False, "generation_failure": generation.get("reason")}
    if row["mode"] == "graph" and not generation.get("graph_check", {}).get("matches", False):
        metrics["contract_valid"] = False
        metrics["pose_success"] = False
    write_json(Path(row["directory"]) / "metrics.json", metrics)
    if row["mode"] == "graph":
        prior_file = Path(row["directory"]) / "prior_coordinates.sdf"
        if prior_file.exists():
            write_json(Path(row["directory"]) / "prior-metrics.json", evaluate_molecule(str(prior_file), case))
    return row["draw_id"]


def generate_phase(args):
    from ..config import FMConfig
    from ..generation.flowr import FlowrBackend, CapabilityError
    from ..schemas import GenerationParameters, GenerationRequest
    output = args.output.resolve()
    plan = json.loads((output / "study.json").read_text())
    checkpoint, mode = args.phase.split("_")
    filename, expected_hash = CHECKPOINTS[checkpoint]
    config = FMConfig(gpu=args.gpu, port=8876, python=sys.executable,
                      repo_path="/home/uslee/code/flowr_root", repo_revision=plan["source_revision"],
                      checkpoint_path=str(args.checkpoints / filename), checkpoint_sha256=expected_hash,
                      inpaint_mode=mode, max_atoms=64, max_pocket_atoms=1000)
    backend = FlowrBackend(config)
    # Deterministic pocket/graph preparation is identical across seeds and priors.
    # Cache only this CPU operation; sampling and integration still run per draw.
    from flowr.gen import utils as upstream_utils
    native_load = upstream_utils.load_data_from_pdb
    systems = {}

    def cached_load(load_args, **kwargs):
        key = (load_args.pdb_file, load_args.ligand_file, tuple(sorted(kwargs.items())))
        if key not in systems:
            systems[key] = native_load(load_args, **kwargs)
        return copy.deepcopy(systems[key])

    upstream_utils.load_data_from_pdb = cached_load
    health = backend.health()
    write_json(output / "phases" / args.phase / "health.json", {
        **health, "physical_gpu": args.gpu, "pid": os.getpid(),
        "benchmark_code_sha256": sha256_file(__file__),
        "training_graph_condition_provenance": "unknown; graph_inpainting key not saved"})
    cases = plan["cases"][:6] if args.conformance else plan["cases"]
    seeds = min(5, plan["seeds"]) if args.conformance else plan["seeds"]
    if args.probe_only:
        cases, seeds = cases[:1], 1
    native_generate = backend.model._generate
    capture = {}

    def capture_prior(ligand, pocket, *positional, **keywords):
        # This is an audit copy only. COM metadata remains on CPU even after
        # sampler tensors move to CUDA; keep the entire recording operation on CPU.
        cpu_coords = ligand["coords"].detach().cpu()
        cpu_mask = ligand["mask"].detach().cpu()
        absolute = backend.model.builder.undo_zero_com_batch(
            cpu_coords, cpu_mask, [system.com.detach().cpu() for system in pocket["complex"]])
        coords = absolute[0, cpu_mask[0].bool()].numpy()
        reference = read_sdf(capture["reference_sdf"])
        for index, xyz in enumerate(coords):
            reference.GetConformer().SetAtomPosition(index, xyz.astype(float))
        write_sdf(capture["directory"] / "prior_coordinates.sdf", reference)
        received = ligand.get("fragment_mask")
        actual_mask = received[0].detach().cpu().bool().tolist() if received is not None else None
        write_json(capture["directory"] / "model-condition.json", {
            "actual_held_mask": actual_mask, "feature_keys": list(backend.model.feature_keys),
            "prior_coords_absolute": coords.tolist(),
            "prior_control_discrete_graph": "reference graph; coordinates are exact sampler inputs",
            "full_reference_coords_passed_as_ligand_condition": mode == "graph" and capture["prior"] == "reference",
            "expected_free_maps": capture["free_maps"]})
        expected_mask = [a.GetAtomMapNum() not in capture["free_maps"] for a in reference.GetAtoms()]
        if actual_mask != expected_mask:
            raise RuntimeError("CONFORMANCE: actual model mask differs from requested mask")
        return native_generate(ligand, pocket, *positional, **keywords)

    backend.model._generate = capture_prior
    futures = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=cpu_initializer) as pool:
        completed = 0
        for case in cases:
            for key, file in [("input", case["input_sdf"]), ("truth", case["truth_sdf"]),
                              ("pocket", case["pocket_path"])]:
                if sha256_file(file) != case["prepared_hashes"][key]:
                    raise ValueError(f"Prepared input changed: {case['case_id']} {key}")
            conditions = POSE_CONDITIONS if mode == "graph" else [
                (f"fragment_{size}", "fragment", 0.) for size in ("small", "large")]
            if args.probe_only:
                conditions = conditions[:1]
            for condition, prior, noise in conditions:
                free_maps = case["regions"][condition.removeprefix("fragment_")] if mode == "fragment" else []
                for offset in range(seeds):
                    seed = plan["seed"] + offset
                    draw_id = digest([case["case_id"], args.phase, condition, seed])[:20]
                    directory = output / "draws" / args.phase / case["case_id"] / condition / str(seed)
                    result_file = directory / "result.json"
                    if result_file.exists():
                        row = json.loads(result_file.read_text())
                        if not (directory / "metrics.json").exists():
                            futures.append(pool.submit(evaluate_draw, row, case))
                        continue
                    if (directory / "raw.sdf").exists() or (directory / "request.json").exists():
                        raise RuntimeError(f"Incomplete previous draw; inspect before resampling: {directory}")
                    parameters = GenerationParameters(prior=prior, prior_noise_std=noise,
                                                      prior_free_atom_maps=free_maps,
                                                      integration_steps=plan["steps"])
                    reference = case["truth_sdf"] if mode == "fragment" else case["input_sdf"]
                    request = GenerationRequest(graph_smiles=case["graph_smiles"], graph_id=case["graph_id"],
                                                pocket_path=case["pocket_path"], reference_sdf=reference,
                                                parameters=parameters, seed=seed, output_dir=str(directory))
                    capture.update(directory=directory, reference_sdf=reference, free_maps=free_maps, prior=prior)
                    try:
                        generated = backend.generate_fixed_graph(request).model_dump(mode="json")
                    except CapabilityError as exc:
                        generated = {"status": "capability_error", "reason": str(exc), "raw_draws": 0, "nfe": 0}
                    row = {"draw_id": draw_id, "case_id": case["case_id"], "target_id": case["target_id"],
                           "checkpoint": checkpoint, "checkpoint_sha256": expected_hash,
                           "mode": mode, "condition": condition, "seed": seed, "steps": plan["steps"],
                           "free_maps": free_maps, "directory": str(directory), "generation": generated}
                    write_json(result_file, row)
                    if "CONFORMANCE:" in (generated.get("reason") or ""):
                        raise RuntimeError(generated["reason"])
                    futures.append(pool.submit(evaluate_draw, row, case))
                    completed += 1
                    if completed % 10 == 0 or generated["status"] == "capability_error":
                        emit(status="generating", phase=args.phase, conformance=args.conformance,
                             completed_new=completed, case=case["case_id"], result=generated["status"])
        for future in futures:
            future.result()
    emit(status="phase_complete", phase=args.phase, conformance=args.conformance, new_draws=completed)


def baseline_case(case, output):
    output = Path(output) / "cases" / case["case_id"]
    write_json(output / "input-metrics.json", evaluate_molecule(case["input_sdf"], case))
    write_json(output / "native-metrics.json", evaluate_molecule(case["truth_sdf"], case))
    if not case["receptor_path"]:
        write_json(output / "docking.json", {"status": "missing", "reason": "Receptor preparation failed"})
        return
    started = time.monotonic()
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
        from vina import Vina
        initial = read_sdf(case["input_sdf"])
        setup = MoleculePreparation().prepare(Chem.AddHs(initial, addCoords=True))[0]
        text, success, reason = PDBQTWriterLegacy.write_string(setup)
        if not success:
            raise ValueError(reason)
        engine = Vina(sf_name="vina", cpu=1, seed=42, verbosity=0)
        engine.set_receptor(case["receptor_path"])
        engine.set_ligand_from_string(text)
        engine.compute_vina_maps(center=case["box_center"], box_size=case["box_size"])
        engine.dock(exhaustiveness=8, n_poses=10)
        docked = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(engine.poses(n_poses=10), skip_typing=True))
        if len(docked) != 1 or docked[0] is None:
            raise ValueError("Docking conversion did not return one ligand graph")
        molecule = Chem.RemoveHs(docked[0])
        metrics = []
        for conformer in molecule.GetConformers():
            single = Chem.Mol(molecule)
            single.RemoveAllConformers()
            single.AddConformer(Chem.Conformer(conformer), assignId=True)
            path = output / f"docked_{len(metrics)}.sdf"
            write_sdf(path, single)
            metrics.append(evaluate_molecule(str(path), case))
        write_json(output / "docking.json", {"status": "ok", "poses": metrics,
                                              "latency_seconds": time.monotonic() - started,
                                              "exhaustiveness": 8, "requested_poses": 10,
                                              "actual_poses": len(metrics), "seed": 42})
    except Exception as exc:
        write_json(output / "docking.json", {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})


def aggregate(args):
    output = args.output.resolve()
    plan = json.loads((output / "study.json").read_text())
    rows = []
    for file in sorted((output / "draws").glob("*/*/*/*/result.json")):
        row = json.loads(file.read_text())
        metrics_file = file.parent / "metrics.json"
        row["metrics"] = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
        prior_file = file.parent / "prior-metrics.json"
        if prior_file.exists():
            row["prior_metrics"] = json.loads(prior_file.read_text())
        rows.append(row)
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["checkpoint"], row["condition"])].append(row)
    summary = []
    for (checkpoint, condition), group in sorted(by_group.items()):
        total = len(group)
        by_target = defaultdict(list)
        for row in group:
            by_target[row["target_id"]].append(row)
        primary = "pose_success" if group[0]["mode"] == "graph" else "valid_changed_fragment"
        rates = np.array([np.mean([r["metrics"].get(primary, False) for r in samples])
                          for samples in by_target.values()])
        boot = np.random.default_rng(2026).choice(rates, size=(2000, len(rates)), replace=True).mean(1)
        rmsds = [r["metrics"]["rmsd"] for r in group if "rmsd" in r["metrics"]]
        scores = [r["metrics"]["vina"]["score"] for r in group
                  if r["metrics"].get("vina", {}).get("status") == "ok"]
        coverage = sum(any(r["metrics"].get(primary, False) for r in samples) for samples in by_target.values())
        summary.append({"checkpoint": checkpoint, "condition": condition, "attempts": total,
                        "targets": len(by_target), "raw_draws": sum(r["generation"].get("raw_draws", 0) for r in group),
                        "nfe": sum(r["generation"].get("nfe", 0) for r in group),
                        "generation_status": dict(Counter(r["generation"]["status"] for r in group)),
                        "failure_reasons": dict(Counter(r["generation"].get("reason") for r in group
                                                        if r["generation"].get("reason"))),
                        **{key + "_count": sum(bool(r["metrics"].get(key, False)) for r in group)
                           for key in ["build_valid", "contract_valid", "pb_valid", "pose_success",
                                       "changed_graph", "valid_changed_fragment"]},
                        "primary_endpoint": primary, "primary_rate": float(rates.mean()),
                        "primary_target_bootstrap_95ci": np.quantile(boot, [.025, .975]).tolist(),
                        "targets_with_any_primary_success": coverage,
                        "median_rmsd_contract_valid_only": float(np.median(rmsds)) if rmsds else None,
                        "vina_measured": len(scores), "median_vina_measured_only": float(np.median(scores)) if scores else None,
                        "unique_graphs": len({r["metrics"].get("smiles") for r in group if r["metrics"].get("smiles")}),
                        "generation_seconds": sum(r["generation"].get("latency_seconds", 0) for r in group),
                        "peak_allocated_bytes": max(r["generation"].get("peak_memory_bytes", 0) for r in group)})
    completed = len(rows) == plan["planned_fm_draws"] and all(r["metrics"] for r in rows)
    report = {"status": "complete" if completed else "in_progress", "planned_attempts": plan["planned_fm_draws"],
              "recorded_attempts": len(rows), "evaluated_attempts": sum(bool(r["metrics"]) for r in rows),
              "llm_calls": 0, "gpu": plan["gpu"], "groups": summary,
              "limitations": ["Pretraining overlap unknown; pilot only.", "Exact-sequence/scaffold deduplication is not a sequence-cluster split.",
                              "PoseBusters uses a protein-only pocket without water/cofactor checks on the full complex.",
                              "Vina is score-only; absent receptor/score results are reported as missing.",
                              "Whole-graph training provenance unknown."]}
    write_json(output / "summary.json", report)
    with (output / "draws.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    lines = ["# GPU 3 FLOWR 생성 품질 실험", "", f"상태: {report['status']}; 기록 {len(rows)}/{plan['planned_fm_draws']}, LLM 호출 0.", "",
             "PDBbind2019 원본 복합체를 사용했다. Pose 입력은 native 좌표 정렬 없이 만든 ETKDG conformer다. Fragment는 native held core를 조건으로 사용한다.", "",
             "| checkpoint | condition | attempts | graph/held 계약 | PB-valid | primary 성공 | 적어도 1회 성공 target |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for group in summary:
        success = group["pose_success_count"] if group["primary_endpoint"] == "pose_success" else group["valid_changed_fragment_count"]
        lines.append(f"| {group['checkpoint']} | {group['condition']} | {group['attempts']} | {group['contract_valid_count']} | {group['pb_valid_count']} | {success} | {group['targets_with_any_primary_success']}/{group['targets']} |")
    lines.extend(["", "Pose primary는 계약/PB-valid/RMSD ≤2 Å 동시 통과다. Fragment primary는 held 계약/PB-valid/원본과 다른 graph 동시 통과다. 두 과제의 성공률을 직접 비교하지 않는다.", "",
                  "실패 이유·개별 metric·target bootstrap 신뢰구간은 summary.json/draws.jsonl에 있다. 각 생성의 result.json, metrics.json, model-condition.json에서 실제 model mask와 초기 좌표를 확인할 수 있다.", "",
                  "이 결과는 학습 overlap이 확인되지 않은 pilot이다. 보정·repair 없는 raw 출력 기준이며 docking baseline은 별도 후처리 방법이다."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    emit(status=report["status"], recorded=len(rows), planned=plan["planned_fm_draws"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/LocalData1/uslee/datasets/PDBbind2019/refined-set"))
    parser.add_argument("--checkpoints", type=Path, default=Path("/LocalData1/uslee/checkpoints/FLOWR_root"))
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-concurrency", type=int, choices=[1, 2], default=2,
                        help="Independent checkpoint/mode processes on the selected GPU")
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--conformance", action="store_true")
    parser.add_argument("--probe-only", action="store_true", help="One actual draw to check recording/mask plumbing")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cases <= 100 or not 1 <= args.seeds <= 20 or not 1 <= args.workers <= 8:
        parser.error("Cases/seeds/workers exceed the explicit benchmark caps")
    if args.report_only:
        aggregate(args)
        return
    if args.phase:
        generate_phase(args)
        return
    plan = prepare(args)
    if args.prepare_only:
        emit(status="prepared", planned_draws=plan["planned_fm_draws"])
        return
    device = resolve_device(args.gpu, inventory())
    write_json(args.output / "gpu.json", device)
    plan["gpu_phase_concurrency"] = args.gpu_concurrency
    plan["code_sha256"] = sha256_file(__file__)
    plan["preprocessing_cache"] = "Deterministic source complex cached per phase; deep copy per draw; RNG re-seeded per draw"
    write_json(args.output / "study.json", plan)
    env = worker_environment(device, str(ROOT / "src"))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=cpu_initializer) as pool:
        futures = [pool.submit(baseline_case, case, str(args.output.resolve())) for case in plan["cases"]
                   if not (args.output / "cases" / case["case_id"] / "docking.json").exists()]
        def phase_task(phase, conformance):
            command = [sys.executable, str(ROOT / "scripts/benchmark_fm_conditions.py"),
                       "--output", str(args.output.resolve()), "--phase", phase,
                       "--checkpoints", str(args.checkpoints), "--gpu", args.gpu,
                       "--workers", str(args.workers)]
            if conformance:
                command.append("--conformance")
            log_file = args.output / "phases" / phase / ("conformance.log" if conformance else "generation.log")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            emit(status="starting_phase", phase=phase, conformance=conformance, log=str(log_file))
            with log_file.open("a") as log:
                subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
            return phase

        for conformance in (True, False):
            with ThreadPoolExecutor(max_workers=args.gpu_concurrency) as phases:
                phase_futures = [phases.submit(phase_task, phase, conformance) for phase in PHASES]
                for future in as_completed(phase_futures):
                    future.result()
                    aggregate(args)
        for future in futures:
            future.result()
    aggregate(args)


if __name__ == "__main__":
    main()
