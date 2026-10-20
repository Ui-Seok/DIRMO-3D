"""Measure actual batch-16 pose-only FLOWR inference and three-stage validity."""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import Lipinski

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dirmo.chemistry.graphs import graph_check, read_sdf, smiles, stereo_from_coordinates, write_sdf
from dirmo.evaluation.fm_quality import CHECKPOINTS
from dirmo.gpu import inventory, resolve_device, worker_environment
from dirmo.io import sha256_file, write_json


def emit(**values):
    print(json.dumps(values), flush=True)


def torsion_region(mol):
    matches = {tuple(sorted(pair)) for pair in mol.GetSubstructMatches(Lipinski.RotatableBondSmarts)}
    options = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if tuple(sorted((i, j))) not in matches or bond.GetIsConjugated():
            continue
        cut = Chem.RWMol(mol)
        cut.RemoveBond(i, j)
        for fragment in Chem.GetMolFrags(cut.GetMol(), sanitizeFrags=False):
            if not 2 <= len(fragment) <= min(8, mol.GetNumAtoms() - 4):
                continue
            moving, anchor = (i, j) if i in fragment else (j, i)
            options.append((tuple(sorted(fragment)), anchor, moving))
    if not options:
        raise ValueError("No eligible rotatable terminal region")
    free, anchor, moving = min(options, key=lambda x: (-len(x[0]), x[0], x[1:]))
    return {"free_indices": list(free), "free_maps": [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in free],
            "anchor_index": anchor, "moving_axis_index": moving,
            "axis_atom_maps": [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in (anchor, moving)]}


def distort(mol, region, angle):
    result = Chem.Mol(mol)
    xyz = mol.GetConformer().GetPositions()
    origin = xyz[region["anchor_index"]]
    axis = xyz[region["moving_axis_index"]] - origin
    axis = axis / np.linalg.norm(axis)
    theta = np.deg2rad(angle)
    free = region["free_indices"]
    local = xyz[free] - origin
    rotated = (local * np.cos(theta) + np.cross(axis, local) * np.sin(theta) +
               np.outer(local @ axis, axis) * (1. - np.cos(theta))) + origin
    for index, position in zip(free, rotated):
        result.GetConformer().SetAtomPosition(index, position)
    if not graph_check(mol, result, derive_stereo=True)["matches"]:
        raise ValueError("Torsion distortion changed coordinate stereochemistry")
    return result


def prepare(args):
    output = args.output.resolve()
    if (output / "probe.json").exists():
        return json.loads((output / "probe.json").read_text())
    study = json.loads(args.study.read_text())
    options = []
    for case in study["cases"]:
        try:
            mol = read_sdf(case["truth_sdf"])
            region = torsion_region(mol)
            options.append((case["atoms"] * case["pocket_atoms"], case, mol, region))
        except ValueError:
            continue
    if not options:
        raise ValueError("No eligible case for the batch probe")
    _, case, mol, region = max(options, key=lambda x: x[0])
    output.mkdir(parents=True, exist_ok=True)
    inputs = []
    for index in range(args.batch_size):
        seed = 1042 + index
        difficulty = "small" if index < args.batch_size // 2 else "large"
        rng = np.random.default_rng(seed)
        angle = float(rng.uniform(20, 40) if difficulty == "small" else rng.uniform(50, 70))
        angle *= int(rng.choice([-1, 1]))
        input_file = output / "inputs" / f"{seed}.sdf"
        write_sdf(input_file, stereo_from_coordinates(distort(mol, region, angle)))
        persisted = read_sdf(input_file)
        if not graph_check(mol, persisted, derive_stereo=True)["matches"]:
            raise ValueError("Distorted input stereo changed during serialization")
        inputs.append({"input_sdf": str(input_file), "seed": seed, "difficulty": difficulty,
                       "angle_degrees": angle, "free_maps": region["free_maps"],
                       "pocket_path": case["pocket_path"], "case_id": case["case_id"]})
    plan = {"case": case, "region": region, "inputs": inputs, "batch_size": args.batch_size,
            "steps": args.steps, "purpose": "batch throughput and hard-condition probe, not a 50-case final result",
            "study_source": str(args.study), "script_sha256": sha256_file(__file__)}
    write_json(output / "probe.json", plan)
    return plan


def monitor_memory(uuid, stopped, records):
    while not stopped.is_set():
        try:
            value = subprocess.check_output(["nvidia-smi", "-i", uuid,
                    "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True, timeout=5)
            used, free, util = [int(x.strip()) for x in value.strip().split(",")]
            records.append({"elapsed": time.monotonic(), "used_mib": used, "free_mib": free, "gpu_util": util})
        except Exception as exc:
            records.append({"error": str(exc)})
        stopped.wait(.5)


def generate(args):
    from dirmo.config import FMConfig
    from dirmo.generation.pose_repair import PoseRepairBackend
    plan = json.loads((args.output / "probe.json").read_text())
    filename, expected_hash = CHECKPOINTS[args.checkpoint]
    config = FMConfig(gpu=args.gpu, port=8876, python=sys.executable,
                      repo_path="/home/uslee/code/flowr_root",
                      checkpoint_path=str(args.checkpoints / filename), checkpoint_sha256=expected_hash,
                      inpaint_mode="graph", max_atoms=64, max_pocket_atoms=1000)
    backend = PoseRepairBackend(config)
    health = backend.backend.health()
    write_json(args.output / args.checkpoint / "health.json", health)
    batches = []
    # One full single-sample run provides a timing/control comparison and warms
    # CUDA kernels. Batch-16 receives exactly the same first corrupted input.
    for size in (1, plan["batch_size"]):
        directory = args.output / args.checkpoint / f"batch_{size}"
        items = [{**x, "output_dir": str(directory / str(x["seed"]))} for x in plan["inputs"][:size]]
        records, stopped = [], threading.Event()
        monitor = threading.Thread(target=monitor_memory,
                    args=(os.environ["DIRMO_GPU_UUID"], stopped, records), daemon=True)
        monitor.start()
        emit(status="starting_batch", checkpoint=args.checkpoint, batch_size=size,
             case=plan["case"]["case_id"])
        try:
            results, stats = backend.generate(items, plan["steps"])
        finally:
            stopped.set()
            monitor.join(timeout=6)
        valid_records = [r for r in records if "used_mib" in r]
        stats.update(checkpoint=args.checkpoint, case_id=plan["case"]["case_id"],
                     statuses=dict(Counter(x["status"] for x in results)),
                     peak_device_used_mib=max((r["used_mib"] for r in valid_records), default=None),
                     minimum_device_free_mib=min((r["free_mib"] for r in valid_records), default=None),
                     mean_gpu_util=float(np.mean([r["gpu_util"] for r in valid_records])) if valid_records else None)
        write_json(directory / "memory.json", {"stats": stats, "samples": records})
        batches.append(stats)
        write_json(args.output / args.checkpoint / "summary.json", {"batches": batches})
        emit(**stats)
    first = str(plan["inputs"][0]["seed"])
    left = read_sdf(args.output / args.checkpoint / "batch_1" / first / "raw.sdf")
    right = read_sdf(args.output / args.checkpoint / f"batch_{plan['batch_size']}" / first / "raw.sdf")
    comparison = {"same_input": first, "max_coordinate_difference_angstrom":
                  float(np.linalg.norm(left.GetConformer().GetPositions() - right.GetConformer().GetPositions(), axis=1).max()),
                  "graph_batch1": graph_check(read_sdf(plan["inputs"][0]["input_sdf"]), left, True)["matches"],
                  f"graph_batch{plan['batch_size']}": graph_check(read_sdf(plan["inputs"][0]["input_sdf"]), right, True)["matches"]}
    write_json(args.output / args.checkpoint / "summary.json", {"batches": batches, "batch_consistency": comparison})


def evaluate(args):
    from dirmo.evaluation.metrics import posebusters
    from posebusters.modules import energy_ratio
    # The deterministic 50-conformer reference uses one CPU thread and the same
    # cached result across stages for a given stereograph; raw poses stay intact.
    original = energy_ratio.get_average_energy
    @lru_cache(maxsize=None)
    def average(inchi, n_confs=50, num_threads=0):
        return original(inchi, n_confs, 1)
    energy_ratio.get_average_energy = average
    plan = json.loads((args.output / "probe.json").read_text())
    case = plan["case"]
    native = read_sdf(case["truth_sdf"])
    free = plan["region"]["free_indices"]

    def measure(path):
        mol = stereo_from_coordinates(read_sdf(path))
        graph = graph_check(native, mol, True)
        result = {"sdf": str(path), "graph_preserved": graph["matches"],
                  "free_rmsd": float(np.sqrt(np.mean(np.sum((mol.GetConformer().GetPositions()[free] -
                                      native.GetConformer().GetPositions()[free]) ** 2, axis=1))))}
        for name, pocket in (("pocket", case["pocket_path"]), ("full_protein", case["protein_path"])):
            try:
                result[name] = posebusters(mol, pocket)
            except Exception as exc:
                result[name] = {"passes": False, "error": f"{type(exc).__name__}: {exc}"}
        return result

    result = {"case_id": case["case_id"], "native": measure(case["truth_sdf"]), "inputs": [], "generated": {}}
    write_json(args.output / "physical-validity.json", result)
    for item in plan["inputs"]:
        result["inputs"].append({**item, "metrics": measure(item["input_sdf"])})
        write_json(args.output / "physical-validity.json", result)
    for checkpoint in ("base", "v2"):
        directory = args.output / checkpoint / f"batch_{plan['batch_size']}"
        if not directory.exists():
            continue
        result["generated"][checkpoint] = []
        for item in plan["inputs"]:
            file = directory / str(item["seed"]) / "raw.sdf"
            metrics = measure(file) if file.exists() else {"build_failed": True}
            result["generated"][checkpoint].append({"seed": item["seed"], "metrics": metrics})
            write_json(args.output / "physical-validity.json", result)
    result["complete"] = True
    write_json(args.output / "physical-validity.json", result)
    emit(status="physical_evaluation_complete", case_id=case["case_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--study", type=Path, default=ROOT / "runs/fm_quality_gpu3_20261013_v1/study.json")
    parser.add_argument("--checkpoints", type=Path, default=Path("/LocalData1/uslee/checkpoints/FLOWR_root"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=list(CHECKPOINTS))
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.evaluate:
        evaluate(args)
    elif args.checkpoint:
        generate(args)
    else:
        plan = prepare(args)
        device = resolve_device(args.gpu, inventory())
        if device["total_mib"] - device["used_mib"] < 8000:
            raise RuntimeError("GPU has less than 8 GiB free; no probe started")
        write_json(args.output / "gpu.json", device)
        env = worker_environment(device, str(ROOT / "src"))
        for checkpoint in CHECKPOINTS:
            log = args.output / checkpoint / "generation.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, __file__, "--checkpoint", checkpoint, "--gpu", args.gpu,
                       "--output", str(args.output), "--checkpoints", str(args.checkpoints)]
            with log.open("a") as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=stream, check=True)
            emit(status="checkpoint_probe_complete", checkpoint=checkpoint, batch_size=plan["batch_size"])
        emit(status="gpu_probes_complete", output=str(args.output))


if __name__ == "__main__":
    main()
