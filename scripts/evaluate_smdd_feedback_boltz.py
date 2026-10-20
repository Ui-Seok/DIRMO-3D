#!/usr/bin/env python3
"""Check selected feedback-run survivors with the SMDD Boltz oracle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from dirmo.evaluation.smdd_boltz import binding_conditions, cofold_input, evaluation_config
from dirmo.io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/dirmo-boltz2")
    parser.add_argument("--memory-fraction", type=float, default=.27)
    parser.add_argument("--max-parallel-samples", type=int, default=1)
    parser.add_argument("--min-free-mib", type=int, default=14500)
    parser.add_argument("--long-protein-residues", type=int, default=400)
    parser.add_argument("--long-protein-min-free-mib", type=int, default=30000,
                        help="Conservative memory gate based on the previous BRAF oracle OOM")
    args = parser.parse_args()
    if importlib.metadata.version("boltz") != "2.2.1":
        raise ValueError("Use the isolated .venv-boltz environment with Boltz 2.2.1")
    if (not 0 < args.memory_fraction <= 1 or args.max_parallel_samples < 1 or args.min_free_mib < 0
            or args.long_protein_min_free_mib < 0 or args.long_protein_residues < 1):
        raise ValueError("Invalid resource limits")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Specify one physical GPU using CUDA_VISIBLE_DEVICES")
    import torch
    if torch.cuda.device_count() != 1:
        raise ValueError("Boltz evaluation must see exactly one GPU")
    base = json.loads((args.run / "report.json").read_text())
    pending = json.loads((args.run / "pending_boltz.json").read_text())
    output = args.run / "boltz"
    output.mkdir(exist_ok=True)
    identity = {"packages": {p: importlib.metadata.version(p) for p in ["boltz", "torch", "rdkit", "numpy"]},
                "weights_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in args.cache.glob("*.ckpt")},
                "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
                "actual_gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                "seed": 42, "sampling_steps": 200,
                "max_parallel_samples": args.max_parallel_samples, "no_kernels": True}
    if not identity["weights_sha256"]:
        raise ValueError("Download Boltz weights to --cache before evaluating")
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("Boltz generation parameters/environment changed; preserve old outputs and use a new boltz folder")
    write_json(identity_path, identity)
    jobs = []
    for job in pending:
        source = Path(job["source"])
        config = evaluation_config(yaml.safe_load((source / "task.yaml").read_text()))
        data = cofold_input((source / "protein.fasta").read_text(), job["product_smiles"], config)
        size = sum(len(s["protein"]["sequence"]) for s in data["sequences"] if "protein" in s)
        jobs.append((size, job, config, data))
    jobs.sort(key=lambda j: j[0])
    results = []
    started = time.monotonic()
    for size, job, config, data in jobs:
        name = "candidate_" + job["job_id"]
        folder = output / name
        folder.mkdir(exist_ok=True)
        input_path = folder / f"{name}.yaml"
        yaml_text = yaml.safe_dump(data, sort_keys=False)
        if input_path.exists() and input_path.read_text() != yaml_text:
            raise ValueError("Cached Boltz input changed")
        input_path.write_text(yaml_text)
        affinity_path = folder / f"boltz_results_{name}" / "predictions" / name / f"affinity_{name}.json"
        def complete():
            return (affinity_path.exists() and len(list(affinity_path.parent.glob(f"{name}_model_*.cif")))
                    == config["diffusion_samples"])
        entry = {**job, "protein_residues": size, "config": config,
                 "full_benchmark_success": None, "status": "unmeasured",
                 "memory_fraction": args.memory_fraction}
        reused = complete()
        if not reused:
            free, total = torch.cuda.mem_get_info()
            long_protein = size >= args.long_protein_residues
            free_floor = args.long_protein_min_free_mib if long_protein else args.min_free_mib
            required = max(free_floor * 1024**2, total * args.memory_fraction + 1024**3)
            # A memory allowance that previously failed on BRAF is not retried
            # blindly just because physical memory is now free. These are
            # conservative scheduling limits, not a claim of exact model need.
            allowance = total * args.memory_fraction
            if long_protein and allowance < args.long_protein_min_free_mib * 1024**2 * .9:
                entry.update(reason="conservative_long_protein_memory_limit", free_mib=free // 1024**2,
                             required_free_mib=round(required / 1024**2),
                             configured_allowance_mib=round(allowance / 1024**2))
                results.append(entry)
                write_json(output / "results.json", results)
                print(json.dumps({"event": "boltz_deferred", "job_id": job["job_id"],
                                  "reason": entry["reason"]}), flush=True)
                continue
            if free < required:
                entry.update(reason="insufficient_free_gpu_memory", free_mib=free // 1024**2,
                             required_mib=round(required / 1024**2))
                results.append(entry)
                write_json(output / "results.json", results)
                print(json.dumps({"event": "boltz_deferred", "job_id": job["job_id"], **entry}), flush=True)
                continue
            limited = ("import torch; "
                       f"torch.cuda.set_per_process_memory_fraction({args.memory_fraction}, 0); "
                       "from boltz.main import cli; cli()")
            command = [sys.executable, "-c", limited, "predict", str(input_path.resolve()),
                       "--out_dir", str(folder.resolve()), "--cache", str(args.cache.resolve()),
                       "--use_msa_server", "--use_potentials", "--no_kernels", "--seed", "42",
                       "--diffusion_samples", str(config["diffusion_samples"]),
                       "--diffusion_samples_affinity", str(config["diffusion_samples"]),
                       "--sampling_steps", "200", "--max_parallel_samples", str(args.max_parallel_samples),
                       "--num_workers", "0"]
            entry["command"] = command
            print(json.dumps({"event": "boltz_started", "task_id": job["task_id"], "job_id": job["job_id"]}), flush=True)
            with (folder / "boltz.log").open("a") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         env={**os.environ, "WANDB_MODE": "disabled"})
            entry["returncode"] = process.returncode
            entry["log"] = str(folder / "boltz.log")
        if complete() and (reused or entry.get("returncode") == 0):
            try:
                checked = binding_conditions(json.loads(affinity_path.read_text()), config)
                entry.update(status="measured", binding_conditions=checked,
                             full_benchmark_success=checked["all_passed"], affinity_json=str(affinity_path),
                             reused=reused)
            except (ValueError, KeyError, TypeError) as exc:
                entry.update(reason="invalid_affinity_output", error=f"{type(exc).__name__}: {exc}",
                             affinity_json=str(affinity_path))
        results.append(entry)
        write_json(output / "results.json", results)
        print(json.dumps({"event": "boltz_finished", "job_id": job["job_id"],
                          "status": entry["status"], "full_success": entry["full_benchmark_success"]}), flush=True)
    lookup = {r["job_id"]: r for r in results}
    for job in pending:
        result = lookup[job["job_id"]]
        for selection in job["selections"]:
            chosen = next(s for s in base["selected"] if s["task_id"] == selection["task_id"]
                          and s["mode"] == selection["mode"])
            chosen["assessment"]["full_benchmark_success"] = result["full_benchmark_success"]
            chosen["assessment"]["boltz_result"] = result
    for mode, counts in base["comparison"].items():
        selected = [s for s in base["selected"] if s["mode"] == mode]
        counts.update(
            full_success_confirmed=sum(s["assessment"]["full_benchmark_success"] is True for s in selected),
            full_failure_confirmed=sum(s["assessment"]["full_benchmark_success"] is False for s in selected),
            unmeasured_binding=sum(s["assessment"]["full_benchmark_success"] is None for s in selected))
    base.update(status="complete" if all(r["status"] == "measured" for r in results) else "binding_partial",
                boltz_seconds=time.monotonic() - started)
    write_json(args.run / "report_with_boltz.json", base)
    write_json(output / "report.json", {"status": base["status"], "results": results,
                                       "comparison": base["comparison"], "seconds": base["boltz_seconds"]})
    print(json.dumps({k: v for k, v in base.items() if k != "selected"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
