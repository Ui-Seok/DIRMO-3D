#!/usr/bin/env python3
"""Evaluate survivors only: any failed hard/ADMET gate already rules out success."""
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--admet", type=Path, default=Path("runs/smdd_full_01/admet"))
    parser.add_argument("--output", type=Path, default=Path("runs/smdd_full_01/boltz_heldout"))
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/dirmo-boltz2")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"], choices=["train", "validation", "test"])
    parser.add_argument("--memory-fraction", type=float, default=0.27)
    parser.add_argument("--max-parallel-samples", type=int, default=2)
    args = parser.parse_args()
    if importlib.metadata.version("boltz") != "2.2.1":
        raise ValueError("Use Boltz 2.2.1 to match SMDD")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("Invalid memory fraction")
    admet = json.loads((args.admet / "report.json").read_text())
    if admet["status"] != "complete":
        raise ValueError("Complete ADMET evaluation before running Boltz")
    args.output.mkdir(parents=True, exist_ok=True)
    pending = [r for r in json.loads((args.admet / "pending_boltz.json").read_text()) if r["split"] in args.splits]
    jobs = []
    for record in pending:
        source = Path(record["source"])
        task = yaml.safe_load((source / "task.yaml").read_text())
        config = evaluation_config(task)
        data = cofold_input((source / "protein.fasta").read_text(), record["product_smiles"], config)
        size = sum(len(s["protein"]["sequence"]) for s in data["sequences"] if "protein" in s)
        jobs.append((size, record, config, data))
    jobs.sort(key=lambda job: job[0])
    results = []
    started = time.monotonic()
    for _, record, config, data in jobs:
        job = f"{record['model']}_{record['task_id']}"
        folder = args.output / job
        folder.mkdir(exist_ok=True)
        input_path = folder / f"{job}.yaml"
        input_path.write_text(yaml.safe_dump(data, sort_keys=False))
        # A subprocess releases each model before the next job, retaining CLI resumability.
        limited_cli = ("import torch; "
                       f"torch.cuda.set_per_process_memory_fraction({args.memory_fraction}, 0); "
                       "from boltz.main import cli; cli()")
        command = [sys.executable, "-c", limited_cli, "predict", str(input_path.resolve()),
                   "--out_dir", str(folder.resolve()), "--cache", str(args.cache.resolve()),
                   "--use_msa_server", "--use_potentials", "--no_kernels", "--seed", "42",
                   "--diffusion_samples", str(config["diffusion_samples"]),
                   "--diffusion_samples_affinity", str(config["diffusion_samples"]),
                   "--sampling_steps", "200", "--max_parallel_samples", str(args.max_parallel_samples),
                   "--num_workers", "0"]
        entry = {**record, "command": command, "config": config}
        print(json.dumps({"event": "boltz_started", "task_id": record["task_id"], "model": record["model"]}), flush=True)
        environment = {**os.environ, "WANDB_MODE": "disabled"}
        with (folder / "boltz.log").open("a") as log:
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        affinity_path = folder / f"boltz_results_{job}" / "predictions" / job / f"affinity_{job}.json"
        cif_files = list(affinity_path.parent.glob(f"{job}_model_*.cif"))
        if process.returncode == 0 and affinity_path.exists() and len(cif_files) == config["diffusion_samples"]:
            entry.update(status="measured", affinity_json=str(affinity_path),
                         binding_conditions=binding_conditions(json.loads(affinity_path.read_text()), config))
            entry["full_benchmark_success"] = entry["binding_conditions"]["all_passed"]
        else:
            entry.update(status="unmeasured", returncode=process.returncode, log=str(folder / "boltz.log"),
                         structure_samples_found=len(cif_files), full_benchmark_success=None)
        results.append(entry)
        write_json(args.output / "results.json", results)
        print(json.dumps({"event": "boltz_finished", "task_id": record["task_id"], "status": entry["status"],
                          "full_success": entry["full_benchmark_success"]}), flush=True)
    summary = {}
    for split in args.splits:
        for model, item in admet["splits"][split].items():
            matched = [r for r in results if r["split"] == split and r["model"] == model]
            summary.setdefault(split, {})[model] = {
                "rows": item["rows"], "known_failure_before_boltz": item["full_success_known_false"],
                "full_success": sum(r["full_benchmark_success"] is True for r in matched),
                "full_failure": item["full_success_known_false"] + sum(r["full_benchmark_success"] is False for r in matched),
                "unmeasured": sum(r["full_benchmark_success"] is None for r in matched)}
    complete = all(r["status"] == "measured" for r in results)
    report = {"status": "complete" if complete else "partial", "splits": summary, "results": results,
              "seconds": time.monotonic() - started, "used_for_checkpoint_selection": False,
              "packages": {p: importlib.metadata.version(p) for p in ["boltz", "torch", "rdkit", "numpy"]},
              "memory_fraction": args.memory_fraction, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "limitations": ["These are predicted oracle properties, not experimental binding measurements.",
                              "Train survivors are outside this held-out validation/test binding evaluation.",
                              "No optional CUDA extension kernels; same sample count, seed and diffusion steps as SMDD."]}
    write_json(args.output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
