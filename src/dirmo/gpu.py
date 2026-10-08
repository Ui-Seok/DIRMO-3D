from __future__ import annotations

import csv
import io
import os
import subprocess


def inventory() -> list[dict]:
    out = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"
    ], text=True, timeout=15)
    return [{"index": int(row[0]), "uuid": row[1].strip(), "name": row[2].strip(),
             "total_mib": int(row[3]), "used_mib": int(row[4])} for row in csv.reader(io.StringIO(out))]


def resolve_device(selector: str, devices: list[dict]) -> dict:
    matches = [d for d in devices if selector in (str(d["index"]), d["uuid"])]
    if len(matches) != 1:
        raise ValueError(f"GPU {selector!r} does not identify one physical device")
    device = matches[0]
    if "A6000" not in device["name"] or device["total_mib"] < 47000:
        raise ValueError("Only the authorized A6000 48GB class is supported")
    return device


def worker_environment(device: dict, source_path: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=device["uuid"], DIRMO_GPU_UUID=device["uuid"],
               DIRMO_PHYSICAL_GPU=str(device["index"]), TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1", FLOWR_DEVICE="cuda:0")
    env["PYTHONPATH"] = source_path
    return env


def assert_worker_mask():
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not mask or "," in mask or mask != os.environ.get("DIRMO_GPU_UUID"):
        raise RuntimeError("Worker must receive exactly one resolved GPU UUID before importing torch")
