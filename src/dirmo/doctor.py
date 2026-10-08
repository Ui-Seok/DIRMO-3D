from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .generation.flowr import audit_paths
from .gpu import inventory, resolve_device
from .io import environment, revision, sha256_file


def interpreter_probe(path: str | None, packages: list[str]) -> dict:
    if not path or not Path(path).is_file():
        return {"status": "missing", "python": path}
    code = ("import sys,json,importlib.metadata as m; "
            "print(json.dumps({'python':sys.version,'packages':{p:"
            "next((d.version for d in m.distributions(name=p)),None) for p in " + repr(packages) + "}}))")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([path, "-c", code], env=env, text=True, capture_output=True, timeout=30)
    return {"status": "ok" if result.returncode == 0 else "failed", "interpreter": path,
            "details": json.loads(result.stdout) if result.returncode == 0 else result.stderr}


def doctor(config) -> dict:
    result = {"environment": environment(), "llm": {"path": config.llm.model_path,
                  "exists": bool(config.llm.model_path and Path(config.llm.model_path).is_dir()),
                  "revision": revision(config.llm.model_path),
                  "environment": interpreter_probe(config.llm.python, ["torch", "transformers", "accelerate"])},
              "fm": {"capabilities": audit_paths(config.fm),
                     "environment": interpreter_probe(config.fm.python, ["torch", "lightning", "rdkit", "tensordict"])}}
    if config.fm.checkpoint_path and Path(config.fm.checkpoint_path).is_file():
        result["fm"]["checkpoint_sha256"] = sha256_file(config.fm.checkpoint_path)
    try:
        devices = inventory()
        result["gpu_inventory"] = devices
        mapping = {role: resolve_device(getattr(config, role).gpu, devices) for role in ("llm", "fm")}
        if mapping["llm"]["uuid"] == mapping["fm"]["uuid"]:
            raise ValueError("GPU selectors resolve to the same device")
        result["gpu_mapping"] = mapping
    except Exception as exc:
        result["gpu_status"] = {"status": "unavailable", "reason": str(exc)}
    llm_environment = result["llm"]["environment"]
    fm_environment = result["fm"]["environment"]
    llm_packages = llm_environment.get("details", {}).get("packages", {}) if llm_environment["status"] == "ok" else {}
    fm_details = fm_environment.get("details", {}) if fm_environment["status"] == "ok" else {}
    result["ready_for_two_worker_inference"] = bool(result["llm"]["exists"] and
          llm_packages.get("torch") and llm_packages.get("transformers") and
          fm_details.get("python", "").startswith("3.12.") and fm_details.get("packages", {}).get("torch") and
          not result["fm"]["capabilities"]["blockers"] and "gpu_mapping" in result)
    return result
