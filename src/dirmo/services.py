from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .generation.flowr import audit_paths
from .gpu import inventory, resolve_device, worker_environment
from .io import digest, write_json


class WorkerClient:
    def __init__(self, port: int, token: str):
        self.url, self.token = f"http://127.0.0.1:{port}", token

    def request(self, route: str, payload: dict | None = None, timeout: float = 10) -> dict:
        req = urllib.request.Request(self.url + route,
                data=json.dumps(payload).encode() if payload is not None else None,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with urllib.request.urlopen(req, timeout=max(0.01, timeout)) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Worker HTTP {exc.code}: {exc.read().decode()}") from exc
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def health(self):
        return self.request("/health")


def clients(config):
    manifest = json.loads((Path(config.runtime_dir) / "workers.json").read_text())
    if manifest["config_hash"] != digest(config):
        raise ValueError("Running workers use a different resolved config")
    return {role: WorkerClient(record["port"], record["token"]) for role, record in manifest["workers"].items()}


def owned_process(record: dict) -> bool:
    try:
        argv = Path(f"/proc/{record['pid']}/cmdline").read_bytes().decode().split("\0")
        return "dirmo.worker" in argv and record["instance"] in argv
    except (OSError, KeyError):
        return False


def start_workers(config, only: str | None = None, timeout: float = 300) -> dict:
    roles = [only] if only else ["llm", "fm"]
    if "fm" in roles and audit_paths(config.fm)["blockers"]:
        raise RuntimeError("; ".join(audit_paths(config.fm)["blockers"]))
    devices = inventory()
    chosen = {role: resolve_device(getattr(config, role).gpu, devices) for role in roles}
    if not only and chosen["llm"]["uuid"] == chosen["fm"]["uuid"]:
        raise ValueError("Physical GPU selectors resolve to the same GPU")
    for role in roles:
        cfg, device = getattr(config, role), chosen[role]
        if not cfg.python or not Path(cfg.python).is_file():
            raise FileNotFoundError(f"Set {role}.python to a separate compatible Python interpreter")
        if device["total_mib"] - device["used_mib"] < cfg.minimum_free_mib:
            raise RuntimeError(f"Insufficient available VRAM for {role}: {device}")
    directory = Path(config.runtime_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "workers.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(owned_process(r) for r in old["workers"].values()):
            raise RuntimeError("Workers already active; stop them before starting another instance")
    resolved = directory / "config.json"
    write_json(resolved, config)
    manifest = {"config_hash": digest(config), "workers": {}}
    source = str(Path(__file__).resolve().parents[1])
    try:
        for role in roles:
            cfg = getattr(config, role)
            token, instance = secrets.token_hex(24), secrets.token_hex(12)
            env = worker_environment(chosen[role], source)
            env["DIRMO_HTTP_TOKEN"] = token
            log_path = directory / f"{role}.log"
            with log_path.open("w") as log:
                process = subprocess.Popen([cfg.python, "-m", "dirmo.worker", "--role", role,
                      "--config", str(resolved), "--instance", instance], env=env,
                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=directory)
            record = {"pid": process.pid, "port": cfg.port, "token": token, "instance": instance,
                      "gpu": chosen[role], "log": str(log_path)}
            manifest["workers"][role] = record
            write_json(manifest_path, manifest)
            manifest_path.chmod(0o600)
        deadline = time.monotonic() + timeout
        pending = set(roles)
        while pending and time.monotonic() < deadline:
            for role in list(pending):
                record = manifest["workers"][role]
                if not owned_process(record):
                    raise RuntimeError(f"{role} worker exited. Inspect {record['log']}")
                try:
                    record["health"] = WorkerClient(record["port"], record["token"]).health()
                    if record["health"]["actual_gpu_uuid"] != chosen[role]["uuid"]:
                        raise RuntimeError("Worker actual UUID differs from resolved physical device")
                    pending.remove(role)
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    pass
            if pending:
                time.sleep(0.5)
        if pending:
            raise TimeoutError(f"Worker load timed out: {pending}")
        write_json(manifest_path, manifest)
        manifest_path.chmod(0o600)
        return {role: {k: v for k, v in record.items() if k != "token"}
                for role, record in manifest["workers"].items()}
    except Exception:
        stop_workers(config)
        raise


def stop_workers(config, only: str | None = None) -> dict:
    manifest_path = Path(config.runtime_dir) / "workers.json"
    if not manifest_path.exists():
        return {"stopped": [], "reason": "No managed worker manifest"}
    manifest = json.loads(manifest_path.read_text())
    stopped = []
    for role, record in manifest["workers"].items():
        if only and role != only:
            continue
        if owned_process(record):
            try:
                WorkerClient(record["port"], record["token"]).request("/shutdown", {}, timeout=2)
            except Exception:
                pass
            deadline = time.monotonic() + 5
            while owned_process(record) and time.monotonic() < deadline:
                time.sleep(0.1)
            if owned_process(record):
                os.kill(record["pid"], signal.SIGTERM)
                deadline = time.monotonic() + 5
                while owned_process(record) and time.monotonic() < deadline:
                    time.sleep(0.1)
            if owned_process(record):
                os.kill(record["pid"], signal.SIGKILL)
        stopped.append(role)
    manifest["workers"] = {k: v for k, v in manifest["workers"].items() if k not in stopped}
    write_json(manifest_path, manifest)
    return {"stopped": stopped}
