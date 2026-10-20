#!/usr/bin/env python3
"""Compare independent proposals and tool-feedback editing on validation only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from dirmo.data.smdd import json_text
from dirmo.evaluation.smdd_admet import DATASET_PROPERTIES
from dirmo.evaluation.smdd_feedback import (
    assess_plan, attach_admet, messages, next_state, select_attempt, tool_feedback,
)
from dirmo.io import write_json
from dirmo.services import WorkerClient


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CpuAdmet:
    def __init__(self, output):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Run with CUDA_VISIBLE_DEVICES= (ADMET must remain on CPU)")
        if importlib.metadata.version("admet_ai") != "1.4.0":
            raise ValueError("Use the isolated .venv-admet with ADMET-AI 1.4.0")
        from admet_ai import ADMETModel
        from admet_ai.constants import DEFAULT_MODELS_DIR
        self.model = ADMETModel(num_workers=0, drugbank_path=None,
                                fingerprint_multiprocessing_min=10**9)
        if self.model.device.type != "cpu":
            raise ValueError("ADMET model unexpectedly uses CUDA")
        self.path = output / "admet_cache.json"
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.identity = {
            "packages": {p: importlib.metadata.version(p) for p in
                         ["admet_ai", "chemprop", "torch", "rdkit", "numpy"]},
            "models_sha256": {str(p.relative_to(DEFAULT_MODELS_DIR)): sha(p)
                              for p in Path(DEFAULT_MODELS_DIR).rglob("*.pt")},
            "device": "cpu", "smiles_policy": "Exact raw output token; no canonicalization before ADMET",
        }
        identity_path = output / "admet_identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != self.identity:
            raise ValueError("ADMET environment or weights changed; use a new output directory")
        write_json(identity_path, self.identity)

    def predict_many(self, smiles):
        missing = sorted(set(smiles) - set(self.cache))
        if missing:
            predicted = self.model.predict(smiles=missing)
            for smi in missing:
                self.cache[smi] = {prop: float(predicted.loc[smi, dataset])
                                   for dataset, prop in DATASET_PROPERTIES.items()}
            write_json(self.path, self.cache)
        return {smi: self.cache[smi] for smi in smiles}


def append(path, entry):
    with path.open("a") as handle:
        handle.write(json_text(entry) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_report(rows, attempts, output, calls, started):
    result = {"status": "rdkit_admet_complete", "split": "validation", "cases": len(rows),
              "actual_llm_requests": len(calls),
              "tokens": {key: sum(r["response"].get(key) or 0 for r in calls.values())
                         for key in ["input_tokens", "output_tokens", "total_tokens"]},
              "generation_seconds": sum(r["response"].get("latency_seconds") or 0 for r in calls.values()),
              "invocation_seconds": time.monotonic() - started,
              "comparison": {}, "selected": [], "limitations": [
                  "No weight updates, adapters, encoder, or witness-derived prompt information.",
                  "Each arm has the same proposal budget; round 1 is shared, not two model requests.",
                  "Independent proposals change only proposal_number to vary deterministic-seed prompts.",
                  "Feedback arm changes both current substrate and actual tool feedback; this is a bundled intervention.",
                  "Edit hypotheses and declared atom locations are not independently verified causal explanations.",
                  "RDKit/ADMET pass is only a pre-Boltz gate, not full optimization success.",
                  "ADMET and Boltz are predictions, not experimental measurements.",
              ]}
    pending = {}
    for mode in ["independent", "feedback"]:
        selected = []
        for row in rows:
            items = [r for r in attempts if r["mode"] == mode and r["task_id"] == row["task_id"]]
            chosen = select_attempt(items)
            if chosen is None:
                raise ValueError(f"Missing attempts for {row['task_id']}/{mode}")
            a = chosen["assessment"]
            entry = {"task_id": row["task_id"], "mode": mode, "round": chosen["round"],
                     "assessment": a, "attempts": len(items)}
            selected.append(entry)
            if a.get("pre_boltz_passed"):
                key = hashlib.sha256(json_text([row["source"], a["oracle_smiles"]]).encode()).hexdigest()[:20]
                job = pending.setdefault(key, {"job_id": key, "task_id": row["task_id"],
                    "source": row["source"], "product_smiles": a["oracle_smiles"], "selections": []})
                job["selections"].append({"mode": mode, "round": chosen["round"], "task_id": row["task_id"]})
        mode_attempts = [r for r in attempts if r["mode"] == mode]
        result["comparison"][mode] = {
            "cases": len(rows), "budgeted_proposals": len(mode_attempts),
            "any_valid_graph": sum(any(i["assessment"].get("smiles_valid") for i in mode_attempts
                                       if i["task_id"] == r["task_id"]) for r in rows),
            "any_changed_local_pass": sum(any(i["assessment"].get("changed") and
                i["assessment"].get("local_hard_constraints_passed") for i in mode_attempts
                if i["task_id"] == r["task_id"]) for r in rows),
            "selected_pre_boltz_pass": sum(s["assessment"].get("pre_boltz_passed", False) for s in selected),
            "first_proposal_pre_boltz_pass": sum(i["assessment"].get("pre_boltz_passed", False)
                                                for i in mode_attempts if i["round"] == 1),
            "plan_valid_attempts": sum(i["assessment"]["plan_valid"] for i in mode_attempts),
            "valid_graph_attempts": sum(bool(i["assessment"].get("smiles_valid")) for i in mode_attempts),
            "changed_local_pass_attempts": sum(bool(i["assessment"].get("changed") and
                 i["assessment"].get("local_hard_constraints_passed")) for i in mode_attempts),
            "admet_evaluated_attempts": sum("admet_conditions" in i["assessment"] for i in mode_attempts),
            "full_success_confirmed": 0,
            "full_failure_confirmed": sum(s["assessment"]["full_benchmark_success"] is False for s in selected),
            "unmeasured_binding": sum(s["assessment"]["full_benchmark_success"] is None for s in selected),
        }
        result["selected"].extend(selected)
    write_json(output / "pending_boltz.json", list(pending.values()))
    result["unique_boltz_jobs"] = len(pending)
    if not pending:
        result["status"] = "complete"
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("runs/smdd_lora_02/data/validation.jsonl"))
    parser.add_argument("--runtime-dir", type=Path, default=Path("worker-runtime"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Optional first N validation cases for debugging")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.dataset.name != "validation.jsonl":
        raise ValueError("This development experiment accepts validation.jsonl only; test remains untouched")
    if not 1 <= args.rounds <= 10 or args.timeout <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("Invalid rounds, timeout, or limit")
    rows = read_lines(args.dataset)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows or len({r["task_id"] for r in rows}) != len(rows):
        raise ValueError("Dataset is empty or contains duplicate tasks")
    if args.output.exists() and not args.resume:
        raise FileExistsError("Use a new output directory or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.runtime_dir / "workers.json").read_text())
    worker = manifest["workers"]["llm"]
    client = WorkerClient(worker["port"], worker["token"])
    health = client.health()
    if health.get("role") != "llm" or health.get("instance") != worker["instance"]:
        raise ValueError("LLM server identity does not match runtime manifest")
    if health["model_identity"].get("thinking_applied"):
        raise ValueError("Use the non-thinking frozen base worker for short structured edit plans")
    module_path = Path(__import__("dirmo.evaluation.smdd_feedback", fromlist=["__file__"]).__file__)
    protocol = {"dataset": str(args.dataset.resolve()), "dataset_sha256": sha(args.dataset),
                "task_ids": [r["task_id"] for r in rows], "rounds": args.rounds,
                "model_id": health["model_identity"]["model_id"],
                "server_config_sha256": sha(args.runtime_dir / "config.json"),
                "runner_sha256": sha(Path(__file__)), "feedback_module_sha256": sha(module_path),
                "selection": "full success, pre-Boltz pass, local pass, valid graph, changed, fewer ADMET failures, earlier round"}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Protocol changed; use a new output directory")
    write_json(protocol_path, protocol)
    write_json(args.output / "server_identity.json", health)
    started = time.monotonic()
    oracle = CpuAdmet(args.output)
    refs = oracle.predict_many([r["reference_smiles"] for r in rows])
    # Audit differences; do not replace official baseline targets.
    write_json(args.output / "reference_audit.json", [
        {"task_id": r["task_id"], "reprediction": refs[r["reference_smiles"]],
         "delta_from_published": {c["property"]: refs[r["reference_smiles"]][c["property"]] - c["baseline"]
                                  for c in r["goals"] + r["hold"] if c["property"] != "binding_affinity"}}
        for r in rows])
    calls_path, attempts_path = args.output / "llm_calls.jsonl", args.output / "attempts.jsonl"
    calls = {r["request_id"]: r for r in read_lines(calls_path)}
    attempts = read_lines(attempts_path)
    completed = {(r["task_id"], r["mode"], r["round"]): r for r in attempts}
    pending_path = args.output / "pending_request.json"
    if pending_path.exists() and json.loads(pending_path.read_text())["request_id"] not in calls:
        raise ValueError("A previous HTTP request has unknown outcome; do not silently repeat it")
    for row in rows:
        current, current_admet, feedback, tried = row["reference_smiles"], refs[row["reference_smiles"]], None, []
        for round_number in range(1, args.rounds + 1):
            for mode in ["independent", "feedback"]:
                key = (row["task_id"], mode, round_number)
                if key in completed:
                    record = completed[key]
                else:
                    substrate = current if mode == "feedback" else row["reference_smiles"]
                    prompt = messages(row, substrate, refs[row["reference_smiles"]],
                                      current_admet if mode == "feedback" else refs[row["reference_smiles"]],
                                      round_number, feedback if mode == "feedback" else None)
                    arm = "shared" if round_number == 1 else mode
                    request_id = f"{row['task_id']}:{arm}:{round_number}"
                    prompt_sha = hashlib.sha256(json_text(prompt).encode()).hexdigest()
                    if request_id not in calls:
                        write_json(pending_path, {"request_id": request_id, "prompt_sha256": prompt_sha})
                        response = client.request("/propose", {"messages": prompt}, timeout=args.timeout)
                        if response.get("model_identity", {}).get("model_id", protocol["model_id"]) != protocol["model_id"]:
                            raise ValueError("Worker model changed during experiment")
                        record_call = {"request_id": request_id, "messages": prompt, "prompt_sha256": prompt_sha,
                                       "response": {k: v for k, v in response.items() if k not in ["model_identity", "generation"]}}
                        append(calls_path, record_call)
                        calls[request_id] = record_call
                        pending_path.unlink()
                    elif calls[request_id]["prompt_sha256"] != prompt_sha:
                        raise ValueError("Prompt differs from saved response; use a new run")
                    response = calls[request_id]["response"]
                    assessment = assess_plan(response.get("raw", ""), substrate, row["reference_smiles"])
                    if assessment.get("changed") and assessment.get("local_hard_constraints_passed"):
                        try:
                            vals = oracle.predict_many([assessment["oracle_smiles"]])[assessment["oracle_smiles"]]
                            attach_admet(assessment, row, vals)
                        except (ValueError, RuntimeError, KeyError) as exc:
                            # Leave the response journal intact for a tool-only
                            # resume. Missing tool output is not a measured fail.
                            raise RuntimeError(f"ADMET failed for {request_id}; resume after fixing the oracle") from exc
                    record = {"task_id": row["task_id"], "mode": mode, "round": round_number,
                              "request_id": request_id, "current_smiles": substrate,
                              "assessment": assessment, "server_status": response.get("status"),
                              "first_round_shared": round_number == 1}
                    append(attempts_path, record)
                    attempts.append(record)
                    completed[key] = record
                    print(json_text({"event": "attempt", "task_id": row["task_id"], "mode": mode,
                        "round": round_number, "plan_valid": assessment["plan_valid"],
                        "local_pass": assessment.get("local_hard_constraints_passed"),
                        "pre_boltz_pass": assessment["pre_boltz_passed"],
                        "failed": tool_feedback(assessment, [])['failed_conditions']}), flush=True)
                if mode == "feedback":
                    assessed = record["assessment"]
                    if assessed.get("oracle_smiles"):
                        tried.append(assessed["oracle_smiles"])
                    feedback = tool_feedback(assessed, list(tried))
                    current = next_state(row["reference_smiles"], assessed)
                    current_admet = (assessed.get("admet_properties") if current != row["reference_smiles"]
                                     else refs[row["reference_smiles"]])
    report = make_report(rows, attempts, args.output, calls, started)
    print(json_text({k: v for k, v in report.items() if k != "selected"}), flush=True)


if __name__ == "__main__":
    main()
