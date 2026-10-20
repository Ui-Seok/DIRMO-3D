from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_cases(path: str):
    from .schemas import Case
    source = Path(path).resolve()
    text = source.read_text()
    raw = [json.loads(line) for line in text.splitlines() if line.strip()] if source.suffix == ".jsonl" else json.loads(text)
    raw = raw if isinstance(raw, list) else [raw]
    cases = []
    for entry in raw:
        for key in ("protein_path", "pocket_path", "initial_ligand_path", "receptor_path"):
            if entry.get(key):
                candidate = Path(entry[key])
                entry[key] = str((source.parent / candidate).resolve() if not candidate.is_absolute() else candidate)
        cases.append(Case.model_validate(entry))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Duplicate manifest case IDs")
    return cases


def policy(name, artifact=None, cost_weight=None):
    from .routing.learned import LearnedRouter
    from .routing.policies import FixedActionRouter, FixedSequenceRouter, LLMRouterBaseline, RuleRouter
    from .schemas import ActionKind
    if name == "learned":
        if not artifact:
            raise ValueError("--router-model is required for learned routing")
        if cost_weight is None:
            return LearnedRouter(artifact)
        return LearnedRouter(artifact, cost_weight)
    return {"rule": RuleRouter(), "llm": LLMRouterBaseline(), "sequence": FixedSequenceRouter(),
            "edit": FixedActionRouter(ActionKind.CHEMICAL_EDIT),
            "pose": FixedActionRouter(ActionKind.POSE_RESAMPLE)}[name]


def main(argv=None):
    # CPU orchestrator mask is established before importing RDKit/evaluation/training.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    parser = argparse.ArgumentParser(description="DIRMO: bounded diagnosis-guided molecular optimization")
    parser.add_argument("--config", default="configs/local.json")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--output")
    start = commands.add_parser("start-workers")
    start.add_argument("--only", choices=["llm", "fm"])
    start.add_argument("--timeout", type=float, default=300)
    stop = commands.add_parser("stop-workers")
    stop.add_argument("--only", choices=["llm", "fm"])
    commands.add_parser("health")
    for command in ("evaluate-initial", "optimize", "collect", "compare"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", required=True)
        sub.add_argument("--output", required=True)
        if command in ("optimize", "evaluate-initial"):
            sub.add_argument("--resume", action="store_true")
        if command == "optimize":
            sub.add_argument("--router", choices=["rule", "learned", "llm", "sequence", "edit", "pose"], default="rule")
            sub.add_argument("--router-model")
            sub.add_argument("--router-cost-weight", type=float,
                             help="Learned routing only: utility charged per wall-clock second. "
                                  "Must be set against the objective in use; see LearnedRouter.")
        if command == "collect":
            sub.add_argument("--dataset", required=True)
            sub.add_argument("--seed-offset", type=int, default=0,
                             help="Nonnegative offset from config.seed for a new collection; workers stay loaded")
            sub.add_argument("--regions", type=int, default=0,
                             help="Also offer N REDESIGN_REGION actions: free a substituent and let "
                                  "the generator choose what goes there")
            sub.add_argument("--oracle-edits", type=int, default=0,
                             help="Also run the top-N enumerated QED-improving edits beside the planner's "
                                  "candidates, so edit value and planner quality can be told apart")
        if command == "compare":
            sub.add_argument("--policies", nargs="+", choices=["rule", "learned", "llm", "sequence", "edit", "pose"],
                             default=["rule", "edit", "pose", "sequence"])
            sub.add_argument("--router-model")
            sub.add_argument("--router-cost-weight", type=float,
                             help="Learned routing only: utility charged per wall-clock second.")
    train = commands.add_parser("train-router")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--kernel-id")
    train.add_argument("--group-unit", choices=["target", "series"], default="target")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--device", choices=["fm", "cpu"], default="fm")
    args = parser.parse_args(argv)
    if args.command == "collect" and args.seed_offset < 0:
        parser.error("--seed-offset must be nonnegative")
    if args.command == "collect" and not 0 <= args.oracle_edits <= 16:
        parser.error("--oracle-edits must be between 0 and 16")
    if args.command == "collect" and not 0 <= args.regions <= 8:
        parser.error("--regions must be between 0 and 8")
    from .config import load_config
    from .io import write_json
    config = load_config(args.config)
    try:
        if args.command == "doctor":
            from .doctor import doctor
            result = doctor(config)
            if args.output:
                write_json(args.output, result)
        elif args.command in ("start-workers", "stop-workers", "health"):
            from .services import clients, start_workers, stop_workers
            if args.command == "start-workers":
                result = start_workers(config, args.only, args.timeout)
            elif args.command == "stop-workers":
                result = stop_workers(config, args.only)
            else:
                result = {role: client.health() for role, client in clients(config).items()}
        elif args.command == "train-router":
            from .services import stop_workers
            from .training.router import train_router
            # Free the FM GPU before gate fitting. The deployed gate always infers on CPU.
            stop_workers(config, only="fm")
            if args.device == "cpu":
                result = train_router(args.dataset, args.output, args.kernel_id, args.group_unit, args.epochs, config.seed)
            else:
                import subprocess
                from .gpu import inventory, resolve_device, worker_environment
                if not config.fm.python or not Path(config.fm.python).is_file():
                    raise ValueError("Set fm.python before training on the FM GPU, or choose --device cpu")
                device = resolve_device(config.fm.gpu, inventory())
                if device["total_mib"] - device["used_mib"] < 1024:
                    raise RuntimeError("FM GPU has less than 1 GiB available for gate training")
                env = worker_environment(device, str(Path(__file__).resolve().parents[1]))
                command = [config.fm.python, "-m", "dirmo.training.worker", "--dataset", args.dataset,
                           "--output", args.output, "--group-unit", args.group_unit, "--epochs", str(args.epochs),
                           "--seed", str(config.seed)]
                if args.kernel_id:
                    command.extend(["--kernel-id", args.kernel_id])
                process = subprocess.run(command, env=env, text=True, capture_output=True, timeout=600)
                if process.returncode:
                    raise RuntimeError(f"Gate worker failed: {process.stderr[-2000:]}")
                result = json.loads(process.stdout)
        else:
            from .pipeline.run import Run, optimize
            from .services import clients
            cases = load_cases(args.manifest)
            if args.command != "evaluate-initial":
                active = clients(config)
                if "llm" not in active or "fm" not in active:
                    raise ValueError("Both workers must be healthy before optimization/collection/comparison")
            results = []
            # A sweep over many cases must not lose hours of measured work to one bad case,
            # and a restart must not refuse the output directories it already filled.
            marker = {"collect": "branch_coverage.json", "optimize": "result.json",
                      "compare": "comparison.json", "evaluate-initial": "initial_evaluation.json"}[args.command]
            consecutive = 0
            for case in cases:
                output = Path(args.output) / case.case_id
                finished = output / marker
                if finished.is_file():
                    results.append({"case_id": case.case_id, "output": str(output.resolve()),
                                    "status": "already_complete",
                                    "result": json.loads(finished.read_text())})
                    continue
                try:
                    if args.command == "compare":
                        comparisons = []
                        for name in args.policies:
                            run = Run(case, config, output / name)
                            result = optimize(run, active["fm"], active["llm"],
                                              policy(name, args.router_model, args.router_cost_weight))
                            comparisons.append({"policy": name, "result": result,
                                                "router_cost_weight": args.router_cost_weight
                                                if name == "learned" else None})
                        result = {"case_id": case.case_id, "policies": comparisons,
                                  "protocol": "same frozen executor/objective/seed/caps; measured actual costs reported",
                                  "limitation": "One case/seed is not evidence of policy superiority"}
                        write_json(output / "comparison.json", result)
                    else:
                        run = Run(case, config, output, resume=getattr(args, "resume", False))
                        if args.command == "evaluate-initial":
                            result = run.state.original_ligand.observation.model_dump(mode="json")
                            write_json(output / "initial_evaluation.json", result)
                        elif args.command == "collect":
                            from .data.collect import collect
                            result = collect(run, active["fm"], active["llm"], args.dataset,
                                                 args.seed_offset, args.oracle_edits, args.regions)
                        else:
                            router = policy(args.router, args.router_model, args.router_cost_weight)
                            result = optimize(run, active["fm"], active["llm"], router)
                            result["routing"] = {"policy": args.router, "router_model": args.router_model,
                                                 "cost_weight": getattr(router, "cost_weight", None)}
                            write_json(output / "result.json", result)
                    results.append({"case_id": case.case_id, "output": str(output.resolve()),
                                    "status": "ok", "result": result})
                    consecutive = 0
                except Exception as exc:
                    # Recorded, never retried silently. A systematic failure (a dead worker,
                    # say) shows up as a run of these, so stop rather than burn the budget.
                    results.append({"case_id": case.case_id, "output": str(output.resolve()),
                                    "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
                    consecutive += 1
                    if consecutive >= 5:
                        results.append({"status": "aborted", "reason": "5 consecutive case failures; "
                                        "inspect the worker logs before resuming"})
                        break
            result = results
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "blocked", "reason": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    return 0
