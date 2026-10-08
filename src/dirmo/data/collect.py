from __future__ import annotations

import json
from pathlib import Path

from ..io import append_jsonl, digest, write_json
from ..llm.proposals import Planner
from ..pipeline.budget import BudgetExhausted
from ..pipeline.run import Executor, Run
from ..routing.features import FEATURE_NAMES, features
from ..routing.policies import available_actions
from ..schemas import ActionKind


def collect(run: Run, fm_client, llm_client, dataset: str) -> dict:
    executor = Executor(run, fm_client)
    planner = Planner(llm_client, run.config.llm, run.directory, run.budget,
                      persist=lambda event: run.save(pending=event))
    run.state.proposal = planner.propose(run.state, run.case)
    run.save()
    run.refresh()
    snapshot = run.state.model_copy(deep=True)
    snapshot_path = run.directory / "branch_snapshot.json"
    write_json(snapshot_path, snapshot)
    actions = [a for a in available_actions(snapshot, run.case, run.config.fm.parameters,
                            executor.capabilities, executor.nfe_factor, run.config.seed) if a.kind != ActionKind.STOP]
    coverage = []
    for action in actions:
        # Each branch starts from the same actual pre-action measurements and proposal.
        # Only the global compute ledger changes as branches consume resources.
        run.state = snapshot.model_copy(deep=True)
        run.refresh()
        feature_state = snapshot.model_copy(deep=True)
        input_features = features(feature_state, action, run.case)
        try:
            # Feasible action seed IDs depend on graph history, which is identical in each branch.
            before_cost = run.budget.used.copy()
            before_wall = run.budget.elapsed
            outcomes = executor.execute(action)
        except BudgetExhausted as exc:
            coverage.append({"action_id": action.action_id, "status": "not_executed", "reason": str(exc)})
            continue
        except ValueError as exc:
            coverage.append({"action_id": action.action_id, "status": "not_executed", "reason": str(exc)})
            continue
        for outcome in outcomes:
            after = run.state.work_item.observation
            before = snapshot.work_item.observation
            # Failed or incomplete utility remains null; it is not an invented reward.
            delta = after.utility - before.utility if (after.utility is not None and before.utility is not None) else None
            costs = {k: run.budget.used[k] - before_cost[k] for k in run.budget.resources}
            costs["wall_seconds"] = run.budget.elapsed - before_wall
            record = {**outcome, "pre_state_id": snapshot.state_id, "snapshot_hash": digest(snapshot),
                      "snapshot_path": str(snapshot_path), "group_id": run.case.target_id,
                      "feature_names": FEATURE_NAMES, "pre_action_features": input_features,
                      "utility_change": delta, "compute_cost": costs,
                      "model_identity": json.loads((run.directory / "fm_health.json").read_text())["model_identity"],
                      "integration_steps": action.generation.integration_steps,
                      "protocol": "common snapshot; common seed for alternatives; global finite branch budget"}
            append_jsonl(dataset, record)
        coverage.append({"action_id": action.action_id, "status": "executed", "draws": len(outcomes)})
    run.state = snapshot
    run.save()
    summary = {"pre_state_id": snapshot.state_id, "available_actions": len(actions),
               "executed_actions": sum(r["status"] == "executed" for r in coverage), "coverage": coverage,
               "dataset": str(Path(dataset).resolve()), "performance_claim": None}
    write_json(run.directory / "branch_coverage.json", summary)
    return summary
