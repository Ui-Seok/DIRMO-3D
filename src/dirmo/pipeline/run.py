from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..chemistry.graphs import (execute_edit, graph_check, graph_id, input_conformer, mapped,
                               read_sdf, smiles, write_sdf)
from ..chemistry.properties import graph_properties
from ..diagnosis import assemble_evidence, wilson
from ..evaluation.metrics import Evaluator
from ..evaluation.pocket import prepare_fixed_pocket
from ..io import append_jsonl, digest, environment, revision, sha256_file, write_json
from ..llm.proposals import Planner
from ..routing.learned import LearnedRouter
from ..routing.policies import available_actions
from ..schemas import (ActionKind, EvaluationRecord, GenerationRequest, GenerationResult, Observation,
                       State, WorkItem)
from .budget import Budget, BudgetExhausted


def input_identity(case, config):
    checksums = {name: sha256_file(path) for name in ("initial_ligand_path", "protein_path", "pocket_path",
                 "receptor_path") if (path := getattr(case, name))}
    source = Path(__file__).resolve().parents[1]
    implementation = {str(path.relative_to(source)): sha256_file(path) for path in sorted(source.rglob("*.py"))}
    return {"case": case.model_dump(mode="json"), "config": config.model_dump(mode="json"),
            "implementation_sha256": digest(implementation),
            "input_sha256": checksums, "flowr_revision": revision(config.fm.repo_path),
            "llm_revision": revision(config.llm.model_path), "environment": environment(),
            "checkpoint_sha256": sha256_file(config.fm.checkpoint_path) if config.fm.checkpoint_path else None}


class Run:
    def __init__(self, case, config, directory: str | Path, resume: bool = False):
        self.case = case.model_copy(deep=True)
        self.config, self.directory = config, Path(directory).resolve()
        original = mapped(read_sdf(case.initial_ligand_path))
        if self.case.box_center is None:
            self.case.box_center = tuple(float(v) for v in original.GetConformer().GetPositions().mean(0))
        identity = input_identity(self.case, config)
        self.identity_hash = digest(identity)
        self.fixed_pocket = str(self.directory / "inputs" / "fixed_pocket.pdb")
        self.pending = None
        manifest_path = self.directory / "manifest.json"
        if resume:
            manifest = json.loads(manifest_path.read_text())
            if manifest["identity_hash"] != self.identity_hash:
                raise ValueError("Resume input/config/model/environment fingerprint differs")
            checkpoint = json.loads((self.directory / "checkpoint.json").read_text())
            if digest(checkpoint["state"]) != checkpoint["state_hash"]:
                raise ValueError("Resume state hash mismatch")
            for path, checksum in checkpoint["artifact_sha256"].items():
                if sha256_file(path) != checksum:
                    raise ValueError(f"Resume artifact changed: {path}")
            self.state = State.model_validate(checkpoint["state"])
            self.pending = checkpoint["pending_action"]
            if self.pending:
                # Do not automatically replay an intervention whose remote outcome is unknown.
                raise ValueError("Interrupted in-flight action: reservation retained. Inspect pending_action and worker "
                                 "logs; recover the recorded result before resuming. No automatic duplicate draw.")
            self.budget = Budget(self.case.budgets, self.state.budget_used,
                                 wall_used=checkpoint["wall_seconds"])
            self.budget.events = json.loads((self.directory / "costs.json").read_text())["events"]
            self.evaluator = Evaluator(config.evaluation, self.case, self.fixed_pocket, original)
            self.evaluator.baseline = self.state.original_ligand.observation
            append_jsonl(self.directory / "events.jsonl", {"event": "resume", "cache_hit": False,
                                                          "state_hash": checkpoint["state_hash"]})
            return
        if manifest_path.exists():
            raise FileExistsError("Run already exists; use --resume or another output directory")
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, {**identity, "identity_hash": self.identity_hash})
        write_json(self.directory / "resolved_config.json", config)
        initial = str(self.directory / "inputs" / "initial.sdf")
        write_sdf(initial, original)
        prep = prepare_fixed_pocket(case.pocket_path or case.protein_path, initial, self.fixed_pocket,
                    config.evaluation.pocket_cutoff, already_pocket=bool(case.pocket_path))
        write_json(self.directory / "inputs" / "preparation.json", {**prep, "notes": case.preparation_notes,
                    "box_center": self.case.box_center, "box_size": self.case.box_size,
                    "atom_map_convention": "Existing complete maps, otherwise 1-based initial heavy-atom order"})
        self.budget = Budget(case.budgets)
        self.evaluator = Evaluator(config.evaluation, self.case, self.fixed_pocket, original)
        self.budget.charge("evaluate_initial", evaluation_calls=1)
        observation = self.evaluator.evaluate(original)
        self.evaluator.baseline = observation
        item = WorkItem(item_id="original", graph_id=graph_id(original), mapped_smiles=smiles(original, True),
                        sdf_path=initial, atom_mapping={a.GetAtomMapNum(): a.GetAtomMapNum() for a in original.GetAtoms()},
                        observation=observation)
        self.state = State(state_id="initial", case_id=case.case_id, original_ligand=item,
                           best_feasible=item if observation.feasible else None, work_item=item,
                           history=[item], evidence=assemble_evidence(observation))
        append_jsonl(self.directory / "observations.jsonl", observation)
        self.save()

    def refresh(self):
        self.state.budget_used = self.budget.used.copy()
        self.state.remaining_budget = self.budget.remaining()
        payload = self.state.model_dump(mode="json")
        payload.pop("state_id")
        self.state.state_id = digest(payload)

    def save(self, pending: dict | None = None):
        self.pending = pending
        self.refresh()
        artifacts = {self.fixed_pocket, self.state.original_ligand.sdf_path, self.state.work_item.sdf_path,
                     *(item.sdf_path for item in self.state.history)}
        if self.state.best_feasible:
            artifacts.add(self.state.best_feasible.sdf_path)
        state = self.state.model_dump(mode="json")
        write_json(self.directory / "checkpoint.json", {"state": state, "state_hash": digest(state),
                   "artifact_sha256": {p: sha256_file(p) for p in sorted(artifacts)},
                   "wall_seconds": self.budget.elapsed, "pending_action": pending})
        write_json(self.directory / "costs.json", {"used": self.budget.used, "wall_seconds": self.budget.elapsed,
                   "limits": self.case.budgets.model_dump(mode="json"), "events": self.budget.events})

    def incorporate(self, item):
        self.state.work_item = item
        self.state.history.append(item)
        self.state.evidence = assemble_evidence(item.observation)
        before = self.state.best_feasible
        if item.observation.feasible and item.observation.utility is not None and (
                before is None or item.observation.utility > before.observation.utility):
            self.state.best_feasible = item
            self.state.plateau = 0
        else:
            self.state.plateau += 1
        append_jsonl(self.directory / "observations.jsonl", item.observation)
        for hypothesis in self.state.hypotheses:
            if hypothesis.status == "untested":
                hypothesis.status = "uncertain"
            append_jsonl(self.directory / "hypothesis_outcomes.jsonl", {
                "hypothesis_id": hypothesis.hypothesis_id, "statement": hypothesis.statement,
                "planned_test": hypothesis.test, "resulting_observation_id": item.observation.observation_id,
                "observed_failures": item.observation.failures, "status": hypothesis.status,
                "interpretation": "Measured outcome linked for review; no automatic causal confirmation"})

    def finish(self, reason: str) -> dict:
        original = self.state.original_ligand
        candidates = [item for item in self.state.history if item.observation and item.observation.feasible and
                      item.observation.target_met and item.observation.utility is not None]
        best = max(candidates, key=lambda item: item.observation.utility) if candidates else self.state.best_feasible
        self.state.best_feasible = best
        if original.observation.utility is None:
            improved = False  # Cannot claim improvement without a measured baseline.
        else:
            improved = bool(best and best.item_id != "original" and best.observation.target_met and
                            best.observation.utility - original.observation.utility >=
                            self.case.objective.minimum_improvement)
        selected = best if improved else original
        status = "success" if improved else "no_improvement"
        self.state.status = status
        shutil.copyfile(selected.sdf_path, self.directory / "final.sdf")
        result = {"status": status, "termination_reason": reason, "selected": selected.model_dump(mode="json"),
                  "best_feasible": best.model_dump(mode="json") if best else None,
                  "original_feasible": original.observation.feasible,
                  "incomplete_evaluation": selected.observation.incomplete_evaluation,
                  "cost": {**self.budget.used, "wall_seconds": self.budget.elapsed},
                  "graph_recoverability": {gid: wilson(sum(1 for i in self.state.history if i.graph_id == gid and
                       i.source == "fm_raw" and i.observation.feasible), count)
                       for gid, count in self.state.graph_attempts.items()}}
        write_json(self.directory / "result.json", result)
        self.save()
        return result


class Executor:
    def __init__(self, run: Run, fm_client):
        self.run, self.client = run, fm_client
        health = fm_client.health()
        self.capabilities = health.get("capabilities", {})
        self.nfe_factor = health.get("nfe_per_step_upper_bound", 1)
        self.kernel_base = self.capabilities.get("kernel_base", {})
        write_json(run.directory / "fm_health.json", health)

    def kernel_id(self, parameters):
        return digest({**self.kernel_base, "integration_steps": parameters.integration_steps,
                       "pocket_sha256": sha256_file(self.run.fixed_pocket)})

    def execute(self, action, state=None) -> list[dict]:
        run = self.run
        state = state or run.state
        if action.kind == ActionKind.STOP:
            return []
        if action.kind == ActionKind.EXTRA_VALIDATION:
            raise ValueError("EXTRA_VALIDATION is disabled in the first executor")
        run.refresh()
        allowed = available_actions(state, run.case, run.config.fm.parameters,
                                    self.capabilities, self.nfe_factor, run.config.seed)
        valid = next((a for a in allowed if a.action_id == action.action_id), None)
        if valid is None or action != valid:
            raise ValueError("Action is not in the current validated feasible action set")
        pre_state = state.model_dump(mode="json")
        records = []
        if action.kind == ActionKind.CHEMICAL_EDIT:
            parent = read_sdf(state.work_item.sdf_path)
            product, mapping = execute_edit(parent, action.candidate, read_sdf(state.original_ligand.sdf_path),
                                            run.case.protected)
            initial = input_conformer(product, parent, action.seeds[0])
            path = run.directory / "preparations" / f"{digest(action)[:20]}.sdf"
            write_sdf(path, initial)
            state.work_item = WorkItem(item_id=action.candidate.candidate_id, graph_id=graph_id(product),
                        mapped_smiles=smiles(product, True), sdf_path=str(path),
                        atom_mapping=mapping["parent_to_product_maps"], source="input_conformer")
            state.candidate_history.append(graph_id(product))
            state.proposal = None
        expected = read_sdf(state.work_item.sdf_path)
        graph = state.work_item.graph_id
        for seed in action.seeds:
            nfe_cap = (action.generation.integration_steps + 1) * self.nfe_factor
            request_key = digest([graph, self.kernel_id(action.generation), seed])
            if request_key in state.generation_attempts:
                raise ValueError("Repeated graph/kernel/seed request would not be an independent draw")
            if state.graph_attempts.get(graph, 0) >= run.case.budgets.seeds_per_graph:
                raise BudgetExhausted("Graph seed cap reached")
            run.budget.charge("fm_attempt", fm_draws=1, fm_nfe=nfe_cap)
            state.graph_attempts[graph] = state.graph_attempts.get(graph, 0) + 1
            state.generation_attempts.append(request_key)
            directory = run.directory / "generation" / request_key[:24]
            request = GenerationRequest(graph_smiles=state.work_item.mapped_smiles, graph_id=graph,
                       pocket_path=run.fixed_pocket, reference_sdf=state.work_item.sdf_path,
                       protected_atom_mapping=state.work_item.atom_mapping, parameters=action.generation,
                       seed=seed, output_dir=str(directory))
            # Durable reservation precedes the RPC. Resume never silently redraws pending work.
            run.save(pending={"stage": "fm", "action": action.model_dump(mode="json"),
                              "seed": seed, "request": request.model_dump(mode="json")})
            try:
                response = GenerationResult.model_validate(self.client.request("/generate", request.model_dump(
                           mode="json"), timeout=min(600, run.budget.remaining()["wall_seconds"])))
            except Exception as exc:
                response = GenerationResult(status="failed", raw_draws=1, nfe=nfe_cap,
                             kernel_id=self.kernel_id(action.generation), reason=f"RPC failed; cost unknown: {exc}")
            if response.raw_draws > 1 or response.nfe > nfe_cap:
                raise RuntimeError("Backend exceeded the reserved raw-draw/NFE bound; aborting run")
            run.budget.refund("fm_reported_actual", fm_draws=1 - response.raw_draws, fm_nfe=nfe_cap - response.nfe)
            if response.raw_draws == 0:
                state.graph_attempts[graph] -= 1
            if response.status == "ok":
                raw = read_sdf(response.sdf_path)
                check = graph_check(expected, raw, derive_stereo=True)
                if check["matches"]:
                    # Atom ordering is handled by graph isomorphism, not SDF ordering.
                    correspondence = check["expected_index_to_actual_index"]
                    for i, j in correspondence.items():
                        raw.GetAtomWithIdx(j).SetAtomMapNum(expected.GetAtomWithIdx(i).GetAtomMapNum())
                    mapped_path = directory / "raw_mapped.sdf"
                    write_sdf(mapped_path, raw)
                    run.budget.charge("evaluate_fm_raw", evaluation_calls=1)
                    run.save(pending={"stage": "evaluation", "raw_sdf": str(mapped_path),
                                      "action": action.model_dump(mode="json")})
                    obs = run.evaluator.evaluate(raw)
                    item = WorkItem(item_id=request_key[:24], graph_id=graph, mapped_smiles=smiles(raw, True),
                              sdf_path=str(mapped_path), atom_mapping=state.work_item.atom_mapping,
                              observation=obs, seed=seed, kernel_id=response.kernel_id, source="fm_raw")
                    run.incorporate(item)
                else:
                    response.status, response.reason = "graph_mismatch", "orchestrator graph validation failed"
            if response.status != "ok":
                properties = graph_properties(smiles(expected))
                records_failed = {"properties": EvaluationRecord(status="ok", values=properties),
                                  "fm": EvaluationRecord(status="failed", reason=response.reason)}
                observation = Observation(observation_id=digest([request_key, response.reason]), graph_id=graph,
                    evaluations=records_failed, metrics=properties, graph_matches=response.status != "graph_mismatch",
                    failures=["fm_failed", response.status], incomplete_evaluation=True)
                state.work_item = state.work_item.model_copy(update={"observation": observation})
                state.evidence = assemble_evidence(observation)
                state.plateau += 1
                append_jsonl(run.directory / "observations.jsonl", observation)
            cost = {k: run.budget.used[k] - pre_state["budget_used"].get(k, 0) for k in run.budget.resources}
            record = {"pre_state_id": pre_state["state_id"], "target_id": run.case.target_id,
                      "series_id": run.case.series_id, "action": action.model_dump(mode="json"),
                      "observation_before": pre_state["work_item"]["observation"], "seed": seed,
                      "kernel_id": response.kernel_id, "generation_result": response.model_dump(mode="json"),
                      "resulting_work_item": state.work_item.model_dump(mode="json"),
                      "feasible": bool(state.work_item.observation and state.work_item.observation.feasible),
                      "evaluation_status": {k: r.status for k, r in state.work_item.observation.evaluations.items()},
                      "compute_cost": cost}
            append_jsonl(run.directory / "actions.jsonl", record)
            records.append(record)
            run.save()
        return records


def optimize(run: Run, fm_client, llm_client, router) -> dict:
    if run.state.status in ("success", "no_improvement") and (run.directory / "result.json").exists():
        return json.loads((run.directory / "result.json").read_text())
    executor = Executor(run, fm_client)
    planner = Planner(llm_client, run.config.llm, run.directory, run.budget,
                      persist=lambda event: run.save(pending=event))
    state = run.state
    reason = "maximum_rounds"
    while state.round_index < run.case.budgets.max_rounds:
        run.refresh()
        if state.plateau >= run.case.budgets.plateau_rounds:
            reason = "plateau"
            break
        if run.budget.remaining()["wall_seconds"] <= 0:
            reason = "wall_budget"
            break
        try:
            if state.proposal is None and run.budget.can_afford(llm_calls=1, llm_tokens=run.config.llm.context_cap):
                run.save(pending={"stage": "llm", "state_id": state.state_id})
                state.proposal = planner.propose(state, run.case)
                if state.proposal:
                    state.hypotheses.extend(state.proposal.hypotheses)
                run.save()
            run.refresh()
            actions = available_actions(state, run.case, run.config.fm.parameters,
                                        executor.capabilities, executor.nfe_factor, run.config.seed)
            if isinstance(router, LearnedRouter):
                action = router.select(state, actions, run.case, executor.kernel_id(run.config.fm.parameters))
            else:
                action = router.select(state, actions)
            if action.kind == ActionKind.STOP:
                reason = action.reason
                break
            executor.execute(action)
            state.round_index += 1
            run.save()
            best, baseline = state.best_feasible, state.original_ligand.observation
            if best and best.observation.target_met and baseline.utility is not None and (
                    best.observation.utility - baseline.utility >= run.case.objective.minimum_improvement):
                reason = "success"
                break
        except BudgetExhausted as exc:
            reason = str(exc)
            run.save()
            break
    return run.finish(reason)
