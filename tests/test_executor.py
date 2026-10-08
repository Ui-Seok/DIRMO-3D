"""Injected worker unit tests only: no mock output is reported as real FM generation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dirmo.chemistry.graphs import graph_id, read_sdf, write_sdf
from dirmo.config import Config
from dirmo.llm.proposals import Planner
from dirmo.pipeline.budget import Budget
from dirmo.pipeline.run import Executor, Run
from dirmo.routing.policies import available_actions
from dirmo.schemas import (ActionKind, BudgetLimits, Case, GenerationParameters, MetricGoal, Objective,
                           Proposal)
from test_core import candidate, molecule, observation


class FakeFM:
    """Deterministic failure/success fixture for executor state transitions."""

    def __init__(self):
        self.requests = []

    def health(self):
        return {"model_identity": {"unit_fixture": True}, "nfe_per_step_upper_bound": 1,
                "capabilities": {"fixed_graph": True, "condition_changes": ["integration_steps"],
                                 "kernel_base": {"unit_fixture": True}}}

    def request(self, route, payload, timeout=10):
        self.requests.append(payload)
        if len(self.requests) == 1:
            return {"status": "failed", "raw_draws": 1, "nfe": 65, "kernel_id": "unit-kernel",
                    "reason": "unit simulated build failure"}
        path = Path(payload["output_dir"]) / "raw.sdf"
        write_sdf(path, read_sdf(payload["reference_sdf"]))
        return {"status": "ok", "sdf_path": str(path), "raw_draws": 1,
                "nfe": payload["parameters"]["integration_steps"] + 1, "kernel_id": "unit-kernel"}


class FakeLLM:
    def health(self):
        return {"model_identity": {"fixture": True}}

    def request(self, route, payload, timeout=10):
        return {"status": "ok", "raw": "malformed", "total_tokens": 50}


class ExecutorTests(unittest.TestCase):
    def test_failed_edited_graph_is_recovered_with_next_draw(self):
        with tempfile.TemporaryDirectory() as temp:
            root, mol = Path(temp), molecule()
            ligand, protein = root / "initial.sdf", root / "protein.pdb"
            write_sdf(ligand, mol)
            protein.write_text("ATOM      1  CA  ALA A   1       4.000   0.000   0.000  1.00 20.00           C  \nEND\n")
            case = Case(case_id="unit", target_id="unit", initial_ligand_path=str(ligand), pocket_path=str(protein),
                        objective=Objective(metrics={"qed": MetricGoal(direction="maximize", target=0.5)},
                                            required_evaluations=["properties"]))
            with patch("dirmo.pipeline.run.Evaluator.evaluate", return_value=observation(graph_id(mol), 0.4, target=False)):
                run = Run(case, Config(), root / "run")
            run.state.proposal = Proposal(hypotheses=[], evidence_ids=[], edit_candidates=[candidate(mol)],
                                          expected_tradeoffs=[])
            run.refresh()
            fm = FakeFM()
            executor = Executor(run, fm)
            actions = available_actions(run.state, case, GenerationParameters(), executor.capabilities)
            edit = next(a for a in actions if a.kind == ActionKind.CHEMICAL_EDIT)
            executor.execute(edit)
            edited_id = run.state.work_item.graph_id
            self.assertNotEqual(edited_id, run.state.original_ligand.graph_id)
            self.assertEqual(run.state.work_item.source, "input_conformer")
            run.refresh()
            recover = next(a for a in available_actions(run.state, case, GenerationParameters(),
                                                        executor.capabilities) if a.kind == ActionKind.POSE_RESAMPLE)
            with patch("dirmo.pipeline.run.Evaluator.evaluate", return_value=observation(edited_id, 0.8)):
                executor.execute(recover)
            self.assertEqual(fm.requests[0]["graph_id"], fm.requests[1]["graph_id"])
            self.assertEqual(run.budget.used["fm_draws"], 2)
            self.assertEqual(run.budget.used["fm_nfe"], 130)
            self.assertEqual(run.state.best_feasible.graph_id, edited_id)
            self.assertEqual(run.finish("unit")['status'], "success")
            self.assertIsNone(json.loads((run.directory / "checkpoint.json").read_text())["pending_action"])

    def test_llm_parse_retries_are_bounded_and_charged(self):
        with tempfile.TemporaryDirectory() as temp:
            root, mol = Path(temp), molecule()
            path = root / "initial.sdf"
            write_sdf(path, mol)
            from dirmo.schemas import State, WorkItem
            item = WorkItem(item_id="unit", graph_id=graph_id(mol), mapped_smiles="[CH3:1][CH2:2][OH:3]",
                            sdf_path=str(path), observation=observation(graph_id(mol)))
            state = State(state_id="unit", case_id="unit", original_ligand=item, work_item=item)
            case = Case(case_id="unit", target_id="unit", pocket_path="unused", initial_ligand_path=str(path))
            budget = Budget(BudgetLimits(llm_calls=2, llm_tokens=12000))
            checkpoints = []
            planner = Planner(FakeLLM(), Config().llm, root, budget, persist=checkpoints.append)
            self.assertIsNone(planner.propose(state, case))
            self.assertEqual(budget.used["llm_calls"], 2)
            self.assertEqual(budget.used["llm_tokens"], 100)
            self.assertEqual(len(checkpoints), 2)
            self.assertEqual(len((root / "llm.jsonl").read_text().splitlines()), 2)

    def test_posebusters_renamed_columns_are_used(self):
        import pandas as pd
        from dirmo.evaluation.metrics import posebusters
        mock = type("Preset", (), {
            "config": {"modules": [{"name": "Chemistry", "chosen_binary_test_output": ["passes_rdkit_sanity_checks"],
                                      "rename_outputs": {"passes_rdkit_sanity_checks": "Sanitization"}}]},
            "bust": lambda self, **kw: pd.DataFrame([{"sanitization": True}]),
        })
        with patch("posebusters.PoseBusters", return_value=mock()):
            result = posebusters(molecule(), "unused.pdb")
        self.assertTrue(result["passes"])
        self.assertEqual(result["checks"], {"sanitization": True})


if __name__ == "__main__":
    unittest.main()
