"""CPU unit tests. Fixtures/mocked workers are NOT real-model integration evidence."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError
from rdkit import Chem
from rdkit.Chem import AllChem

from dirmo.chemistry.graphs import (ChemicalError, execute_edit, graph_check, graph_id, mapped,
                                   smiles, write_sdf)
from dirmo.config import Config
from dirmo.diagnosis import wilson
from dirmo.evaluation.metrics import Evaluator
from dirmo.gpu import assert_worker_mask, resolve_device, worker_environment
from dirmo.llm.proposals import parse_proposal
from dirmo.pipeline.budget import Budget, BudgetExhausted
from dirmo.pipeline.run import Run
from dirmo.routing.features import FEATURE_NAMES, features
from dirmo.routing.policies import RuleRouter, available_actions
from dirmo.schemas import (Action, ActionKind, BudgetLimits, Case, EditCandidate, EditOperation,
                           EvaluationRecord, GenerationParameters, MetricGoal, Objective, Observation,
                           Protection)
from dirmo.training.router import fit_model, group_components, grouped_folds, train_router


def molecule(smi="CCO"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    assert AllChem.EmbedMolecule(mol, params) == 0
    return mapped(Chem.RemoveHs(mol))


def candidate(parent, **kw):
    base = dict(candidate_id="edit1", parent_graph_id=graph_id(parent), changed_atom_maps=[3],
                operation=EditOperation(kind="replace_atom", atom_map=3, element="N"),
                protection_explanation="Unit fixture; preserve carbon atoms", evidence_ids=["e1"])
    base.update(kw)
    return EditCandidate(**base)


def observation(gid, value=1, feasible=True, target=True, failures=None):
    return Observation(observation_id=f"obs{value}", graph_id=gid, evaluations={}, metrics={"qed": value},
                       feasible=feasible, target_met=target, utility=value, failures=failures or [])


class ChemistryTests(unittest.TestCase):
    def test_permuted_atom_order_and_aromatic_normalization(self):
        mol = molecule("c1ccccc1O")
        reversed_mol = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
        Chem.Kekulize(reversed_mol, clearAromaticFlags=True)
        check = graph_check(mol, reversed_mol)
        self.assertTrue(check["matches"])
        self.assertEqual(graph_id(mol), graph_id(reversed_mol))

    def test_charge_stereo_and_connectivity_mismatches(self):
        for expected, actual in [("C[NH3+]", "CN"), ("C[C@H](O)F", "C[C@@H](O)F"),
                                 ("CC=O", "CCO"), ("COC", "CCO")]:
            self.assertFalse(graph_check(Chem.MolFromSmiles(expected), Chem.MolFromSmiles(actual))["matches"])

    def test_real_edit_and_mapping(self):
        parent = molecule()
        product, report = execute_edit(parent, candidate(parent), parent, Protection(atom_maps=[1]))
        self.assertEqual(smiles(product), "CCN")
        self.assertEqual(report["parent_to_product_maps"], {1: 1, 2: 2, 3: 3})
        self.assertNotEqual(graph_id(parent), graph_id(product))

    def test_invalid_edit_and_protection(self):
        parent = molecule()
        with self.assertRaises(ChemicalError):
            execute_edit(parent, candidate(parent), parent, Protection(atom_maps=[3]))
        with self.assertRaises(ChemicalError):
            execute_edit(parent, candidate(parent, changed_atom_maps=[2, 3]), parent, Protection())
        with self.assertRaises(ChemicalError):
            execute_edit(parent, candidate(parent, operation=EditOperation(kind="remove_terminal_atom", atom_map=2)),
                         parent, Protection())

    def test_ambiguous_maps_are_rejected(self):
        parent = molecule()
        with self.assertRaises(ChemicalError):
            execute_edit(parent, candidate(parent, operation=None, mapped_product_smiles="CCN"), parent, Protection())

    def test_schema_rejects_code_and_unsupported_guidance(self):
        with self.assertRaises(ValidationError):
            GenerationParameters.model_validate({"prior": "random", "qed_guidance": 1})
        with self.assertRaises(ValidationError):
            EditOperation.model_validate({"kind": "python", "atom_map": 1, "code": "arbitrary code"})
        with self.assertRaises(ValidationError):
            Action(action_id="a", kind=ActionKind.CHEMICAL_EDIT)


class ProposalTests(unittest.TestCase):
    def test_evidence_is_validated(self):
        raw = json.dumps({"hypotheses": [{"hypothesis_id": "h", "statement": "test", "evidence_ids": ["e1"],
                          "test": "measure"}], "evidence_ids": ["e1"], "edit_candidates": [], "expected_tradeoffs": []})
        self.assertEqual(len(parse_proposal(raw, {"e1"}, "g", 2).hypotheses), 1)
        with self.assertRaises(ValueError):
            parse_proposal(raw, {"e2"}, "g", 2)

    def test_claimed_hypothesis_confirmation_rejected(self):
        raw = json.dumps({"hypotheses": [{"hypothesis_id": "h", "statement": "test", "evidence_ids": ["e1"],
                          "test": "measure", "status": "supported"}], "evidence_ids": ["e1"],
                          "edit_candidates": [], "expected_tradeoffs": []})
        with self.assertRaises(ValueError):
            parse_proposal(raw, {"e1"}, "g", 2)


class BudgetAndGPUTests(unittest.TestCase):
    def test_failed_attempt_stays_charged(self):
        budget = Budget(BudgetLimits(fm_draws=1, fm_nfe=10))
        budget.charge("failed FM", fm_draws=1, fm_nfe=10)
        with self.assertRaises(BudgetExhausted):
            budget.charge("retry", fm_draws=1)
        self.assertEqual(budget.used["fm_draws"], 1)
        self.assertEqual(budget.used["fm_nfe"], 10)

    def test_wall_budget_and_refund(self):
        budget = Budget(BudgetLimits(wall_seconds=10), wall_used=11)
        self.assertFalse(budget.can_afford(fm_draws=1))
        with self.assertRaises(BudgetExhausted):
            budget.charge("past deadline", evaluation_calls=1)
        with self.assertRaises(ValueError):
            budget.refund("invalid", fm_draws=100)

    def test_physical_to_logical_gpu_mapping(self):
        devices = [{"index": 1, "uuid": "GPU-one", "name": "NVIDIA RTX A6000", "total_mib": 49140}]
        device = resolve_device("1", devices)
        env = worker_environment(device, "/test/source")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-one")
        self.assertEqual(env["DIRMO_PHYSICAL_GPU"], "1")
        with patch.dict(os.environ, env):
            assert_worker_mask()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1", "DIRMO_GPU_UUID": "GPU-one"}):
            with self.assertRaises(RuntimeError):
                assert_worker_mask()
        with self.assertRaises(ValidationError):
            Config.model_validate({"llm": {"gpu": "1", "port": 8000}, "fm": {"gpu": "1", "port": 8001}})

    def test_wilson_small_sample_uncertainty(self):
        result = wilson(1, 2)
        self.assertGreater(result["interval"][1] - result["interval"][0], 0.7)
        self.assertIn("high", result["uncertainty"])
        self.assertIsNone(wilson(0, 0)["interval"])


class EvaluationTests(unittest.TestCase):
    def test_missing_required_metric_cannot_be_success(self):
        from dirmo.config import EvaluationConfig
        mol = molecule()
        case = Case(case_id="unit", target_id="unit", pocket_path="unavailable.pdb", initial_ligand_path="fixture.sdf")
        evaluator = Evaluator(EvaluationConfig(posebusters=False, vina=True), case, "unavailable.pdb", mol)
        result = evaluator.evaluate(mol)
        self.assertEqual(result.evaluations["vina"].status, "missing")
        self.assertFalse(result.feasible)
        self.assertFalse(result.target_met)
        self.assertIsNone(result.utility)
        self.assertTrue(result.incomplete_evaluation)
        self.assertIn("qed", result.metrics)

    def test_protected_coordinates_are_not_aligned(self):
        from dirmo.evaluation.metrics import preservation
        mol = molecule()
        translated = Chem.Mol(mol)
        conf = translated.GetConformer()
        for i, point in enumerate(conf.GetPositions() + [5, 0, 0]):
            conf.SetAtomPosition(i, point)
        case = Case(case_id="unit", target_id="unit", pocket_path="unused", initial_ligand_path="unused",
                    protected=Protection(atom_maps=[1], max_atom_displacement=1))
        result = preservation(mol, translated, case, EvaluationRecord(status="disabled"))
        self.assertFalse(result["hard_pass"])
        self.assertAlmostEqual(result["checks"][0]["distance_angstrom"], 5)


class StateAndRouterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mol = molecule()
        self.ligand = self.root / "initial.sdf"
        write_sdf(self.ligand, self.mol)
        self.protein = self.root / "protein.pdb"
        # Unit fixture, not a real protein integration test.
        self.protein.write_text("ATOM      1  CA  ALA A   1       4.000   0.000   0.000  1.00 20.00           C  \nEND\n")
        self.case = Case(case_id="unit", target_id="target", pocket_path=str(self.protein),
                         initial_ligand_path=str(self.ligand),
                         objective=Objective(metrics={"qed": MetricGoal(direction="maximize", target=0.5)},
                                             required_evaluations=["properties", "geometry"]))
        self.config = Config()

    def tearDown(self):
        self.temp.cleanup()

    def fixture_run(self):
        obs = observation(graph_id(self.mol), value=0.4, target=False)
        with patch("dirmo.pipeline.run.Evaluator.evaluate", return_value=obs):
            return Run(self.case, self.config, self.root / "run")

    def test_no_improvement_returns_original(self):
        run = self.fixture_run()
        result = run.finish("unit_stop")
        self.assertEqual(result["status"], "no_improvement")
        self.assertEqual(result["selected"]["item_id"], "original")

    def test_best_from_history_not_last_and_must_meet_targets(self):
        run = self.fixture_run()
        base = run.state.original_ligand
        best = base.model_copy(update={"item_id": "best", "observation": observation(base.graph_id, 0.8)})
        later = base.model_copy(update={"item_id": "later", "observation": observation(base.graph_id, 0.6)})
        fails_target = base.model_copy(update={"item_id": "miss", "observation": observation(base.graph_id, 1, target=False)})
        for item in (best, later, fails_target):
            run.incorporate(item)
        result = run.finish("unit_stop")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["selected"]["item_id"], "best")

    def test_failed_edited_graph_remains_recoverable(self):
        run = self.fixture_run()
        product, _ = execute_edit(self.mol, candidate(self.mol), self.mol, Protection())
        item = run.state.work_item.model_copy(update={"graph_id": graph_id(product), "mapped_smiles": smiles(product, True),
                                                       "source": "input_conformer", "observation": None})
        run.state.work_item = item
        run.refresh()
        actions = available_actions(run.state, self.case, GenerationParameters(),
                                    {"fixed_graph": True, "condition_changes": ["integration_steps"]})
        chosen = RuleRouter().select(run.state, actions)
        self.assertEqual(chosen.kind, ActionKind.POSE_RESAMPLE)
        self.assertEqual(run.state.work_item.graph_id, graph_id(product))
        self.assertNotEqual(run.state.work_item.graph_id, run.state.original_ligand.graph_id)

    def test_feasibility_mask_nfe_and_extra_validation(self):
        run = self.fixture_run()
        run.state.remaining_budget = {"fm_draws": 1, "fm_nfe": 10, "evaluation_calls": 1, "wall_seconds": 100}
        actions = available_actions(run.state, self.case, GenerationParameters(), {"fixed_graph": True})
        self.assertEqual([a.kind for a in actions], [ActionKind.STOP])

    def test_resume_and_tampering(self):
        run = self.fixture_run()
        resumed = Run(self.case, self.config, self.root / "run", resume=True)
        self.assertEqual(resumed.state.original_ligand.graph_id, run.state.original_ligand.graph_id)
        original_path = Path(resumed.state.original_ligand.sdf_path)
        original_path.write_text(original_path.read_text() + "tampered\n")
        with self.assertRaises(ValueError):
            Run(self.case, self.config, self.root / "run", resume=True)

    def test_pending_action_never_automatically_replayed(self):
        run = self.fixture_run()
        run.budget.charge("pending draw", fm_draws=1, fm_nfe=65)
        run.save(pending={"stage": "fm", "action": "fixture"})
        with self.assertRaisesRegex(ValueError, "Interrupted in-flight"):
            Run(self.case, self.config, self.root / "run", resume=True)
        checkpoint = json.loads((run.directory / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["state"]["budget_used"]["fm_draws"], 1)

    def test_features_do_not_include_action_outcome(self):
        run = self.fixture_run()
        action = Action(action_id="pose_resample", kind=ActionKind.POSE_RESAMPLE)
        f = features(run.state, action, self.case)
        self.assertEqual(len(f), len(FEATURE_NAMES))
        self.assertNotIn("utility_change", FEATURE_NAMES)


class TrainingTests(unittest.TestCase):
    def test_target_and_state_branches_never_cross_folds(self):
        rows = [{"target_id": "A", "pre_state_id": "s1"}, {"target_id": "B", "pre_state_id": "s1"},
                {"target_id": "B", "pre_state_id": "s2"}, {"target_id": "C", "pre_state_id": "s3"}]
        groups = group_components(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        for train, test in grouped_folds(rows)[0]:
            self.assertFalse({groups[i] for i in train} & {groups[i] for i in test})

    def test_single_target_has_no_generalization_claim(self):
        splits, report = grouped_folds([{"target_id": "A", "pre_state_id": "s1"},
                                       {"target_id": "A", "pre_state_id": "s2"}])
        self.assertFalse(splits)
        self.assertEqual(report["status"], "insufficient_groups")

    def test_standardization_uses_only_train_rows(self):
        # Synthetic unit fixture; this is not collected branch data or an integration result.
        x = np.array([[0.0, 2.0], [2.0, 4.0], [1000.0, 1000.0]])
        y = np.array([[0.1, 1], [0.2, 2], [99, 99]])
        _, scaling, _ = fit_model(x, y, np.array([0, 1]), epochs=2, seed=7)
        self.assertEqual(scaling["mean"], [1.0, 3.0])
        self.assertAlmostEqual(scaling["label_mean"][0], 0.15)

    def test_mlp_artifact_roundtrip_and_kernel_separation(self):
        # Entire dataset is an explicit synthetic UNIT fixture in a temporary directory.
        from dirmo.routing.learned import LearnedRouter
        rows = [{"kernel_id": "unit-only", "pre_state_id": f"s{i}", "target_id": f"T{i // 2}",
                 "utility_change": 0.1 * i, "feasible": bool(i % 2),
                 "action": {"kind": "POSE_RESAMPLE"}, "integration_steps": 64,
                 "feature_names": FEATURE_NAMES, "pre_action_features": [float(i)] * len(FEATURE_NAMES),
                 "compute_cost": {"wall_seconds": i + 1}} for i in range(4)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "synthetic-unit.jsonl"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
            report = train_router(str(dataset), str(root / "gate.json"), epochs=2)
            gate = LearnedRouter(str(root / "gate.json"))
            prediction = gate.predictions(rows[0]["pre_action_features"])
            self.assertEqual(prediction.shape, (2,))
            self.assertTrue(np.isfinite(prediction).all())
            self.assertEqual(report["split"]["groups"], 2)
            self.assertEqual(gate.artifact["supported_action_kinds"], ["POSE_RESAMPLE"])
            rows[0]["kernel_id"] = "another-unit-kernel"
            dataset.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "multiple or zero kernels"):
                train_router(str(dataset), str(root / "must-not-exist.json"), epochs=2)


if __name__ == "__main__":
    unittest.main()
