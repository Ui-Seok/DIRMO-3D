"""Injected worker unit tests only: no mock output is reported as real FM generation."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dirmo.chemistry.graphs import graph_id, read_sdf, write_sdf
from dirmo.config import Config
from dirmo.data.collect import collect
from dirmo.io import digest, read_jsonl, sha256_file
from dirmo.llm.proposals import Planner
from dirmo.pipeline.budget import Budget
from dirmo.pipeline.run import Executor, Run
from dirmo.routing.policies import available_actions
from dirmo.schemas import (ActionKind, BudgetLimits, Case, EditOperation, GenerationParameters, MetricGoal, Objective,
                           Proposal, State, WorkItem, kernel_identity)
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
    def test_collection_offsets_add_independent_draws_and_skip_existing_requests(self):
        class MeasuredFM(FakeFM):
            def request(self, route, payload, timeout=10):
                self.requests.append(payload)
                path = Path(payload["output_dir"]) / "raw.sdf"
                write_sdf(path, read_sdf(payload["reference_sdf"]))
                steps = payload["parameters"]["integration_steps"]
                kernel = digest(kernel_identity({"unit_fixture": True},
                                GenerationParameters.model_validate(payload["parameters"]),
                                sha256_file(payload["pocket_path"])))
                return {"status": "ok", "sdf_path": str(path), "raw_draws": 1,
                        "nfe": steps + 1, "kernel_id": kernel}

        with tempfile.TemporaryDirectory() as temp:
            root, mol = Path(temp), molecule()
            ligand, protein = root / "initial.sdf", root / "protein.pdb"
            write_sdf(ligand, mol)
            protein.write_text("ATOM      1  CA  ALA A   1       4.000   0.000   0.000  1.00 20.00           C  \nEND\n")
            case = Case(case_id="unit", target_id="unit", initial_ligand_path=str(ligand), pocket_path=str(protein),
                        objective=Objective(metrics={"qed": MetricGoal(direction="maximize", target=0.5)},
                                            required_evaluations=["properties"]))
            fm, dataset = MeasuredFM(), root / "branches.jsonl"
            for index, offset in enumerate((0, 1, 0)):
                with patch("dirmo.pipeline.run.Evaluator.evaluate", return_value=observation(graph_id(mol), 0.4)):
                    run = Run(case, Config(), root / f"run_{index}")
                with patch("dirmo.pipeline.run.Evaluator.evaluate",
                           return_value=observation(graph_id(mol), 0.8 + offset * 0.1)):
                    summary = collect(run, fm, FakeLLM(), str(dataset), seed_offset=offset)
                self.assertEqual(summary["seed_base"], 42 + offset)
                self.assertEqual(summary["executed_actions"], 2 if index < 2 else 0)
                if index == 2:
                    self.assertTrue(all("Repeated graph/kernel/seed" in entry["reason"]
                                        for entry in summary["coverage"]))
            rows = read_jsonl(dataset)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(fm.requests), 4)
            self.assertEqual([row["seed"] for row in rows], [42, 42, 43, 43])
            self.assertEqual(rows[0]["kernel_id"], rows[2]["kernel_id"])
            self.assertNotEqual(rows[0]["kernel_id"], rows[1]["kernel_id"])
            self.assertAlmostEqual(rows[0]["utility_change"], 0.4)
            self.assertAlmostEqual(rows[2]["utility_change"], 0.5)
            self.assertTrue(all(row["compute_cost"]["fm_draws"] == 1 for row in rows))

    def test_chemical_rejection_retries_without_accepting_unmapped_smiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root, mol = Path(temp), molecule()
            path = root / "initial.sdf"
            write_sdf(path, mol)
            item = WorkItem(item_id="unit", graph_id=graph_id(mol),
                            mapped_smiles="[CH3:1][CH2:2][OH:3]", sdf_path=str(path),
                            observation=observation(graph_id(mol)))
            state = State(state_id="unit", case_id="unit", original_ligand=item, work_item=item)
            case = Case(case_id="unit", target_id="unit", pocket_path="unused", initial_ligand_path=str(path))
            bad = candidate(mol, operation=None, mapped_product_smiles="CCN", evidence_ids=[])
            good = candidate(mol, evidence_ids=[])
            proposals = [Proposal(hypotheses=[], evidence_ids=[], edit_candidates=[edit], expected_tradeoffs=[])
                         for edit in (bad, good)]
            client = FakeLLM()
            calls = []

            def request(route, payload, timeout=10):
                calls.append(payload)
                return {"status": "ok", "raw": proposals[len(calls) - 1].model_dump_json(), "total_tokens": 50}

            client.request = request
            budget = Budget(BudgetLimits(llm_calls=2, llm_tokens=12000))
            result = Planner(client, Config().llm, root, budget).propose(state, case)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result.edit_candidates[0].operation.kind, "replace_atom")
            self.assertIn("mapping_ambiguous", calls[1]["messages"][-1]["content"])
            self.assertEqual(budget.used["llm_calls"], 2)
            self.assertEqual(budget.used["llm_tokens"], 100)
            events = [json.loads(line) for line in (root / "llm.jsonl").read_text().splitlines()]
            self.assertTrue(events[0]["parsing_valid"])
            self.assertFalse(events[0]["chemical_valid"])
            self.assertEqual(events[1]["chemical_valid_count"], 1)

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


class OracleArmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root, mol = Path(self.temp.name), molecule()
        ligand, protein = root / "initial.sdf", root / "protein.pdb"
        write_sdf(ligand, mol)
        protein.write_text("ATOM      1  CA  ALA A   1       4.000   0.000   0.000  1.00 20.00           C  \nEND\n")
        self.mol = mol
        self.case = Case(case_id="unit", target_id="unit", initial_ligand_path=str(ligand),
                         pocket_path=str(protein), budgets=BudgetLimits(candidate_cap=1),
                         objective=Objective(metrics={"qed": MetricGoal(direction="maximize", target=0.5)},
                                             required_evaluations=["properties"]))
        with patch("dirmo.pipeline.run.Evaluator.evaluate", return_value=observation(graph_id(mol), 0.4)):
            self.run = Run(self.case, Config(), root / "run")
        self.fm = FakeFM()
        self.executor = Executor(self.run, self.fm)

    def tearDown(self):
        self.temp.cleanup()

    def ids(self, extra=()):
        actions = available_actions(self.run.state, self.case, GenerationParameters(),
                                    self.executor.capabilities, 1, 42, extra)
        return [a.action_id for a in actions]

    def test_extra_candidates_are_offered_past_the_planner_cap(self):
        planned = candidate(self.mol, candidate_id="planned")
        dropped = candidate(self.mol, candidate_id="dropped",
                            operation=EditOperation(kind="replace_atom", atom_map=3, element="F"))
        oracle = candidate(self.mol, candidate_id="oracle-1",
                           operation=EditOperation(kind="replace_atom", atom_map=3, element="Cl"))
        self.run.state.proposal = Proposal(hypotheses=[], evidence_ids=[],
                                           edit_candidates=[planned, dropped], expected_tradeoffs=[])
        ids = self.ids([oracle])
        # candidate_cap=1 keeps the planner's condition identical whether or not an
        # oracle arm is present, so the second planner candidate stays dropped.
        self.assertIn("edit:planned", ids)
        self.assertNotIn("edit:dropped", ids)
        self.assertIn("edit:oracle-1", ids)

    def test_extra_candidates_work_without_any_proposal(self):
        self.assertIsNone(self.run.state.proposal)
        self.assertIn("edit:oracle-1", self.ids([candidate(self.mol, candidate_id="oracle-1")]))

    def test_two_sources_proposing_one_graph_yield_one_action(self):
        planned = candidate(self.mol, candidate_id="planned")
        same_graph = candidate(self.mol, candidate_id="oracle-1")
        self.run.state.proposal = Proposal(hypotheses=[], evidence_ids=[],
                                           edit_candidates=[planned], expected_tradeoffs=[])
        ids = self.ids([same_graph])
        self.assertEqual([i for i in ids if i.startswith("edit:")], ["edit:planned"])

    def test_executor_revalidation_accepts_an_extra_candidate(self):
        oracle = candidate(self.mol, candidate_id="oracle-1")
        executor = Executor(self.run, self.fm, extra_candidates=[oracle])
        action = next(a for a in available_actions(self.run.state, self.case, GenerationParameters(),
                                                   executor.capabilities, 1, 42, [oracle])
                      if a.action_id == "edit:oracle-1")
        with patch("dirmo.pipeline.run.Evaluator.evaluate", return_value=observation(graph_id(self.mol), 0.9)):
            records = executor.execute(action)
        self.assertEqual(len(records), 1)
        # Without the executor carrying the same list this raises "not in the current
        # validated feasible action set".
        plain = Executor(self.run, self.fm)
        with self.assertRaisesRegex(ValueError, "validated feasible action set"):
            plain.execute(action)

    def test_the_loaded_mode_masks_actions_it_cannot_execute(self):
        # Offering an action the mode cannot execute spends a draw on a certain failure:
        # whole-graph actions need graph_inpainting, a redesign needs inpainting_mode.
        graph = dict(self.executor.capabilities, graph_mode="graph",
                     priors=["random", "reference", "fragment"])
        fragment = dict(graph, graph_mode="fragment")

        def kinds(caps, regions):
            return {a.kind for a in available_actions(self.run.state, self.case,
                        GenerationParameters(), caps, 1, 42, (), regions)}

        self.assertIn(ActionKind.POSE_RESAMPLE, kinds(graph, 2))
        self.assertNotIn(ActionKind.REDESIGN_REGION, kinds(graph, 2))
        offered = kinds(fragment, 2)
        self.assertIn(ActionKind.REDESIGN_REGION, offered)
        self.assertNotIn(ActionKind.POSE_RESAMPLE, offered)
        self.assertNotIn(ActionKind.CHANGE_GENERATION_CONDITION, offered)
        # Asking for no regions in fragment mode leaves nothing but STOP.
        self.assertEqual(kinds(fragment, 0), {ActionKind.STOP})

    def test_generation_kernel_drops_the_pocket_but_the_request_kernel_keeps_it(self):
        from dirmo.schemas import generation_kernel, kernel_identity
        kernel = kernel_identity({"unit": True}, GenerationParameters(), "pocket-hash")
        self.assertIn("pocket_sha256", kernel)
        view = generation_kernel(kernel)
        self.assertNotIn("pocket_sha256", view)
        # Everything that describes how the draw is made survives the view.
        self.assertEqual({k: v for k, v in kernel.items() if k != "pocket_sha256"}, view)
        self.assertNotEqual(self.executor.kernel_id(GenerationParameters()),
                            self.executor.generation_kernel_id(GenerationParameters()))

    def test_collection_records_candidate_source(self):
        oracle = candidate(self.mol, candidate_id="oracle-1")
        dataset = Path(self.temp.name) / "branches.jsonl"
        with patch("dirmo.data.collect.ranked_edits", return_value=[oracle]), \
             patch("dirmo.pipeline.run.Evaluator.evaluate", return_value=observation(graph_id(self.mol), 0.8)):
            summary = collect(self.run, self.fm, FakeLLM(), str(dataset), oracle_edits=1)
        self.assertEqual(summary["oracle_edits_offered"], 1)
        self.assertEqual(summary["oracle_candidate_ids"], ["oracle-1"])
        rows = read_jsonl(dataset)
        self.assertEqual(len({row["generation_kernel_sha256"] for row in rows
                              if row["integration_steps"] == 64}), 1)
        sources = {row["action"]["action_id"]: row["candidate_source"] for row in rows}
        self.assertEqual(sources["edit:oracle-1"], "oracle")
        self.assertIsNone(sources["pose_resample"])


class FlowrPriorTests(unittest.TestCase):
    def test_random_prior_without_optional_features_preserves_native_sampling(self):
        try:
            import torch
            from flowr.data.interpolate import GeometricNoiseSampler
        except ModuleNotFoundError:
            self.skipTest("FLOWR native CPU dependencies are optional outside the FM environment")
        from dirmo.generation.flowr import prepare_graph_prior
        sampler = GeometricNoiseSampler(4, 3, 3)
        target = sampler.sample_molecule(4)
        initial = sampler.sample_molecule(4)
        interpolant = SimpleNamespace(prior_sampler=sampler)
        prepare_graph_prior(interpolant, "random")
        mask = torch.ones(4, dtype=torch.bool)
        torch.manual_seed(19)
        actual = sampler.inpaint_graph(target, initial, mask, mode="random")

        native = GeometricNoiseSampler(4, 3, 3, n_hybridization_types=2)
        featured = target._copy_with(hybridization=torch.tensor([[0.0, 1.0]] * 4))
        torch.manual_seed(19)
        expected = native.inpaint_graph(featured, initial, mask, mode="random")
        self.assertTrue(torch.equal(actual.coords, expected.coords))
        self.assertTrue(torch.isfinite(actual.coords).all())
        self.assertTrue(torch.allclose(actual.coords.mean(0), torch.zeros(3), atol=1e-6))
        for field in ("atomics", "charges", "bond_indices", "bond_types"):
            self.assertTrue(torch.equal(getattr(actual, field), getattr(target, field)))
        self.assertIsNone(actual.hybridization)
        self.assertEqual(actual.fragment_mode, "graph")
        self.assertTrue(actual.fragment_mask.all())

    def test_reference_prior_starts_from_the_reference_coordinates(self):
        try:
            import torch
            from flowr.data.interpolate import GeometricNoiseSampler
        except ModuleNotFoundError:
            self.skipTest("FLOWR native CPU dependencies are optional outside the FM environment")
        from dirmo.generation.flowr import prepare_graph_prior
        sampler = GeometricNoiseSampler(4, 3, 3)
        target = sampler.sample_molecule(4)
        initial = sampler.sample_molecule(4)
        prepare_graph_prior(SimpleNamespace(prior_sampler=sampler), "reference")
        mask = torch.ones(4, dtype=torch.bool)
        actual = sampler.inpaint_graph(target, initial, mask, mode="random")

        # The reference pose enters untouched: no draw, and no re-centring that would
        # fight the upstream output transform.
        self.assertTrue(torch.equal(actual.coords, target.coords))
        for field in ("atomics", "charges", "bond_indices", "bond_types"):
            self.assertTrue(torch.equal(getattr(actual, field), getattr(target, field)))
        self.assertIsNone(actual.hybridization)
        self.assertEqual(actual.fragment_mode, "graph")
        self.assertTrue(actual.fragment_mask.all())

    def test_reference_prior_noise_is_seeded_and_bounded(self):
        try:
            import torch
            from flowr.data.interpolate import GeometricNoiseSampler
        except ModuleNotFoundError:
            self.skipTest("FLOWR native CPU dependencies are optional outside the FM environment")
        from dirmo.generation.flowr import prepare_graph_prior
        mask = torch.ones(4, dtype=torch.bool)

        def draw(seed):
            sampler = GeometricNoiseSampler(4, 3, 3)
            torch.manual_seed(0)
            target = sampler.sample_molecule(4)
            initial = sampler.sample_molecule(4)
            prepare_graph_prior(SimpleNamespace(prior_sampler=sampler), "reference", 0.3)
            torch.manual_seed(seed)
            return target, sampler.inpaint_graph(target, initial, mask, mode="random")

        target, first = draw(11)
        _, same = draw(11)
        _, other = draw(12)
        # Reproducible for one seed, genuinely different across seeds, and a small
        # displacement rather than a fresh draw.
        self.assertTrue(torch.equal(first.coords, same.coords))
        self.assertFalse(torch.equal(first.coords, other.coords))
        self.assertLess(float((first.coords - target.coords).norm(dim=1).mean()), 1.5)
        self.assertGreater(float((first.coords - target.coords).norm(dim=1).mean()), 0.0)

    def test_prior_noise_is_rejected_for_the_random_prior(self):
        from dirmo.generation.flowr import CapabilityError, prepare_graph_prior
        with self.assertRaises(CapabilityError):
            prepare_graph_prior(SimpleNamespace(prior_sampler=object()), "random", 0.3)

    def test_unknown_prior_is_a_capability_error(self):
        from dirmo.generation.flowr import CapabilityError, prepare_graph_prior
        with self.assertRaises(CapabilityError):
            prepare_graph_prior(SimpleNamespace(prior_sampler=object()), "harmonic")


if __name__ == "__main__":
    unittest.main()
