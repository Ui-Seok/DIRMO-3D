"""Original-baseline anchoring, leakage, feedback, and truthful gate statuses."""
import json
import unittest

from dirmo.evaluation.smdd_feedback import (
    assess_plan, attach_admet, messages, molecule_analysis, next_state, select_attempt, tool_feedback,
)


class FeedbackTests(unittest.TestCase):
    def row(self):
        return {"reference_smiles": "CCO", "pocket": {},
                "goals": [{"property": "clearance", "direction": "minimize", "threshold": 5, "baseline": 40}],
                "hold": [{"property": "bbb", "direction": "change", "tolerance": .1, "baseline": .8}],
                "witness_smiles": "SECRET_WITNESS", "witness_metrics": {"SECRET_FUTURE": 0},
                "answer": "SECRET_ANSWER", "task_id": "SECRET_ID", "source": "SECRET_PATH"}

    def plan(self, smiles, maps=None):
        return json.dumps({"focus_atom_maps": maps or [3], "edit_hypothesis": "Replace the terminal group.",
                           "product_smiles": smiles})

    def test_input_does_not_leak_and_original_baselines_survive_revision(self):
        prompt = messages(self.row(), "CCN", {"clearance": 40}, {"clearance": 37}, 2,
                          {"failed_conditions": [{"property": "clearance", "improvement": 3}]})
        self.assertNotIn("SECRET", json.dumps(prompt))
        obj = json.loads(prompt[1]["content"])
        self.assertEqual(obj["goals"][0]["baseline"], 40)
        self.assertEqual(obj["original_molecule"]["smiles"], "CCO")
        self.assertEqual(obj["current_molecule"]["admet_prediction"]["clearance"], 37)
        self.assertIn("tool_feedback", obj)

    def test_ids_are_current_local_ids_and_boolean_is_not_atom_number(self):
        self.assertFalse(assess_plan(self.plan("CCN", [True]), "CCO", "CCO")["plan_valid"])
        self.assertFalse(assess_plan(self.plan("CCN", [4]), "CCO", "CCO")["plan_valid"])
        self.assertFalse(assess_plan(self.plan("CCN", [2, 2]), "CCO", "CCO")["plan_valid"])
        self.assertTrue(assess_plan(self.plan("CCN"), "CCO", "CCO")["plan_valid"])

    def test_graph_checks_and_raw_oracle_string_have_separate_roles(self):
        raw = "O[C@@H](C)F"
        checked = assess_plan(self.plan(raw), "CCO", "CCO")
        self.assertEqual(checked["oracle_smiles"], raw)
        self.assertNotEqual(checked["product_smiles"], raw)
        # Copying current is still a change relative to ORIGINAL, but low
        # similarity to ORIGINAL cannot be hidden by re-anchoring to current.
        anchor = assess_plan(self.plan("CCN"), "CCN", "CCO")
        self.assertTrue(anchor["changed"])
        self.assertFalse(anchor["checks"]["tanimoto"])

    def test_feedback_contains_measured_failures_and_not_fabricated_binding(self):
        assessment = {"plan_valid": True, "smiles_valid": True, "changed": True,
                      "local_hard_constraints_passed": True, "checks": {}, "oracle_smiles": "CCN"}
        attach_admet(assessment, self.row(), {"clearance": 37, "bbb": .6})
        feedback = tool_feedback(assessment, ["CCN"])
        self.assertEqual(len(feedback["failed_conditions"]), 2)
        self.assertEqual(feedback["failed_conditions"][0]["improvement"], 3)
        self.assertFalse(assessment["pre_boltz_passed"])
        attach_admet(assessment, self.row(), {"clearance": 30, "bbb": .8})
        self.assertTrue(assessment["pre_boltz_passed"])
        self.assertIsNone(assessment["full_benchmark_success"])
        self.assertIn("unmeasured", tool_feedback(assessment, [])['binding_status'])

    def test_bad_graph_cannot_become_next_substrate(self):
        checked = assess_plan(self.plan("CCO.C"), "CCO", "CCO")
        self.assertEqual(next_state("CCO", checked), "CCO")
        self.assertEqual(next_state("CCO", assess_plan(self.plan("[CH3:1]CO"), "CCO", "CCO")), "CCO")

    def test_atom_logp_attributions_include_hydrogens(self):
        for smi in ["CCO", "c1ccncc1", "C[N+](C)(C)C"]:
            analysis = molecule_analysis(smi)
            self.assertAlmostEqual(sum(a[-1] for a in analysis["atoms"]),
                                   analysis["rdkit_properties"]["logp"], places=3)

    def test_selection_prefers_measured_gate_over_valid_but_failed_product(self):
        attempts = [
            {"round": 1, "assessment": {"smiles_valid": True, "changed": True,
                                         "local_hard_constraints_passed": True, "pre_boltz_passed": False}},
            {"round": 2, "assessment": {"smiles_valid": True, "changed": True,
                                         "local_hard_constraints_passed": True, "pre_boltz_passed": True,
                                         "full_benchmark_success": None}}]
        self.assertEqual(select_attempt(attempts)["round"], 2)


if __name__ == "__main__":
    unittest.main()
