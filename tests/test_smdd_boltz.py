import unittest

from dirmo.evaluation.smdd_boltz import binding_conditions, cofold_input


class SmddBoltzTests(unittest.TestCase):
    def test_auth_chains_and_repeated_sequence_remain_in_cofold(self):
        config = {"pocket_contacts": [["C", 2]], "co_ligands": None}
        data = cofold_input(">id|Chains A, B[auth C]|protein\nACDE\n", "CCO", config)
        self.assertEqual([s["protein"]["id"] for s in data["sequences"][:2]], ["A", "C"])
        self.assertEqual(data["sequences"][1]["protein"]["sequence"], "ACDE")

    def test_per_model_mean_strict_binding_threshold_and_affinity_hold(self):
        result = {"affinity_probability_binary": .99, "affinity_probability_binary1": .6,
                  "affinity_probability_binary2": .8, "affinity_pred_value": -9,
                  "affinity_pred_value1": -1.2, "affinity_pred_value2": -.8}
        config = {"baseline_affinity": -.5, "affinity_objective": {"threshold": .3}}
        checked = binding_conditions(result, config)
        self.assertAlmostEqual(checked["binding_probability"], .7)
        self.assertFalse(checked["binding_passed"])
        self.assertTrue(checked["affinity_goal_passed"])
        config = {"baseline_affinity": -1.5, "affinity_hold_constant": {"tolerance": .3}}
        self.assertFalse(binding_conditions(result, config)["affinity_hold_passed"])


if __name__ == "__main__":
    unittest.main()
