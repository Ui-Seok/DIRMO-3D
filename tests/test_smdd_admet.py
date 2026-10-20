"""Goal deltas and asymmetric hold tolerances, including deferred affinity."""
import unittest

from dirmo.evaluation.smdd_admet import conditions, oracle_smiles


class SmddAdmetTests(unittest.TestCase):
    def test_oracle_preserves_raw_smiles_instead_of_graph_canonical_string(self):
        record = {"raw": '```json\n{"product_smiles":"O[C@@H](C)F"}\n```',
                  "fence_removed_diagnostic": {"smiles_valid": True, "product_smiles": "C[C@H](O)F"}}
        self.assertEqual(oracle_smiles(record), "O[C@@H](C)F")

    def test_improvements_use_deltas_not_absolute_thresholds(self):
        row = {"goals": [{"property": "herg", "direction": "minimize", "threshold": .1, "baseline": .7},
                          {"property": "bbb", "direction": "maximize", "threshold": .1, "baseline": .4}], "hold": []}
        result = conditions(row, {"herg": .55, "bbb": .55})
        self.assertTrue(result["all_admet_goals_passed"])
        self.assertAlmostEqual(result["goals"][0]["improvement"], .15)
        self.assertFalse(conditions(row, {"herg": .65, "bbb": .55})["all_admet_goals_passed"])

    def test_holds_are_asymmetric_and_binding_is_unmeasured(self):
        row = {"goals": [{"property": "binding_affinity", "direction": "minimize", "threshold": .3, "baseline": -.7}],
               "hold": [{"property": "herg", "direction": "increase", "tolerance": .1, "baseline": .4},
                        {"property": "bbb", "direction": "change", "tolerance": .1, "baseline": .4},
                        {"property": "solubility", "direction": "decrease", "tolerance": .5, "baseline": -4}]}
        result = conditions(row, {"herg": .1, "bbb": .45, "solubility": -3})
        self.assertTrue(result["all_admet_holds_passed"])
        self.assertIsNone(result["goals"][0]["passed"])
        self.assertFalse(result["binding_affinity_measured"])
        self.assertFalse(conditions(row, {"herg": .6, "bbb": .45, "solubility": -3})["all_admet_holds_passed"])
        self.assertFalse(conditions(row, {"herg": .1, "bbb": .1, "solubility": -3})["all_admet_holds_passed"])
        self.assertFalse(conditions(row, {"herg": .1, "bbb": .45, "solubility": -5})["all_admet_holds_passed"])


if __name__ == "__main__":
    unittest.main()
