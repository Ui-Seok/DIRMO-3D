"""Leakage, output validation, and completion-loss alignment for SMDD SFT."""
import json
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from dirmo.data.smdd import assess_product, leakage_keys, load_pairs, make_messages, read_fasta, split_pairs


class SmddDataTests(unittest.TestCase):
    def test_official_witness_label_does_not_invent_sdf_stereochemistry_or_maps(self):
        from rdkit import Chem
        import yaml
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "smdd_004_fixture_0"
            folder.mkdir()
            (folder / "task.yaml").write_text(yaml.safe_dump(dict(
                id=folder.name, target_ligand=dict(chain="L", residue_number=1, resname="LIG"),
                pocket_residues=[["A", 1]], baseline_values={}, optimization_objectives=[], hold_constant=[])))
            (folder / "protein.fasta").write_text(">id|Chain A|protein\nA\n")
            (folder / "complex.pdb").write_text("")
            (folder / "witness.smi").write_text("CC(O)F\n")
            for filename, smi in [("reference.sdf", "CC(O)F"), ("witness.sdf", "C[C@H](O)F")]:
                mol = Chem.MolFromSmiles(smi)
                for i, atom in enumerate(mol.GetAtoms(), 1):
                    atom.SetAtomMapNum(i)
                with Chem.SDWriter(str(folder / filename)) as writer:
                    writer.write(mol)
            row = load_pairs(Path(root))[0]
            self.assertEqual(row["answer"], '{"product_smiles":"CC(O)F"}')
            self.assertNotEqual(row["witness_smiles"], row["witness_sdf_smiles"])
            self.assertEqual(row["pocket"]["unavailable_distances"], 1)

    def test_fasta_chain_labels_override_record_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "protein.fasta"
            path.write_text(">7OKF_2|Chain B|peptide\nAWVIPA\n>7OKF_1|Chain A|protein\nGPGADSCIQ\n")
            self.assertEqual(read_fasta(path), {"B": "AWVIPA", "A": "GPGADSCIQ"})

    def test_transitive_protein_molecule_scaffold_groups_do_not_leak(self):
        molecules = [("a", "CCO", "CCN"), ("a", "CCC", "CCCl"),
                     ("b", "CCN", "CCBr"), ("c", "Oc1ccccc1", "Nc1ccccc1"),
                     ("d", "Fc1ccccc1", "Clc1ccccc1"), ("e", "CC(=O)O", "CC(=O)N"),
                     ("f", "CCCC", "CCCCO"), ("g", "c1ccncc1", "Cc1ccncc1"),
                     ("h", "CCOC", "CCNC"), ("i", "CCCCC", "CCCCCO")]
        rows = [dict(task_id=str(i), protein_key=p, reference_smiles=a, witness_smiles=b, goals=[])
                for i, (p, a, b) in enumerate(molecules)]
        splits, report = split_pairs(rows)
        self.assertEqual(sum(len(s) for s in splits.values()), len(rows))
        self.assertEqual(report["overlaps"], {"train:validation": 0, "train:test": 0, "validation:test": 0})
        location = {r["task_id"]: name for name, values in splits.items() for r in values}
        self.assertEqual(location["0"], location["1"])
        self.assertEqual(location["0"], location["2"])
        self.assertEqual(location["3"], location["4"])
        self.assertEqual(split_pairs(rows)[0], splits)
        self.assertNotIn(("scaffold", ""), leakage_keys(rows[0]))

    def test_prompt_whitelist_excludes_future_outcomes(self):
        row = dict(reference_mapped_smiles="[CH3:1][OH:2]", goals=[], hold=[], pocket={},
                   witness_smiles="SECRET_WITNESS", witness_metrics={"SECRET_FUTURE": 999},
                   answer="SECRET_ANSWER", task_id="SECRET_ID")
        prompt = json.dumps(make_messages(row))
        self.assertNotIn("SECRET", prompt)
        self.assertIn("reference_mapped_smiles", prompt)

    def test_format_and_graph_checks_are_not_goal_success(self):
        for raw in ['CCO', '{"product_smiles":"CCO","reason":"extra"}', '{"product_smiles":null}']:
            self.assertFalse(assess_product(raw, "CCO")["json_valid"])
        for smi in ["CCO.C", "[CH3:1]CO", "[CH3]"]:
            result = assess_product(json.dumps({"product_smiles": smi}), "CCO")
            self.assertTrue(result["smiles_valid"])
            self.assertFalse(result["local_hard_constraints_passed"])
        copy = assess_product('{"product_smiles":"CCO"}', "CCO", "CCN")
        self.assertFalse(copy["changed"])
        self.assertIsNone(copy["oracle_goal_success"])
        self.assertFalse(copy["witness_exact_match"])


class CompletionLossTests(unittest.TestCase):
    def test_matches_explicit_masked_next_token_loss_and_excludes_prompt_labels(self):
        import torch
        import torch.nn.functional as F
        from dirmo.training.smdd_sft import completion_loss

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(11, 5)
                self.head = torch.nn.Linear(5, 11)

            def forward(self, input_ids, attention_mask, use_cache, logits_to_keep):
                return SimpleNamespace(logits=self.head(self.embedding(input_ids))[:, -logits_to_keep:])

        torch.manual_seed(7)
        model = TinyModel()
        sample = {"input_ids": [1, 2, 3, 4, 5, 6], "prompt_length": 3, "target_ids": [4, 5, 6]}
        actual = completion_loss(model, sample, "cpu")
        logits = model.head(model.embedding(torch.tensor(sample["input_ids"])))
        expected = F.cross_entropy(logits[2:5], torch.tensor([4, 5, 6]))
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        # Only the last prompt token predicts a completion; earlier prompt labels
        # and the token after the final prediction do not enter this toy loss.
        self.assertEqual(float(model.embedding.weight.grad[1:3].abs().sum()), 0)
        self.assertGreater(float(model.embedding.weight.grad[3:6].abs().sum()), 0)
        self.assertEqual(float(model.embedding.weight.grad[6].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
