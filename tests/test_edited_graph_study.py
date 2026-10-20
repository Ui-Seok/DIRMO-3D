"""Chemical correspondence and prevention of reference-free-coordinate leakage."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from dirmo.chemistry.graphs import graph_check, mapped, write_sdf
from dirmo.evaluation.edited_graph import (anchor_indices, assess_saved_graph, graph_from_record,
    graph_only_conformer, graph_record, terminal_edits, unsupported_tensor_features)


class EditedGraphStudyTests(unittest.TestCase):
    def test_unsupported_features_are_rejected_without_altering_graph(self):
        graph = Chem.MolFromSmiles("[13CH3][CH]C")
        before = Chem.MolToSmiles(graph)
        features = unsupported_tensor_features(graph)
        self.assertEqual({x["feature"] for x in features}, {"isotope", "radical_electrons"})
        self.assertEqual(Chem.MolToSmiles(graph), before)
        self.assertFalse(unsupported_tensor_features(Chem.MolFromSmiles("C[NH3+]")))

    def setUp(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCCc1ccccc1"))
        params = AllChem.ETKDGv3()
        params.randomSeed = 19
        self.assertEqual(AllChem.EmbedMolecule(mol, params), 0)
        self.native = mapped(Chem.RemoveHs(mol))
        self.free = [0, 1, 2]
        self.held = {i: p.tolist() for i, p in enumerate(self.native.GetConformer().GetPositions()) if i not in self.free}
        self.anchors = anchor_indices(self.native, self.free, 3)

    def test_edits_retain_maps_bonds_and_protected_atom_types(self):
        products = terminal_edits(self.native, self.free)
        self.assertEqual([n for n, _, _ in products], ["original", "C_to_N", "C_to_O", "C_to_F"])
        for name, product, edited in products:
            stored = graph_record(name, product, edited, self.free)
            restored = graph_from_record(stored)
            self.assertEqual(restored.GetNumConformers(), 0)
            self.assertTrue(graph_check(product, restored)["matches"])
            self.assertEqual([a.GetAtomMapNum() for a in restored.GetAtoms()], stored["atom_maps"])
            self.assertEqual([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in product.GetBonds()],
                             [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondType()) for b in self.native.GetBonds()])
            self.assertEqual([a.GetAtomicNum() for a in product.GetAtoms()][1:],
                             [a.GetAtomicNum() for a in self.native.GetAtoms()][1:])

    def test_initialization_does_not_read_native_free_xyz(self):
        altered = Chem.Mol(self.native)
        xyz = altered.GetConformer().GetPositions()
        xyz[self.free] += [100., -70., 40.]
        altered.GetConformer().SetPositions(xyz)
        first = graph_only_conformer(self.native, self.held, self.anchors, 918)
        second = graph_only_conformer(altered, self.held, self.anchors, 918)
        np.testing.assert_array_equal(first.GetConformer().GetPositions(), second.GetConformer().GetPositions())
        for i, target in self.held.items():
            np.testing.assert_array_equal(first.GetConformer().GetPositions()[i], target)
        third = graph_only_conformer(self.native, self.held, self.anchors, 919)
        self.assertFalse(np.allclose(first.GetConformer().GetPositions()[self.free], third.GetConformer().GetPositions()[self.free]))

    def test_saved_graph_and_core_are_checked_independently(self):
        _, product, _ = terminal_edits(self.native, self.free)[1]
        prepared = graph_only_conformer(product, self.held, self.anchors, 918)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pose.sdf"
            write_sdf(path, prepared)
            held = {i: np.round(xyz, 4).tolist() for i, xyz in self.held.items()}
            good = assess_saved_graph(product, path, held)
            self.assertTrue(good["graph_preserved"])
            self.assertTrue(good["core_preserved"])
            wrong_graph = assess_saved_graph(self.native, path, held)
            self.assertFalse(wrong_graph["graph_preserved"])
            self.assertTrue(wrong_graph["core_preserved"])
            moved = {i: (np.array(xyz) + [1., 0., 0.]).tolist() for i, xyz in held.items()}
            wrong_position = assess_saved_graph(product, path, moved)
            self.assertTrue(wrong_position["graph_preserved"])
            self.assertFalse(wrong_position["core_preserved"])


if __name__ == "__main__":
    unittest.main()
