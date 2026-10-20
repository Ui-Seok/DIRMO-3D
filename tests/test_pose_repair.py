"""CPU contract checks for pose-only batches and physical stereo serialization."""
import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from dirmo.chemistry.graphs import graph_check, mapped, stereo_from_coordinates
from dirmo.generation.pose_repair import clamp_pose, pose_times
from dirmo.evaluation.pose_repair_checks import delete_atoms_batched


def embed(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 7
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError("Fixture embedding failed")
    return mapped(Chem.RemoveHs(mol))


class PoseRepairTests(unittest.TestCase):
    def test_large_protein_subset_preserves_residue_types_and_xyz(self):
        protein=Chem.RWMol()
        for i in range(1100):
            atom=Chem.Atom('O' if i%2 else 'C')
            atom.SetMonomerInfo(Chem.AtomPDBResidueInfo(atomName=' O  ' if i%2 else ' C  ',
                 serialNumber=i+1,residueName='HOH' if i%2 else 'ALA',residueNumber=i+1,isHeteroAtom=bool(i%2)))
            protein.AddAtom(atom)
        for i in range(0,20,2):
            protein.AddBond(i,i+1,Chem.BondType.SINGLE)
        conformer=Chem.Conformer(1100)
        conformer.Set3D(True)
        conformer.SetPositions(np.arange(3300,dtype=float).reshape(1100,3)/10.)
        protein.AddConformer(conformer)
        removed=list(range(21,1100))
        slow=Chem.RWMol(protein)
        for index in reversed(removed):
            slow.RemoveAtom(index)
        fast=delete_atoms_batched(protein,removed)
        self.assertEqual(Chem.MolToPDBBlock(slow),Chem.MolToPDBBlock(fast))
        self.assertEqual(Chem.MolToMolBlock(slow),Chem.MolToMolBlock(fast))

    def test_bulk_filter_preserves_retained_atoms_bonds_and_coordinates(self):
        mol=embed('CC[C@H](O)CC')
        original=mol.GetConformer().GetPositions().copy()
        for indices in ([],[0],[1,3,5],list(range(mol.GetNumAtoms()))):
            slow=Chem.RWMol(mol)
            for index in sorted(indices,reverse=True):
                slow.RemoveAtom(index)
            fast=delete_atoms_batched(mol,indices)
            self.assertEqual(Chem.MolToMolBlock(slow),Chem.MolToMolBlock(fast))
        np.testing.assert_array_equal(mol.GetConformer().GetPositions(),original)

    def test_stereoany_does_not_hide_xyz_but_true_isomer_is_rejected(self):
        expected = embed("C/C=C/C")
        actual = Chem.Mol(expected)
        bond = next(b for b in actual.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
        bond.SetStereo(Chem.BondStereo.STEREOANY)
        restored = stereo_from_coordinates(actual)
        self.assertTrue(graph_check(expected, actual, derive_stereo=True)["matches"])
        np.testing.assert_array_equal(restored.GetConformer().GetPositions(), actual.GetConformer().GetPositions())
        self.assertEqual([a.GetAtomMapNum() for a in restored.GetAtoms()],
                         [a.GetAtomMapNum() for a in actual.GetAtoms()])
        self.assertFalse(graph_check(expected, embed("C/C=C\\C"), derive_stereo=True)["matches"])

    def test_coordinate_and_graph_masks_are_independent_across_samples(self):
        fixed = torch.tensor([[True, True, False, False], [True, False, True, False]])
        valid = torch.ones((2, 4), dtype=torch.bool)
        times = pose_times([torch.tensor([.2, .4]), torch.tensor([.2, .4])], valid, fixed)
        self.assertTrue(torch.all(times[1] == 1))
        self.assertTrue(torch.all(times[0].squeeze(-1)[fixed] == 1))
        self.assertAlmostEqual(float(times[0][0, 2, 0]), .2)
        self.assertAlmostEqual(float(times[0][1, 1, 0]), .4)
        source = {"coords": torch.arange(24).reshape(2, 4, 3).float(),
                  "atomics": torch.randn(2, 4, 3), "bonds": torch.randn(2, 4, 4, 2),
                  "fragment_mask": valid, "fragment_mode": ["graph", "graph"], "mask": valid}
        predicted = {k: torch.zeros_like(source[k]) for k in ("coords", "atomics", "bonds")}
        output = clamp_pose(source, predicted, fixed, ["atomics", "bonds"])
        torch.testing.assert_close(output["coords"][fixed], source["coords"][fixed])
        self.assertTrue(torch.all(output["coords"][~fixed] == 0))
        torch.testing.assert_close(output["atomics"], source["atomics"])
        torch.testing.assert_close(output["bonds"], source["bonds"])


if __name__ == "__main__":
    unittest.main()
