"""Verify PB evidence extraction and repair conditions without loading a model."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from dirmo.chemistry.graphs import graph_check, mapped, read_sdf, stereo_from_coordinates, write_sdf
from dirmo.evaluation.pb_guided import boundary_count, events_from_module, noisy_input, preservation, select_regions


def molecule(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 42
    if AllChem.EmbedMolecule(mol, parameters) != 0:
        raise ValueError('Fixture embedding failed')
    return mapped(stereo_from_coordinates(Chem.RemoveHs(mol)))


class PBGuidedTests(unittest.TestCase):
    def test_real_pb_geometry_failure_identifies_the_bad_bond(self):
        from posebusters.modules.distance_geometry import check_geometry
        mol = molecule('CCCOc1ccccc1')
        xyz = mol.GetConformer().GetPositions()
        xyz[0] += np.array([15., 0., 0.])
        mol.GetConformer().SetPositions(xyz)
        module = {'name': 'Geometry', 'function': 'distance_geometry', 'parameters': {
            'threshold_bad_bond_length': .25, 'threshold_bad_angle': .25, 'threshold_clash': .3}}
        output = check_geometry(mol, **module['parameters'])
        self.assertFalse(output['results']['bond_lengths_within_bounds'])
        events = events_from_module(module, output)
        self.assertTrue(any(event['kind'] == 'bond_length' and event['atoms'] == [0, 1] for event in events))

    def test_ring_mask_is_intact_and_random_control_is_size_boundary_matched(self):
        mol = molecule('CCc1ccccc1CC')
        event = {'kind': 'flatness', 'atoms': list(mol.GetRingInfo().AtomRings()[0]), 'severity': 2.}
        selected = select_regions(mol, [event], 42)
        self.assertTrue(selected['eligible'])
        targeted, random = set(selected['targeted_indices']), set(selected['random_indices'])
        self.assertTrue(set(event['atoms']) <= targeted)
        self.assertEqual(len(targeted), len(random))
        self.assertEqual(boundary_count(mol, targeted), boundary_count(mol, random))
        self.assertLess(len(targeted), mol.GetNumAtoms())

    def test_global_failure_is_not_given_an_invented_fragment_score(self):
        result = select_regions(molecule('CCCOCC'), [], 42)
        self.assertFalse(result['eligible'])

    def test_noise_changes_only_free_coordinates_and_preserves_stereochemistry(self):
        mol = molecule('CC[C@H](O)CCC')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.sdf'
            write_sdf(source, mol)
            first, second = root / 'first.sdf', root / 'second.sdf'
            free = [4, 5, 6]
            noisy_input(source, free, 12, first)
            noisy_input(source, free, 13, second)
            original, left, right = read_sdf(source), read_sdf(first), read_sdf(second)
            held = [i for i in range(mol.GetNumAtoms()) if i not in free]
            np.testing.assert_array_equal(original.GetConformer().GetPositions()[held], left.GetConformer().GetPositions()[held])
            self.assertTrue(graph_check(original, left, True)['matches'])
            self.assertFalse(np.array_equal(left.GetConformer().GetPositions()[free], right.GetConformer().GetPositions()[free]))
            contract = preservation(source, first, free)
            self.assertTrue(contract['graph_preserved'] and contract['held_preserved'])
            self.assertEqual(contract['held_max_drift'], 0.)

    def test_whole_pose_control_has_no_fictitious_fixed_atoms(self):
        mol = molecule('CCCOCC')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, perturbed = root / 'source.sdf', root / 'prior.sdf'
            write_sdf(source, mol)
            free = list(range(mol.GetNumAtoms()))
            noisy_input(source, free, 23, perturbed)
            result = preservation(source, perturbed, free)
            self.assertTrue(result['held_preserved'])
            self.assertEqual(result['held_max_drift'], 0.)
            self.assertGreater(result['rmsd_to_initial_generated_pose'], 0.)


if __name__ == '__main__':
    unittest.main()
