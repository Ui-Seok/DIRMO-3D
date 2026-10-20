"""Check metric denominators and distinctions needed for the SBDD benchmark."""
import unittest

from rdkit import Chem
from rdkit.Chem import AllChem

from dirmo.evaluation.crossdocked_sbdd import canonical, distribution_comparison, geometry, metric_summary, strain


class CrossDockedSBDDTests(unittest.TestCase):
    def test_failed_generation_and_missing_measurement_stay_distinct(self):
        rows = [dict(rdkit_valid=False),
                dict(rdkit_valid=True, connected=True, smiles='CC', pb=dict(passes=True)),
                dict(rdkit_valid=True, connected=True, smiles='CCC', pb=dict(passes=False, missing=True))]
        result = metric_summary(rows)
        self.assertEqual(result['scheduled'], 3)
        self.assertEqual(result['pb_pass_per_attempt'], 1 / 3)
        self.assertEqual(result['pb_pass_per_rdkit_valid'], 1 / 2)
        self.assertEqual(result['pb_pass_per_measured'], 1.)
        self.assertEqual(result['pb_missing_on_valid'], 1)

    def test_missing_vina_does_not_count_as_affinity_loss_or_gain(self):
        rows = [dict(rdkit_valid=True, connected=True, smiles='CC', vina=dict(score=-7.)),
                dict(rdkit_valid=True, connected=True, smiles='CCC', vina=dict(status='missing'))]
        result = metric_summary(rows, dict(vina=dict(score=-6.)))
        self.assertEqual(result['high_affinity']['score']['n'], 1)
        self.assertEqual(result['high_affinity']['score']['improved'], 1)
        self.assertEqual(result['high_affinity']['dock']['n'], 0)

    def test_geometry_distribution_and_strain_are_defined_for_a_real_3d_molecule(self):
        molecule = Chem.AddHs(Chem.MolFromSmiles('CCCO'))
        self.assertEqual(AllChem.EmbedMolecule(molecule, randomSeed=42), 0)
        molecule = Chem.RemoveHs(molecule)
        row = dict(geometry=geometry(molecule))
        result = distribution_comparison([row], [row])
        self.assertEqual(result['bond_lengths']['pooled_w1'], 0.)
        self.assertEqual(result['bond_angles']['pooled_w1'], 0.)
        self.assertGreaterEqual(strain(molecule)['strain_kcal_mol'], 0.)

    def test_novelty_smiles_ignore_atom_maps_but_keep_stereochemistry(self):
        self.assertEqual(canonical(Chem.MolFromSmiles('[CH3:1][OH:2]')), 'CO')
        self.assertNotEqual(canonical(Chem.MolFromSmiles('C[C@H](F)Cl')),
                            canonical(Chem.MolFromSmiles('C[C@@H](F)Cl')))


if __name__ == '__main__':
    unittest.main()
