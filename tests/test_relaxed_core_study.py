"""A relaxed coordinate mask must not change the graph or required hold set."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark_relaxed_core import expand_one_bond
from dirmo.chemistry.graphs import mapped,write_sdf
from dirmo.evaluation.edited_graph import assess_saved_graph


class RelaxedCoreTests(unittest.TestCase):
    def test_one_hop_does_not_expand_recursively(self):
        free,held,released=expand_one_bond(Chem.MolFromSmiles("CCCCCC"),[0])
        self.assertEqual(free,[0,1]);self.assertEqual(released,[1]);self.assertEqual(held,[2,3,4,5])

    def test_shared_neighbor_is_released_once_without_changing_stereo(self):
        graph=Chem.MolFromSmiles("C[C@H](O)CC")
        before=Chem.MolToSmiles(graph)
        free,held,released=expand_one_bond(graph,[0,2])
        self.assertEqual(released,[1]);self.assertEqual(free,[0,1,2]);self.assertEqual(held,[3,4])
        self.assertEqual(Chem.MolToSmiles(graph),before)

    def test_released_motion_is_separate_from_required_core_violation(self):
        mol=Chem.AddHs(Chem.MolFromSmiles("CCCCCCc1ccccc1"))
        self.assertEqual(AllChem.EmbedMolecule(mol,randomSeed=42),0)
        mol=mapped(Chem.RemoveHs(mol))
        free,held,released=expand_one_bond(mol,[0,1,2])
        xyz=np.round(mol.GetConformer().GetPositions(),4)
        original={i:xyz[i].tolist() for i in range(mol.GetNumAtoms()) if i not in [0,1,2]}
        required={i:xyz[i].tolist() for i in held}
        changed=Chem.Mol(mol); xyz[released]+=np.array([1.,0.,0.]);changed.GetConformer().SetPositions(xyz)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"changed.sdf";write_sdf(path,changed)
            self.assertTrue(assess_saved_graph(mol,path,required)["core_preserved"])
            self.assertFalse(assess_saved_graph(mol,path,original)["core_preserved"])


if __name__=="__main__":unittest.main()
