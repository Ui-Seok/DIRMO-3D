from functools import lru_cache

from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer


@lru_cache(maxsize=8192)
def graph_properties(canonical_smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError("Invalid graph for property calculation")
    values = {"qed": float(QED.qed(mol)), "sa": float(sascorer.calculateScore(mol)),
              "mw": float(Descriptors.MolWt(mol)), "logp": float(Crippen.MolLogP(mol)),
              "hbd": float(Lipinski.NumHDonors(mol)), "hba": float(Lipinski.NumHAcceptors(mol)),
              "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
              "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol))}
    values["ro5_violations"] = float(sum([values["mw"] > 500, values["logp"] > 5,
                                             values["hbd"] > 5, values["hba"] > 10]))
    return values


@lru_cache(maxsize=1)
def feature_factory():
    return ChemicalFeatures.BuildFeatureFactory(RDConfig.RDDataDir + "/BaseFeatures.fdef")


def pharmacophore(mol: Chem.Mol) -> list[dict]:
    return [{"family": f.GetFamily(), "type": f.GetType(),
             "atom_maps": [mol.GetAtomWithIdx(i).GetAtomMapNum() for i in f.GetAtomIds()],
             "position": list(f.GetPos()), "source": "RDKit BaseFeatures.fdef"}
            for f in feature_factory().GetFeaturesForMol(mol)]
