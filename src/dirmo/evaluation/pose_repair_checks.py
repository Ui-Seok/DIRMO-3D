"""Equivalent atom filtering for large-protein PoseBusters volume checks."""
import numpy as np
from rdkit import Chem


def delete_atoms_batched(mol, indices):
    """Delete the same atoms/order while deferring RDKit index updates once."""
    indices=sorted(indices,reverse=True)
    if not indices:
        return mol
    if mol.GetNumAtoms()>1000 and len(indices)>1000:
        # Even CommitBatchEdit is costly when almost all 67,000 protein atoms
        # are removed. Build the identical retained subset in linear time.
        removed=set(indices)
        retained=[i for i in range(mol.GetNumAtoms()) if i not in removed]
        mapping={old:new for new,old in enumerate(retained)}
        subset=Chem.RWMol()
        for index in retained:
            subset.AddAtom(Chem.Atom(mol.GetAtomWithIdx(index)))
        for bond in mol.GetBonds():
            a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            if a not in mapping or b not in mapping:
                continue
            subset.AddBond(mapping[a],mapping[b],bond.GetBondType())
            copied=subset.GetBondBetweenAtoms(mapping[a],mapping[b])
            copied.SetIsAromatic(bond.GetIsAromatic())
            copied.SetBondDir(bond.GetBondDir())
            stereo_atoms=list(bond.GetStereoAtoms())
            if len(stereo_atoms)==2 and all(i in mapping for i in stereo_atoms):
                copied.SetStereoAtoms(*[mapping[i] for i in stereo_atoms])
                copied.SetStereo(bond.GetStereo())
            for key in bond.GetPropNames(includePrivate=True,includeComputed=False):
                copied.SetProp(key,bond.GetProp(key))
        result=Chem.Mol(subset)
        for key in mol.GetPropNames(includePrivate=True,includeComputed=False):
            result.SetProp(key,mol.GetProp(key))
        for original in mol.GetConformers():
            conformer=Chem.Conformer(len(retained))
            conformer.Set3D(original.Is3D())
            conformer.SetPositions(np.ascontiguousarray(original.GetPositions()[retained]))
            result.AddConformer(conformer,assignId=True)
        return result
    result=Chem.RWMol(mol)
    result.BeginBatchEdit()
    for index in indices:
        result.RemoveAtom(int(index))
    result.CommitBatchEdit()
    return Chem.Mol(result)


def configure_pose_repair_checks():
    # Upstream removes distant/type-filtered protein atoms one at a time, which
    # becomes quadratic for very large PDBs. Keep identical masks, coordinates,
    # atom types, overlap thresholds and ShapeTverskyIndex calculations.
    from posebusters.modules import volume_overlap
    volume_overlap.delete_atoms=delete_atoms_batched
