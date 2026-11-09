"""Pre-action region descriptors; no generated product, model call, or atom ID feature."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures


SPATIAL_FEATURE_NAMES = [
    'parent_heavy_count', 'pocket_heavy_count', 'pocket_residue_count',
    'region_fraction', 'region_C_fraction', 'region_N_fraction', 'region_O_fraction',
    'region_S_fraction', 'region_halogen_fraction', 'region_hetero_fraction',
    'region_aromatic_fraction', 'region_ring_fraction', 'region_formal_charge',
    'region_degree_mean', 'region_radius_gyration', 'region_donor_per_atom',
    'region_acceptor_per_atom', 'region_hydrophobe_per_atom',
    'boundary_bond_count', 'boundary_aromatic_bonds', 'boundary_ring_bonds',
    'region_nearest_protein_min', 'region_nearest_protein_mean',
    'region_nearest_protein_max', 'region_nearest_protein_std',
    'region_nearest_protein_fraction_lt3', 'region_nearest_protein_fraction_lt4',
    'region_nearest_protein_fraction_lt5',
    'region_vdw_ratio_min', 'region_vdw_ratio_nearest_mean',
    'region_pairs_lt3_per_atom', 'region_pairs_lt4_per_atom', 'region_pairs_lt5_per_atom',
    'region_severe_overlap_pairs_per_atom', 'neighbor4_unique_atoms_per_region_atom',
    'neighbor4_unique_residues', 'neighbor4_N_fraction', 'neighbor4_O_fraction',
    'neighbor4_S_fraction', 'neighbor4_aromatic_residue_fraction',
    'neighbor4_acidic_residue_fraction', 'neighbor4_basic_residue_fraction',
    'neighbor4_histidine_fraction', 'neighbor4_missing',
    'held_fraction', 'region_to_held_centroid', 'protected_fraction',
    'region_protected_overlap_fraction', 'region_to_protected_min',
    'protected_missing', 'region_max_bond_radius_ratio', 'region_mean_bond_radius_ratio',
]


@lru_cache(maxsize=1)
def factory():
    return ChemicalFeatures.BuildFeatureFactory(str(__import__('pathlib').Path(RDConfig.RDDataDir)/'BaseFeatures.fdef'))


def region_spatial_features(mol, protein_xyz, protein_elements, protein_residues,
                            region_maps, protected_maps=()):
    """All coordinates belong to the *pre-action* parent, in the same pocket frame.

    Protein residue categories are descriptors, not inferred protonation states.
    Distance cutoffs and VdW ratios are features, not validated interaction labels.
    """
    mapping = {a.GetAtomMapNum():a.GetIdx() for a in mol.GetAtoms()}
    region_maps, protected_maps = set(region_maps),set(protected_maps)
    if len(mapping)!=mol.GetNumAtoms() or 0 in mapping or not region_maps or not region_maps<=mapping.keys():
        raise ValueError('Spatial features require a nonempty, uniquely mapped parent region')
    if not protected_maps<=mapping.keys():
        raise ValueError('Protected atom map missing from pre-action parent')
    ids = sorted(mapping[m] for m in region_maps)
    held = sorted(set(range(mol.GetNumAtoms()))-set(ids))
    protected = sorted(mapping[m] for m in protected_maps)
    xyz = mol.GetConformer().GetPositions()
    px = np.asarray(protein_xyz,dtype=float)
    if px.ndim!=2 or px.shape[1]!=3 or len(px)!=len(protein_elements) or len(px)!=len(protein_residues) or not len(px):
        raise ValueError('Missing/malformed pocket coordinates')
    if not np.isfinite(xyz).all() or not np.isfinite(px).all():
        raise ValueError('Nonfinite pre-action coordinates')
    atoms = [mol.GetAtomWithIdx(i) for i in ids]
    n,total = len(ids),mol.GetNumAtoms()
    table = Chem.GetPeriodicTable()
    distances = np.linalg.norm(xyz[ids,None,:]-px[None,:,:],axis=-1)
    nearest = distances.min(axis=1)
    ligand_radii = np.array([table.GetRvdw(a.GetAtomicNum()) for a in atoms])
    protein_radii = np.array([table.GetRvdw(table.GetAtomicNumber(e)) for e in protein_elements])
    ratios = distances/(ligand_radii[:,None]+protein_radii[None,:])
    neighbor = np.any(distances<4.,axis=0)
    neighbors = np.flatnonzero(neighbor)
    residue_names = [r.split(':')[1] for r in np.asarray(protein_residues)[neighbors]]
    unique_residues = set(np.asarray(protein_residues)[neighbors])
    elements = np.asarray(protein_elements)[neighbors]
    def fraction(values, fn):
        return sum(bool(fn(v)) for v in values)/len(values) if len(values) else 0.
    features = factory().GetFeaturesForMol(mol)
    pharmacophore = {
        family:sum(bool(set(f.GetAtomIds()) & set(ids)) for f in features if f.GetFamily()==family)/n
        for family in ['Donor','Acceptor','Hydrophobe']}
    boundary = [b for b in mol.GetBonds() if (b.GetBeginAtomIdx() in ids)!=(b.GetEndAtomIdx() in ids)]
    touched = [b for b in mol.GetBonds() if b.GetBeginAtomIdx() in ids or b.GetEndAtomIdx() in ids]
    bond_ratios = [np.linalg.norm(xyz[b.GetBeginAtomIdx()]-xyz[b.GetEndAtomIdx()])/
                   (table.GetRcovalent(mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomicNum())+
                    table.GetRcovalent(mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomicNum())) for b in touched]
    centroid = xyz[ids].mean(axis=0)
    values = [total,len(px),len(set(protein_residues)),n/total,
        *[fraction(atoms,lambda a,z=z:a.GetAtomicNum()==z) for z in [6,7,8,16]],
        fraction(atoms,lambda a:a.GetAtomicNum() in [9,17,35,53]),
        fraction(atoms,lambda a:a.GetAtomicNum() not in [1,6]),
        fraction(atoms,lambda a:a.GetIsAromatic()),fraction(atoms,lambda a:a.IsInRing()),
        sum(a.GetFormalCharge() for a in atoms),np.mean([a.GetDegree() for a in atoms]),
        np.sqrt(np.mean(np.sum((xyz[ids]-centroid)**2,axis=1))),
        pharmacophore['Donor'],pharmacophore['Acceptor'],pharmacophore['Hydrophobe'],
        len(boundary),sum(b.GetIsAromatic() for b in boundary),sum(b.IsInRing() for b in boundary),
        nearest.min(),nearest.mean(),nearest.max(),nearest.std(),
        *[np.mean(nearest<t) for t in [3.,4.,5.]],
        ratios.min(),ratios[np.arange(n),distances.argmin(axis=1)].mean(),
        *[np.sum(distances<t)/n for t in [3.,4.,5.]],np.sum(ratios<.8)/n,
        len(neighbors)/n,len(unique_residues),
        *[fraction(elements,lambda e,t=t:e==t) for t in ['N','O','S']],
        fraction(residue_names,lambda r:r in ['PHE','TYR','TRP','HIS']),
        fraction(residue_names,lambda r:r in ['ASP','GLU']),
        fraction(residue_names,lambda r:r in ['LYS','ARG']),
        fraction(residue_names,lambda r:r=='HIS'),float(not len(neighbors)),
        len(held)/total,np.linalg.norm(centroid-xyz[held].mean(axis=0)) if held else 0.,
        len(protected)/total,len(region_maps & protected_maps)/n,
        np.linalg.norm(xyz[ids,None,:]-xyz[None,protected,:],axis=-1).min() if protected else 0.,
        float(not protected),max(bond_ratios) if bond_ratios else 0.,np.mean(bond_ratios) if bond_ratios else 0.]
    if len(values)!=len(SPATIAL_FEATURE_NAMES) or not np.isfinite(values).all():
        raise AssertionError('Spatial feature schema/finite invariant failed')
    return [float(v) for v in values]
