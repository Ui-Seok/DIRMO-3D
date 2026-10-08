from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..io import digest
from ..schemas import EditCandidate, Protection


class ChemicalError(ValueError):
    pass


def normalize(mol: Chem.Mol) -> Chem.Mol:
    if mol is None:
        raise ChemicalError("invalid_graph")
    mol = Chem.RemoveHs(Chem.Mol(mol))
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception as exc:
        raise ChemicalError(f"invalid_graph: {exc}") from exc
    if len(Chem.GetMolFrags(mol)) != 1 or mol.GetNumAtoms() == 0:
        raise ChemicalError("Disconnected or empty graphs are unsupported")
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ChemicalError("Dummy atoms are unsupported")
    return mol


def read_sdf(path: str | Path) -> Chem.Mol:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    if len(supplier) == 0 or supplier[0] is None:
        raise ChemicalError(f"Cannot parse SDF: {path}")
    mol = normalize(supplier[0])
    if mol.GetNumConformers() != 1 or not mol.GetConformer().Is3D():
        raise ChemicalError("A single 3D conformer in the pocket coordinate frame is required")
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ChemicalError("Non-finite ligand coordinates")
    return mol


def write_sdf(path: str | Path, mol: Chem.Mol):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Chem.SDWriter(str(path)) as writer:
        writer.write(mol)


def mapped(mol: Chem.Mol) -> Chem.Mol:
    mol = normalize(mol)
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if any(maps) and (any(v <= 0 for v in maps) or len(set(maps)) != len(maps)):
        raise ChemicalError("mapping_ambiguous: initial mapping must be complete and unique")
    if not any(maps):
        for i, atom in enumerate(mol.GetAtoms(), 1):
            atom.SetAtomMapNum(i)
    return mol


def smiles(mol: Chem.Mol, keep_maps: bool = False) -> str:
    mol = normalize(mol)
    if not keep_maps:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def graph_id(mol: Chem.Mol) -> str:
    return digest(smiles(mol))


def map_indices(mol: Chem.Mol) -> dict[int, int]:
    result = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
    if 0 in result or len(result) != mol.GetNumAtoms():
        raise ChemicalError("mapping_ambiguous: require unique positive maps on every heavy atom")
    return result


def atom_signature(atom):
    return (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope(),
            atom.GetIsAromatic(), atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None)


def graph_check(expected: Chem.Mol, actual: Chem.Mol, derive_stereo: bool = False) -> dict:
    expected, actual = normalize(expected), normalize(actual)
    if derive_stereo:
        Chem.AssignAtomChiralTagsFromStructure(actual, replaceExistingTags=True)
        Chem.DetectBondStereochemistry(actual)
        Chem.AssignStereochemistry(actual, cleanIt=True, force=True)
    exact = smiles(expected) == smiles(actual)
    # Keep the complete stereograph identity stable. Additional stereospecification
    # is also rejected rather than assigning another graph the requested graph ID.
    matches = actual.GetSubstructMatches(expected, useChirality=True, uniquify=False, maxMatches=4096)
    same_size = expected.GetNumAtoms() == actual.GetNumAtoms() and expected.GetNumBonds() == actual.GetNumBonds()
    checked = [match for match in matches if same_size and all(
        atom_signature(expected.GetAtomWithIdx(i))[:4] == atom_signature(actual.GetAtomWithIdx(j))[:4]
        for i, j in enumerate(match))]
    result = {"matches": bool(checked) and exact, "exact_canonical_identity": exact,
              "expected_smiles": smiles(expected), "actual_smiles": smiles(actual),
              "atom_count": actual.GetNumAtoms(), "isomorphism_count": len(checked),
              "checks": ["element", "atom_count", "connectivity", "bond_order", "formal_charge",
                         "isotope", "specified_stereochemistry"], "stereo_derived_from_coordinates": derive_stereo}
    if checked:
        # Select correspondence in the fixed protein frame; never align the ligand.
        if expected.GetNumConformers() and actual.GetNumConformers():
            ref = expected.GetConformer().GetPositions()
            pos = actual.GetConformer().GetPositions()
            match = min(checked, key=lambda m: float(np.square(ref - pos[list(m)]).sum()))
        else:
            match = checked[0]
        result["expected_index_to_actual_index"] = {i: j for i, j in enumerate(match)}
    return result


def protected_graph(original: Chem.Mol, product: Chem.Mol, protection: Protection):
    orig, prod = map_indices(original), map_indices(product)
    protected = set(protection.atom_maps)
    protected.update(v for feat in protection.features if feat.hard for v in feat.atom_maps)
    for atom_map in protected:
        if atom_map not in orig or atom_map not in prod:
            raise ChemicalError(f"protected_atom_missing:{atom_map}")
        if atom_signature(original.GetAtomWithIdx(orig[atom_map])) != atom_signature(
                product.GetAtomWithIdx(prod[atom_map])):
            raise ChemicalError(f"protected_atom_changed:{atom_map}")
    for a in protected:
        for b in protected:
            ob = original.GetBondBetweenAtoms(orig[a], orig[b])
            pb = product.GetBondBetweenAtoms(prod[a], prod[b])
            if (None if ob is None else (ob.GetBondType(), ob.GetStereo())) != (
                    None if pb is None else (pb.GetBondType(), pb.GetStereo())):
                raise ChemicalError("protected_bond_changed")
    for query in protection.smarts:
        smarts = Chem.MolFromSmarts(query)
        if smarts is None or not original.HasSubstructMatch(smarts, useChirality=True):
            raise ChemicalError("Invalid or absent original protected SMARTS")
        if not product.HasSubstructMatch(smarts, useChirality=True):
            raise ChemicalError("protected_substructure_missing")


def execute_edit(parent: Chem.Mol, candidate: EditCandidate, original: Chem.Mol,
                 protection: Protection) -> tuple[Chem.Mol, dict]:
    parent = mapped(parent)
    if candidate.parent_graph_id != graph_id(parent):
        raise ChemicalError("parent_graph_id_mismatch")
    parent_maps = map_indices(parent)
    if candidate.mapped_product_smiles:
        product = normalize(Chem.MolFromSmiles(candidate.mapped_product_smiles))
    else:
        op = candidate.operation
        if op.atom_map not in parent_maps:
            raise ChemicalError("Unknown atom map")
        rw = Chem.RWMol(parent)
        index = parent_maps[op.atom_map]
        orders = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
        if op.kind in ("replace_atom", "add_terminal_atom"):
            if op.element not in {"C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}:
                raise ChemicalError("Unsupported edit element")
            atom = Chem.Atom(op.element)
            atom.SetFormalCharge(op.formal_charge)
            if op.kind == "replace_atom":
                atom.SetAtomMapNum(op.atom_map)
                rw.ReplaceAtom(index, atom)
            else:
                atom.SetAtomMapNum(max(parent_maps) + 1)
                new_idx = rw.AddAtom(atom)
                rw.AddBond(index, new_idx, orders[op.bond_order])
        elif op.kind == "remove_terminal_atom":
            if rw.GetAtomWithIdx(index).GetDegree() != 1:
                raise ChemicalError("Only terminal atom removal is allowed")
            rw.RemoveAtom(index)
        elif op.kind == "change_bond":
            if op.other_map not in parent_maps:
                raise ChemicalError("Unknown other_map")
            bond = rw.GetBondBetweenAtoms(index, parent_maps[op.other_map])
            if bond is None or bond.GetIsAromatic():
                raise ChemicalError("Only existing non-aromatic bonds can change")
            bond.SetBondType(orders[op.bond_order])
        product = normalize(rw.GetMol())
    product_maps = map_indices(product)
    if any(v <= max(parent_maps) for v in set(product_maps) - set(parent_maps)):
        raise ChemicalError("mapping_ambiguous: new atoms must use fresh map IDs")
    correspondence = {v: v for v in set(parent_maps) & set(product_maps)}
    if candidate.atom_correspondence and candidate.atom_correspondence != correspondence:
        raise ChemicalError("mapping_ambiguous: claimed correspondence differs from retained map IDs")
    changed = set(parent_maps) ^ set(product_maps)
    for v in correspondence:
        old, new = parent.GetAtomWithIdx(parent_maps[v]), product.GetAtomWithIdx(product_maps[v])
        old_neigh = {(a.GetAtomMapNum(), str(parent.GetBondBetweenAtoms(old.GetIdx(), a.GetIdx()).GetBondType()))
                     for a in old.GetNeighbors()}
        new_neigh = {(a.GetAtomMapNum(), str(product.GetBondBetweenAtoms(new.GetIdx(), a.GetIdx()).GetBondType()))
                     for a in new.GetNeighbors()}
        if atom_signature(old) != atom_signature(new) or old_neigh != new_neigh:
            changed.add(v)
    if graph_id(parent) == graph_id(product):
        raise ChemicalError("duplicate_graph: edit did not change the chemical graph")
    if set(candidate.changed_atom_maps) != changed:
        raise ChemicalError(f"changed_atom_maps_mismatch: measured={sorted(changed)}")
    protected_graph(original, product, protection)
    return product, {"parent_to_product_maps": correspondence, "changed_atom_maps": sorted(changed),
                     "mapping_ambiguous": False}


def input_conformer(product: Chem.Mol, reference: Chem.Mol, seed: int) -> Chem.Mol:
    """Preparation only. This conformer must NEVER be accepted as an FM result."""
    mol = Chem.AddHs(product)
    mol.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed % (2**31 - 1)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ChemicalError("Input conformer embedding failed")
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    shift = reference.GetConformer().GetPositions().mean(0) - conf.GetPositions().mean(0)
    for i, xyz in enumerate(conf.GetPositions() + shift):
        conf.SetAtomPosition(i, xyz)
    mol.SetProp("DIRMO_source", "input_conformer_only")
    return mol
