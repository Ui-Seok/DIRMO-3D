from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..io import digest
from ..schemas import EditCandidate, EditOperation, Protection


class ChemicalError(ValueError):
    pass


def normalize(mol: Chem.Mol) -> Chem.Mol:
    if mol is None:
        raise ChemicalError("invalid_graph")
    mol = Chem.RemoveHs(Chem.Mol(mol))
    # Atom maps identify correspondence, not chemically distinct substituents.
    # Reading mapped 3D SDF can otherwise introduce a false stereocentre when
    # equivalent neighbours have different map IDs (e.g. S(O)(O)O).
    atom_maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception as exc:
        raise ChemicalError(f"invalid_graph: {exc}") from exc
    finally:
        for atom, atom_map in zip(mol.GetAtoms(), atom_maps):
            atom.SetAtomMapNum(atom_map)
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


def stereo_from_coordinates(mol: Chem.Mol) -> Chem.Mol:
    """Infer stereo from unchanged XYZ, ignoring stale MDL E/Z annotations.

    Atom maps are correspondence labels and must not create stereocentres.
    No expected graph or reference coordinates are used to assign the result.
    """
    actual = Chem.Mol(mol)
    if not actual.GetNumConformers():
        raise ChemicalError("3D coordinates required for stereochemistry")
    maps = [atom.GetAtomMapNum() for atom in actual.GetAtoms()]
    for atom in actual.GetAtoms():
        atom.SetAtomMapNum(0)
    for bond in actual.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.AssignStereochemistryFrom3D(actual, replaceExistingTags=True)
    Chem.AssignStereochemistry(actual, cleanIt=True, force=True)
    for atom, atom_map in zip(actual.GetAtoms(), maps):
        atom.SetAtomMapNum(atom_map)
    return actual


def graph_check(expected: Chem.Mol, actual: Chem.Mol, derive_stereo: bool = False) -> dict:
    expected, actual = normalize(expected), normalize(actual)
    if derive_stereo:
        actual = stereo_from_coordinates(actual)
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


EDIT_ELEMENTS = ("C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
_CHANGED = __import__("re").compile(r"measured=\[([0-9,\s]*)\]")


def _attempt(parent, original, protection, operation, guess):
    """Run the real validator on one operation; adopt its measured changed set and retry once.

    Returns ``(product, changed_atom_maps)`` when ``execute_edit`` accepts the operation,
    otherwise None. The changed set comes back so a caller can build a candidate the
    executor will accept without restating a quantity the validator already measures.
    """
    for _ in range(2):
        try:
            product, _ = execute_edit(parent, EditCandidate(
                candidate_id="probe", parent_graph_id=graph_id(parent),
                changed_atom_maps=sorted(guess), operation=operation,
                protection_explanation="executability probe"), original, protection)
            return product, sorted(guess)
        except ChemicalError as exc:
            found = _CHANGED.search(str(exc))
            if not found:
                return None
            guess = [int(v) for v in found.group(1).split(",") if v.strip()]
            if not guess:
                return None
        except Exception:
            return None
    return None


def _single_edit_trials(parent, maps, fresh):
    """Every single operation in the executor's vocabulary, as (operation, guessed change)."""
    for atom_map, index in maps.items():
        atom = parent.GetAtomWithIdx(index)
        for element in EDIT_ELEMENTS:
            if element != atom.GetSymbol():
                yield EditOperation(kind="replace_atom", atom_map=atom_map, element=element), [atom_map]
            for order in (1, 2, 3):
                yield (EditOperation(kind="add_terminal_atom", atom_map=atom_map, element=element,
                                     bond_order=order), [atom_map, fresh])
        yield EditOperation(kind="remove_terminal_atom", atom_map=atom_map), [atom_map]
        for other, other_index in maps.items():
            if other <= atom_map or parent.GetBondBetweenAtoms(index, other_index) is None:
                continue
            for order in (1, 2, 3):
                yield (EditOperation(kind="change_bond", atom_map=atom_map, other_map=other,
                                     bond_order=order), [atom_map, other])


def ranked_edits(parent: Chem.Mol, original: Chem.Mol, protection: Protection, limit: int = 3,
                 max_sa_loss: float | None = None, exclude=()) -> list[EditCandidate]:
    """Executable single edits that raise measured QED, best first, as ready candidates.

    This is an oracle arm for data collection, not a policy. It uses measured graph
    properties to pick edits a planner may miss, so that "is edit value predictable" can
    be separated from "does the planner find good edits". Nothing here participates in a
    routing decision, and the rank and measured gain are recorded on each candidate.
    """
    from .properties import graph_properties
    base = graph_properties(smiles(parent))
    maps = map_indices(parent)
    seen, scored = set(exclude), []
    for operation, guess in _single_edit_trials(parent, maps, max(maps) + 1):
        attempt = _attempt(parent, original, protection, operation, guess)
        if attempt is None:
            continue
        product, changed = attempt
        identity = graph_id(product)
        if identity in seen:
            continue
        seen.add(identity)
        properties = graph_properties(smiles(product))
        if properties["qed"] <= base["qed"]:
            continue
        if max_sa_loss is not None and properties["sa"] - base["sa"] > max_sa_loss:
            continue
        scored.append((properties["qed"] - base["qed"], operation, changed))
    scored.sort(key=lambda row: -row[0])
    return [EditCandidate(
        candidate_id=f"oracle-{rank}-{operation.kind}-{operation.atom_map}",
        parent_graph_id=graph_id(parent), changed_atom_maps=changed, operation=operation,
        protection_explanation=(f"oracle arm, rank {rank} of {len(scored)} executable QED-improving "
                                f"single edits; measured delta_qed={gain:+.4f}; protection unchanged"))
        for rank, (gain, operation, changed) in enumerate(scored[:limit], 1)]


def executable_edits(parent: Chem.Mol, original: Chem.Mol, protection: Protection) -> dict:
    """Every single operation the executor would currently accept on this work item.

    This is the action space, not a ranking: no property is computed and no edit is
    recommended. It exists so a planner proposes operations that can actually run
    instead of guessing at the allowlist, and it is produced by the same
    ``execute_edit`` that will later execute the chosen candidate.
    """
    maps = map_indices(parent)
    fresh = max(maps) + 1
    replace, add, remove, bonds = {}, {}, [], {}
    for atom_map, index in maps.items():
        atom = parent.GetAtomWithIdx(index)
        allowed = [e for e in EDIT_ELEMENTS if e != atom.GetSymbol() and _attempt(
            parent, original, protection,
            EditOperation(kind="replace_atom", atom_map=atom_map, element=e), [atom_map])]
        if allowed:
            replace[str(atom_map)] = allowed
        attach = sorted({f"{e}{order}" for e in EDIT_ELEMENTS for order in (1, 2, 3)
                         if _attempt(parent, original, protection,
                                        EditOperation(kind="add_terminal_atom", atom_map=atom_map,
                                                      element=e, bond_order=order),
                                        [atom_map, fresh])})
        if attach:
            add[str(atom_map)] = attach
        if _attempt(parent, original, protection,
                       EditOperation(kind="remove_terminal_atom", atom_map=atom_map), [atom_map]):
            remove.append(atom_map)
        for other, other_index in maps.items():
            if other <= atom_map or parent.GetBondBetweenAtoms(index, other_index) is None:
                continue
            orders = [order for order in (1, 2, 3) if _attempt(
                parent, original, protection,
                EditOperation(kind="change_bond", atom_map=atom_map, other_map=other,
                              bond_order=order), [atom_map, other])]
            if orders:
                bonds[f"{atom_map}-{other}"] = orders
    return {"replace_atom": replace, "add_terminal_atom": add, "remove_terminal_atom": sorted(remove),
            "change_bond": bonds, "new_atom_map_ids_start_at": fresh,
            "note": "Executable operations only. No property was computed and nothing is recommended."}


def redesigned(parent: Chem.Mol, raw: Chem.Mol, region: list[int]) -> tuple[Chem.Mol, dict]:
    """Accept a fragment-prior output: enforce what the mode promises, measure what it does not.

    The generator chose the freed atoms' elements and bonds, so there is no product graph to
    compare against. What is enforced is the one guarantee the mode offers -- the atoms
    outside the region keep their element -- plus the atom count, since the tensor width is
    fixed. Coordinate drift of the held atoms is reported rather than enforced, because the
    mode does not actually pin it. Freed atoms get fresh map IDs: they are new chemistry.
    """
    parent, raw = mapped(parent), normalize(raw)
    if raw.GetNumAtoms() != parent.GetNumAtoms():
        raise ChemicalError(f"redesign_atom_count_changed: {parent.GetNumAtoms()} to {raw.GetNumAtoms()}")
    index = map_indices(parent)
    missing = [m for m in region if m not in index]
    if missing or not region:
        raise ChemicalError(f"redesign_region_not_on_parent: {missing}")
    free = {index[m] for m in region}
    parent_positions = parent.GetConformer().GetPositions()
    raw_positions = raw.GetConformer().GetPositions()
    fresh, drift = max(index) + 1, []
    for i, atom in enumerate(raw.GetAtoms()):
        if i in free:
            atom.SetAtomMapNum(fresh)
            fresh += 1
            continue
        held = parent.GetAtomWithIdx(i)
        if atom.GetAtomicNum() != held.GetAtomicNum():
            raise ChemicalError(f"redesign_changed_a_held_atom: index {i} "
                                f"{held.GetSymbol()} to {atom.GetSymbol()}")
        atom.SetAtomMapNum(held.GetAtomMapNum())
        drift.append(float(np.linalg.norm(raw_positions[i] - parent_positions[i])))
    if graph_id(raw) == graph_id(parent):
        raise ChemicalError("redesign_returned_the_parent_graph")
    report = {"region": sorted(region), "held_atoms": len(drift),
              "held_rmsd_angstrom": float(np.sqrt(np.mean(np.square(drift)))) if drift else None,
              "held_max_drift_angstrom": max(drift) if drift else None,
              "parent_graph_id": graph_id(parent), "product_graph_id": graph_id(raw),
              "contract": "held atoms keep their element; their coordinate drift is measured, "
                          "not guaranteed, because the mode does not pin it"}
    return raw, report


def candidate_regions(parent: Chem.Mol, protection: Protection, limit: int = 4,
                      max_fraction: float = 0.4) -> list[list[int]]:
    """Regions a redesign action may free, as atom-map lists, whole substituents first.

    A region is a chemically meaningful tip: all the terminal atoms hanging off one heavy
    atom (a substituent), or a single terminal atom. Whole substituents come first because
    one atom is usually returned unchanged, so a small limit should spend itself on the
    regions that can actually differ. ``max_fraction`` keeps the held part in the majority,
    otherwise the action stops being a redesign and becomes de novo generation. Protected
    atoms are never offered. Nothing here predicts which region is worth freeing.
    """
    maps = map_indices(parent)
    protected = set(protection.atom_maps)
    protected.update(v for feature in protection.features if feature.hard for v in feature.atom_maps)
    cap = max(1, int(parent.GetNumAtoms() * max_fraction))
    groups, singles, seen = [], [], set()
    for atom_map, index in sorted(maps.items()):
        tips = sorted({a.GetAtomMapNum() for a in parent.GetAtomWithIdx(index).GetNeighbors()
                       if a.GetDegree() == 1} - protected)
        if 1 < len(tips) <= cap:
            key = frozenset(tips)
            if key not in seen:
                seen.add(key)
                groups.append(tips)
    for atom_map, index in sorted(maps.items()):
        if atom_map in protected or parent.GetAtomWithIdx(index).GetDegree() != 1:
            continue
        key = frozenset({atom_map})
        if key not in seen:
            seen.add(key)
            singles.append([atom_map])
    groups.sort(key=lambda region: (-len(region), region))
    return (groups + singles)[:limit]


def input_conformer(product: Chem.Mol, reference: Chem.Mol, seed: int) -> Chem.Mol:
    """Preparation only. This conformer must NEVER be accepted as an FM result.

    The atoms the edit retained are rigidly aligned onto the parent's measured pose, so
    the edited graph enters generation in the binding mode that was actually evaluated.
    A centroid shift alone leaves the orientation arbitrary, which makes the generator
    rotate the whole ligand and strains the result. Only a rigid transform is applied:
    no coordinate is invented for a retained atom and no force field is run.
    """
    mol = Chem.AddHs(product)
    mol.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = seed % (2**31 - 1)
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ChemicalError("Input conformer embedding failed")
    mol = Chem.RemoveHs(mol)
    product_index, parent_index = map_indices(mol), map_indices(reference)
    retained = [(product_index[m], parent_index[m]) for m in sorted(product_index) if m in parent_index]
    if len(retained) < 3:
        raise ChemicalError("Too few retained atoms to place the edited graph in the pocket frame")
    AllChem.AlignMol(mol, reference, atomMap=retained)
    mol.SetProp("DIRMO_source", "input_conformer_only")
    mol.SetProp("DIRMO_retained_alignment_atoms", str(len(retained)))
    return mol
