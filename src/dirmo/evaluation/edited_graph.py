"""Explicit graph edits and graph-only starting poses for a frozen FM study."""
from __future__ import annotations

from collections import deque
import base64

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..chemistry.graphs import graph_check, graph_id, mapped, smiles, stereo_from_coordinates


def unsupported_tensor_features(graph):
    """Features omitted by the pinned FLOWR heavy-atom tensor representation.

    Reject instead of neutralizing a radical or erasing an isotope silently.
    This is a necessary eligibility check, not full round-trip conformance.
    """
    features = []
    for atom in graph.GetAtoms():
        if atom.GetNumRadicalElectrons():
            features.append({"atom_index": atom.GetIdx(), "atom_map": atom.GetAtomMapNum(),
                             "feature": "radical_electrons", "value": atom.GetNumRadicalElectrons()})
        if atom.GetIsotope():
            features.append({"atom_index": atom.GetIdx(), "atom_map": atom.GetAtomMapNum(),
                             "feature": "isotope", "value": atom.GetIsotope()})
    return features


def terminal_edits(native, free_indices):
    """Three fixed, unranked replacements at one terminal carbon: C -> N/O/F.

    Atom count, map IDs, and every bond are retained. Requiring a carbon
    neighbour avoids reactive N-F/O-F attachment motifs in this first study.
    No property, PB, docking score, or FM outcome selects the edit.
    """
    free = set(free_indices)
    options = [a.GetIdx() for a in native.GetAtoms()
               if a.GetIdx() in free and a.GetAtomicNum() == 6 and a.GetDegree() == 1
               and not a.GetIsAromatic() and a.GetFormalCharge() == 0
               and a.GetNeighbors()[0].GetAtomicNum() == 6
               and native.GetBondBetweenAtoms(a.GetIdx(), a.GetNeighbors()[0].GetIdx()).GetBondType() == Chem.BondType.SINGLE]
    if not options:
        raise ValueError("No free terminal C-C carbon for the predefined C->N/O/F edits")
    index = min(options)
    result = [("original", Chem.Mol(native), None)]
    for element in ("N", "O", "F"):
        rw = Chem.RWMol(native)
        atom = Chem.Atom(element)
        atom.SetAtomMapNum(native.GetAtomWithIdx(index).GetAtomMapNum())
        rw.ReplaceAtom(index, atom)
        product = rw.GetMol()
        Chem.SanitizeMol(product)
        # The original pose defines inherited stereo; free XYZ is discarded
        # before any ETKDG starting pose is constructed.
        product = mapped(stereo_from_coordinates(product))
        if graph_id(product) == graph_id(native):
            raise ValueError("Edit did not change the graph")
        for i in range(native.GetNumAtoms()):
            if i not in free:
                left, right = native.GetAtomWithIdx(i), product.GetAtomWithIdx(i)
                if (left.GetAtomicNum(), left.GetFormalCharge(), left.GetIsAromatic()) != (
                        right.GetAtomicNum(), right.GetFormalCharge(), right.GetIsAromatic()):
                    raise ValueError("Edit modified a protected atom")
        result.append((f"C_to_{element}", product, index))
    if len({graph_id(m) for _, m, _ in result}) != 4:
        raise ValueError("The four graphs must be distinct")
    return result


def anchor_indices(mol, free_indices, first, count=4):
    """Pick nearby held graph neighbours; never consult free reference XYZ."""
    held = set(range(mol.GetNumAtoms())) - set(free_indices)
    queue, seen, chosen = deque([first]), set(), []
    while queue and len(chosen) < count:
        index = queue.popleft()
        if index in seen or index not in held:
            continue
        seen.add(index)
        chosen.append(index)
        queue.extend(sorted(a.GetIdx() for a in mol.GetAtomWithIdx(index).GetNeighbors() if a.GetIdx() in held))
    if len(chosen) < 3:
        raise ValueError("Need three connected held atoms to place the graph-only conformer")
    return chosen


def graph_from_record(record):
    if "graph_binary_base64" in record:
        mol = Chem.Mol(base64.b64decode(record["graph_binary_base64"]))
        if mol.GetNumConformers():
            raise ValueError("Stored target graph must not carry native free coordinates")
    else:
        mol = Chem.MolFromSmiles(record["mapped_smiles"])
    if mol is None:
        raise ValueError("Invalid stored graph")
    lookup = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms()}
    mol = Chem.RenumberAtoms(mol, [lookup[m] for m in record["atom_maps"]])
    if graph_id(mol) != record["graph_id"]:
        raise ValueError("Graph serialization changed the target identity")
    return mol


def graph_only_conformer(graph, held_xyz, anchors, seed):
    """ETKDG from graph, fit held anchors, then set supplied held XYZ exactly.

    The function receives no native free coordinates. A prior is paired across
    whole/core FM arms. Only the integration-time coordinate mask differs.
    """
    mol = Chem.Mol(graph)
    mol.RemoveAllConformers()
    mol = Chem.AddHs(mol)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed) % (2**31 - 1)
    parameters.numThreads = 1
    parameters.maxIterations = 1000
    if AllChem.EmbedMolecule(mol, parameters) != 0:
        raise ValueError("Graph-only ETKDG initialization failed")
    mol = Chem.RemoveHs(mol)
    xyz = mol.GetConformer().GetPositions()
    source = xyz[anchors]
    target = np.array([held_xyz[i] for i in anchors])
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    u, singular, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    if singular[1] < 1e-6:
        raise ValueError("Collinear held anchors")
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    xyz = (xyz - source_mean) @ rotation + target_mean
    for i, position in held_xyz.items():
        xyz[i] = position
    mol.GetConformer().SetPositions(xyz)
    if not graph_check(graph, mol, derive_stereo=True)["matches"]:
        raise ValueError("Core placement changed input coordinate stereochemistry")
    mol.SetProp("DIRMO_source", "graph_only_ETKDG_local_held_anchor_fit")
    mol.SetProp("DIRMO_native_free_xyz_used", "false")
    return stereo_from_coordinates(mol)


def graph_record(name, mol, changed_index, free_indices):
    from ..chemistry.properties import graph_properties
    graph = Chem.Mol(mol)
    graph.RemoveAllConformers()
    return {"name": name, "mapped_smiles": smiles(mol, True), "graph_id": graph_id(mol),
            "graph_binary_base64": base64.b64encode(graph.ToBinary()).decode(),
            "graph_transport": "RDKit binary graph without conformers; SMILES is an inspectable label, not a stereo round-trip transport",
            "atom_maps": [a.GetAtomMapNum() for a in mol.GetAtoms()],
            "changed_index": changed_index, "free_indices": list(free_indices),
            "properties": graph_properties(smiles(mol))}


def assess_saved_graph(graph, path, held_xyz):
    from ..chemistry.graphs import read_sdf
    try:
        mol = read_sdf(path)
        if mol.GetNumAtoms() != graph.GetNumAtoms():
            return {"graph_preserved": False, "core_max_drift": None, "core_preserved": False}
        check = graph_check(graph, mol, derive_stereo=True)
        pos = mol.GetConformer().GetPositions()
        drift = max((float(np.linalg.norm(pos[i] - xyz)) for i, xyz in held_xyz.items()), default=0.)
        return {"graph_preserved": check["matches"], "core_max_drift": drift,
                "core_preserved": drift <= .001}
    except Exception as exc:
        return {"graph_preserved": False, "core_max_drift": None,
                "core_preserved": False, "error": f"{type(exc).__name__}: {exc}"}
