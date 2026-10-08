from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select, is_aa

from ..chemistry.graphs import read_sdf


def structure(path: str):
    parser = MMCIFParser(QUIET=True) if Path(path).suffix.lower() in (".cif", ".mmcif") else PDBParser(QUIET=True)
    return parser.get_structure("protein", path)


def protein_atoms(path: str) -> tuple[np.ndarray, list[str], list[str]]:
    atoms = [a for r in next(structure(path).get_models()).get_residues() if is_aa(r, standard=False)
             for a in r if a.element not in ("H", "D")]
    if not atoms:
        raise ValueError("No heavy protein atoms found")
    coords = np.array([a.coord for a in atoms], dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite protein coordinates")
    return coords, [a.element.title() for a in atoms], [
        f"{a.parent.parent.id}:{a.parent.resname}:{a.parent.id[1]}{a.parent.id[2].strip()}" for a in atoms]


def prepare_fixed_pocket(source: str, ligand_path: str, output: str, cutoff: float,
                         already_pocket: bool = False) -> dict:
    protein = structure(source)
    pos = read_sdf(ligand_path).GetConformer().GetPositions()
    kept = set()
    for residue in next(protein.get_models()).get_residues():
        if not is_aa(residue, standard=False):
            continue
        xyz = np.array([a.coord for a in residue if a.element not in ("H", "D")])
        if xyz.size and (already_pocket or np.min(np.linalg.norm(xyz[:, None] - pos[None, :], axis=-1)) <= cutoff):
            kept.add(residue.full_id)
    if not kept:
        raise ValueError("No pocket residues within the initial ligand cutoff; check alignment")

    class Selection(Select):
        def accept_model(self, model):
            return model.id == next(protein.get_models()).id

        def accept_residue(self, residue):
            return residue.full_id in kept

        def accept_atom(self, atom):
            return not atom.is_disordered() or atom.get_altloc() in ("A", " ")

    io = PDBIO()
    io.set_structure(protein)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    io.save(output, Selection())
    return {"source": source, "fixed_pocket": output, "residues": len(kept), "cutoff_angstrom": cutoff,
            "already_pocket": already_pocket, "coordinate_frame": "unchanged protein frame",
            "preparation": "first model; protein residues; alternate A/blank; no protonation/repair"}
