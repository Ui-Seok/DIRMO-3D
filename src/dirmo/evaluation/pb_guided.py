"""PoseBusters evidence localization and reproducible pose-only repair masks.

Evidence comes from the exact configured PB modules. It is not a fragment PB
score. Global-only failures remain explicitly unlocalized.
"""
from __future__ import annotations

from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from ..chemistry.graphs import graph_check, read_sdf, stereo_from_coordinates, write_sdf


def events_from_module(module, output):
    params = module.get('parameters', {})
    function = module['function']
    details = output.get('details')
    events = []

    def add(kind, atoms, severity):
        events.append({'kind': kind, 'atoms': sorted(set(int(i) for i in atoms)),
                       'severity': float(severity), 'module': module['name']})

    if function == 'distance_geometry' and isinstance(details, dict):
        for key, column, threshold, sense, kind in (
                ('bonds', 'percent_error', params.get('threshold_bad_bond_length', .2), 'abs', 'bond_length'),
                ('angles', 'bound_absolute_percent_error', params.get('threshold_bad_angle', .2), 'upper', 'bond_angle'),
                ('clash', 'bound_percent_error', params.get('threshold_clash', .2), 'lower', 'internal_clash')):
            frame = details.get(key)
            if frame is None:
                continue
            for _, row in frame.iterrows():
                value = float(row[column])
                bad = abs(value) > threshold if sense == 'abs' else value > threshold if sense == 'upper' else value < -threshold
                if bad:
                    atoms = row['angle'] if key == 'angles' else row['atom_pair']
                    add(kind, atoms, abs(value) / threshold)
    elif function == 'intermolecular_distance' and details is not None:
        for _, row in details.iterrows():
            if bool(row['clash']):
                cutoff = params.get('clash_cutoff', .75)
                add('intermolecular_clash', [row['ligand_atom_id']], max(0., cutoff - float(row['relative_distance'])) / cutoff)
    elif function == 'flatness' and details is not None:
        for atoms, passed, distance in zip(details['planar_group'], details['flatness_passes'], details['max_distance']):
            if not passed:
                add('flatness', atoms, float(distance) / params.get('threshold_flatness', .1))
    return events


def configure_cpu():
    from posebusters.modules import energy_ratio
    from .pose_repair_checks import configure_pose_repair_checks
    if not getattr(energy_ratio.get_average_energy, '_dirmo_single_thread', False):
        original = energy_ratio.get_average_energy

        @lru_cache(maxsize=1024)
        def average(inchi, n_confs=50, num_threads=0):
            cache_root = os.environ.get('DIRMO_PB_ENERGY_CACHE')
            if not cache_root:
                return original(inchi, n_confs, 1)
            from rdkit import rdBase
            from ..io import digest, sha256_file, write_json
            identity = {'inchi': inchi, 'n_confs': n_confs, 'threads': 1,
                        'rdkit': rdBase.rdkitVersion, 'module_sha256': sha256_file(energy_ratio.__file__)}
            path = Path(cache_root) / (digest(identity) + '.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    cached = json.loads(path.read_text())
                    if cached['identity'] == identity and np.isfinite(cached['energy']):
                        return cached['energy']
                except (OSError, ValueError, KeyError):
                    pass
                value = float(original(inchi, n_confs, 1))
                if np.isfinite(value):
                    write_json(path, {'identity': identity, 'energy': value})
                return value

        average._dirmo_single_thread = True
        energy_ratio.get_average_energy = average
    configure_pose_repair_checks()


@lru_cache(maxsize=4)
def protein_mol(path):
    from posebusters import PoseBusters
    from posebusters.tools.loading import safe_load_mol
    settings = PoseBusters(config='dock').config.get('loading', {}).get('mol_cond', {})
    return safe_load_mol(Path(path), **settings)


def measure(path, pocket, *, diagnose=False):
    from posebusters import PoseBusters
    import pandas as pd
    configure_cpu()
    try:
        mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=True)))
        if mol is None:
            raise ValueError('RDKit could not load the saved molecule')
        checker = PoseBusters(config='dock')
        events, detail_errors = [], []
        if diagnose:
            initialize = checker._initialize_modules

            def initialize_with_details():
                # Upstream rebuilds module functions at the beginning of bust().
                initialize()
                for index, (module, function) in enumerate(zip(checker.config['modules'], checker.module_func)):
                    def capture(_function=function, _module=module, **kwargs):
                        result = _function(**kwargs)
                        try:
                            events.extend(events_from_module(_module, result))
                        except Exception as exc:
                            detail_errors.append(f"{_module['name']}: {type(exc).__name__}: {exc}")
                        return result
                    checker.module_func[index] = capture

            checker._initialize_modules = initialize_with_details
        frame = checker.bust(mol_pred=mol, mol_true=None, mol_cond=protein_mol(str(pocket)), full_report=True)
        row = frame.iloc[0]
        keys = []
        for module in checker.config['modules']:
            for name in module.get('chosen_binary_test_output', []):
                rename = module.get('rename_outputs', {}).get(name, name + module.get('rename_suffix', ''))
                keys.append(rename.lower().replace(' ', '_'))
        checks = {key: bool(row[key]) if key in row and not pd.isna(row[key]) else None for key in keys}
        result = {'passes': bool(checks) and all(value is True for value in checks.values()),
                  'checks': checks, 'failed_checks': [key for key, value in checks.items() if value is False],
                  'missing_checks': [key for key, value in checks.items() if value is None], 'sdf': str(path),
                  'continuous': {str(key): float(value) if np.isfinite(value) else None
                                 for key, value in row.items() if key not in checks and isinstance(value, (float, np.floating))}}
        if diagnose:
            result.update(events=events, localization_errors=detail_errors)
        return result
    except Exception as exc:
        return {'passes': False, 'missing': True, 'checks': {}, 'error': f'{type(exc).__name__}: {exc}', 'sdf': str(path)}


def boundary_count(mol, region):
    return sum((bond.GetBeginAtomIdx() in region) != (bond.GetEndAtomIdx() in region) for bond in mol.GetBonds())


def candidate_regions(mol):
    n = mol.GetNumAtoms()
    cap = min(n - 1, max(8, min(12, round(.4 * n))))
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]

    def close_rings(indices):
        indices = set(indices)
        while True:
            expanded = indices | set().union(*(ring for ring in rings if ring & indices))
            if expanded == indices:
                return indices
            indices = expanded

    candidates = set()
    for bond in mol.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        cut = Chem.RWMol(mol)
        cut.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for component in Chem.GetMolFrags(cut.GetMol(), sanitizeFrags=False):
            if 2 <= len(component) <= cap:
                candidates.add(tuple(sorted(component)))
    for atom in mol.GetAtoms():
        region = close_rings({atom.GetIdx()} | {neighbor.GetIdx() for neighbor in atom.GetNeighbors()})
        if 2 <= len(region) <= cap:
            candidates.add(tuple(sorted(region)))
    return [set(region) for region in sorted(candidates)]


def select_regions(mol, events, seed):
    candidates = candidate_regions(mol)
    if not events:
        return {'eligible': False, 'reason': 'No localized PB failure; global energy/overlap/loading failures are not assigned to a fragment'}
    if not candidates:
        return {'eligible': False, 'reason': 'No eligible connected ring-preserving local region'}

    def rank(region):
        covered = [event for event in events if region.intersection(event['atoms'])]
        return (-len(covered), -sum(event['severity'] for event in covered), len(region), boundary_count(mol, region), tuple(sorted(region)))

    targeted = min(candidates, key=rank)
    if not any(targeted.intersection(event['atoms']) for event in events):
        return {'eligible': False, 'reason': 'No candidate region contains a diagnosed failing atom'}
    exact = [r for r in candidates if len(r) == len(targeted) and boundary_count(mol, r) == boundary_count(mol, targeted)]
    alternatives = [r for r in exact if r != targeted]
    # A random selector may legitimately draw the targeted region. Never silently
    # choose a different-sized region and call it a size-matched control.
    pool = exact
    random = pool[int(np.random.default_rng(seed).integers(len(pool)))]
    return {'eligible': True, 'targeted_indices': sorted(targeted), 'random_indices': sorted(random),
            'whole_indices': list(range(mol.GetNumAtoms())), 'candidate_regions': [sorted(r) for r in candidates],
            'random_matching_regions': len(pool), 'distinct_random_alternatives': len(alternatives),
            'random_equals_targeted': random == targeted, 'boundary_bonds': boundary_count(mol, targeted),
            'covered_events': sum(bool(targeted.intersection(event['atoms'])) for event in events),
            'total_events': len(events), 'selection': 'Most PB events covered, severity sum, then smallest ring-preserving connected region; exact atom-count/boundary-matched random control'}


def noisy_input(source, free, seed, destination, sigma=.3):
    original = read_sdf(source)
    free = sorted(free)
    generator = np.random.default_rng(seed)
    positions = original.GetConformer().GetPositions()
    for attempt in range(64):
        candidate = Chem.Mol(original)
        # Generate a full vector even for a local mask: arms share paired random
        # vectors for a candidate seed, and differ only in which atoms are freed.
        noise = generator.normal(0, sigma, size=positions.shape)
        xyz = positions.copy()
        xyz[free] += noise[free]
        candidate.GetConformer().SetPositions(xyz)
        if graph_check(original, candidate, derive_stereo=True)['matches']:
            write_sdf(destination, stereo_from_coordinates(candidate))
            if graph_check(original, read_sdf(destination), derive_stereo=True)['matches']:
                return {'seed': seed, 'sigma_angstrom': sigma, 'rejected_noise_draws': attempt,
                        'free_indices': free, 'input_sdf': str(destination)}
    raise ValueError('Could not initialize free coordinates while preserving generated stereochemistry')


def uff_control(source, free, destination):
    mol = Chem.AddHs(read_sdf(source), addCoords=True)
    fixed = [i for i in range(Chem.RemoveHs(mol).GetNumAtoms()) if i not in set(free)]
    field = AllChem.UFFGetMoleculeForceField(mol)
    if field is None:
        raise ValueError('UFF unavailable')
    for index in fixed:
        field.AddFixedPoint(index)
    convergence = field.Minimize(maxIts=200)
    write_sdf(destination, stereo_from_coordinates(Chem.RemoveHs(mol)))
    return {'uff_convergence_code': convergence, 'definition': 'Ligand-only UFF; same targeted held atoms fixed; max 200 iterations'}


def preservation(source, output, free):
    try:
        original, product = read_sdf(source), read_sdf(output)
        graph = graph_check(original, product, derive_stereo=True)['matches']
        if original.GetNumAtoms() != product.GetNumAtoms():
            return {'graph_preserved': False, 'held_preserved': False}
        held = [i for i in range(original.GetNumAtoms()) if i not in set(free)]
        delta = np.linalg.norm(product.GetConformer().GetPositions() - original.GetConformer().GetPositions(), axis=1)
        drift = float(delta[held].max()) if held else 0.
        return {'graph_preserved': graph, 'held_preserved': drift <= .001, 'held_max_drift': drift,
                'rmsd_to_initial_generated_pose': float(np.sqrt(np.mean(delta ** 2))),
                'note': 'Displacement from generated input, not accuracy against an experimental native pose'}
    except Exception as exc:
        return {'graph_preserved': False, 'held_preserved': False, 'error': f'{type(exc).__name__}: {exc}'}
