"""Read-only provenance audit and local structural-dataset inventory.

Archive membership and exact PDB exclusions are structural checks, not proof of
sequence/ligand-family independence or complete chemical preparation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def processed_name(name):
    protein, ligand = str(name).split('.pdb_', 1)
    return Path(protein).name.replace('_', '-') + '_' + Path(ligand).name.replace('_', '-')


def compare_molecules(source, prepared):
    original = Chem.SDMolSupplier(str(source), removeHs=False)[0]
    saved = Chem.SDMolSupplier(str(prepared), removeHs=False)[0]
    if original is None or saved is None:
        raise ValueError('Cannot parse source or pre-action SDF')
    for molecule in [original, saved]:
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
    same_order = ([a.GetAtomicNum() for a in original.GetAtoms()] ==
                  [a.GetAtomicNum() for a in saved.GetAtoms()])
    same_graph = Chem.MolToSmiles(original) == Chem.MolToSmiles(saved)
    same_connectivity = (Chem.MolToSmiles(original, isomericSmiles=False) ==
                         Chem.MolToSmiles(saved, isomericSmiles=False))
    coordinates = (same_order and original.GetNumConformers() > 0 and
                   saved.GetNumConformers() > 0 and
                   np.allclose(original.GetConformer().GetPositions(), saved.GetConformer().GetPositions(), atol=1e-4, rtol=0))
    return {'same_canonical_isomeric_smiles_ignoring_maps': same_graph,
            'same_connectivity_ignoring_stereo_and_maps': same_connectivity,
            'same_atomic_numbers_and_coordinate_array': bool(coordinates)}


def audit(run, datasets, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((run / 'prepared/rows.json').read_text())
    with np.load(datasets / 'processed_crossdock_noH_full_temp/test.npz') as z:
        test_names = z['names'].tolist()
    test_keys = {processed_name(n) for n in test_names}
    with np.load(datasets / 'processed_crossdock_noH_full_temp/train.npz') as z:
        train_keys = {processed_name(n) for n in z['names']}
    test_pdbs = set()
    for name in test_names:
        protein, ligand = name.split('.pdb_', 1)
        test_pdbs.add(Path(protein).name.split('_')[0].lower())
        test_pdbs.add(Path(ligand).name.split('_')[3].lower())
    states = {}
    for row in rows:
        if row['state_id'] not in states:
            manifest = json.loads(Path(row['snapshot_path']).with_name('manifest.json').read_text())
            case = manifest['case']
            source = Path(case['initial_ligand_path'])
            states[row['state_id']] = {
                'state_id': row['state_id'], 'target_id': row['target_id'],
                'group_id': row['group_id'], 'source_ligand': str(source),
                'source_pocket': case['pocket_path'], 'pre_sdf': row['pre_sdf'],
                'source_exists': source.is_file(),
                'matches_processed_test_index': source.name in test_keys,
                'matches_processed_train_index': source.name in train_keys,
                'source_ligand_sha256': digest(source),
                'pre_sdf_sha256': digest(Path(row['pre_sdf'])),
                **compare_molecules(source, Path(row['pre_sdf']))}
    train_test_names_overlap = sorted(train_keys & test_keys)
    counts = Counter(states[r['state_id']]['matches_processed_test_index'] for r in rows)
    folds = []
    for fold in range(5):
        split = json.loads((run / f'fold_{fold}/split.json').read_text())
        record = {'fold': fold}
        for name in ['train', 'validation', 'test']:
            ids = set(split[name]['state_id'])
            subset = [r for r in rows if r['state_id'] in ids]
            record[name] = {'rows': len(subset), 'states': len(ids),
                            'benchmark_test_derived_rows': sum(states[r['state_id']]['matches_processed_test_index'] for r in subset)}
        folds.append(record)
    (output / 'training_state_sources.json').write_text(json.dumps(list(states.values()), ensure_ascii=False, indent=2) + '\n')
    print('CrossDocked provenance complete; scanning SPINDR archive metadata.', flush=True)
    archive = datasets / 'spindr/spindr.tar.gz?download=1'
    spindr_members = defaultdict(dict)
    with tarfile.open(archive, 'r|gz') as handle:
        for member in handle:
            path = Path(member.name)
            if member.isfile() and path.suffix.lower() in ['.sdf', '.cif'] and len(path.parts) >= 3:
                split, identifier = path.parts[-2], path.stem
                spindr_members[(split, identifier)][path.suffix.lower()] = member.name
    paired_spindr = []
    for (split, identifier), files in sorted(spindr_members.items()):
        if '.sdf' in files and '.cif' in files:
            paired_spindr.append({'split': split, 'system_id': identifier,
                                  'pdb_id': identifier.split('__')[0].lower(),
                                  'crossdock_test_exact_pdb_overlap': identifier.split('__')[0].lower() in test_pdbs,
                                  'ligand_member': files['.sdf'], 'pocket_member': files['.cif']})
    (output / 'spindr_archive_pairs.json').write_text(json.dumps(paired_spindr, indent=2) + '\n')
    print('SPINDR inventory complete; scanning local PLINDER ZIP directories.', flush=True)
    plinder = datasets / 'plinder/2024-06/v2'
    split_table = pq.read_table(plinder / 'splits/split.parquet', columns=['system_id', 'split'])
    split_map = dict(zip(split_table['system_id'].to_pylist(), split_table['split'].to_pylist()))
    local_systems = defaultdict(set)
    for archive_path in sorted((plinder / 'systems').glob('*.zip')):
        with zipfile.ZipFile(archive_path) as handle:
            for name in handle.namelist():
                parts = name.split('/')
                if len(parts) >= 2:
                    if parts[-1] == 'receptor.pdb':
                        local_systems[parts[0]].add('pdb')
                    if '/ligand_files/' in name and name.endswith('.sdf'):
                        local_systems[parts[0]].add('sdf')
    paired_plinder = [identifier for identifier, flags in local_systems.items() if {'pdb', 'sdf'} <= flags]
    plinder_overlap = Counter(split_map.get(s, 'unknown') for s in paired_plinder
                              if s.split('__')[0].lower() in test_pdbs)
    result = {
        'created_at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
        'training_run': str(run), 'training_rows': len(rows), 'training_states': len(states),
        'benchmark_test_derived_rows': counts[True],
        'benchmark_test_derived_states': sum(v['matches_processed_test_index'] for v in states.values()),
        'matching_processed_train_states': sum(v['matches_processed_train_index'] for v in states.values()),
        'source_sdf_byte_identical_to_saved_pre_sdf_states': sum(v['source_ligand_sha256'] == v['pre_sdf_sha256'] for v in states.values()),
        'source_and_pre_sdf_same_graph_states': sum(v['same_canonical_isomeric_smiles_ignoring_maps'] for v in states.values()),
        'source_and_pre_sdf_same_connectivity_states': sum(v['same_connectivity_ignoring_stereo_and_maps'] for v in states.values()),
        'source_and_pre_sdf_same_coordinate_states': sum(v['same_atomic_numbers_and_coordinate_array'] for v in states.values()),
        'processed_test_index_cases': len(test_names),
        'processed_train_test_filename_pair_overlap': len(train_test_names_overlap),
        'test_index_sha256': digest(datasets / 'processed_crossdock_noH_full_temp/test.npz'),
        'current_artifacts_eligible_for_official_crossdocked_test': False,
        'fold_sources': folds,
        'exact_test_receptor_or_ligand_pdb_ids': sorted(test_pdbs),
        'spindr': {
            'raw_archive': str(archive), 'paired_cif_sdf_by_split': dict(Counter(r['split'] for r in paired_spindr)),
            'crossdock_test_exact_pdb_overlap_by_split': dict(Counter(r['split'] for r in paired_spindr if r['crossdock_test_exact_pdb_overlap'])),
            'train_after_exact_test_pdb_exclusion': sum(r['split'] == 'train' and not r['crossdock_test_exact_pdb_overlap'] for r in paired_spindr),
            'full_protein_pdb_sdf_archive': str(datasets / 'spindr/spindr_full.zip?download=1')},
        'plinder': {
            'local_release': str(plinder), 'split_metadata_counts': dict(Counter(split_map.values())),
            'local_systems_with_receptor_pdb_and_ligand_sdf_by_split': dict(Counter(split_map.get(s, 'unknown') for s in paired_plinder)),
            'local_exact_test_pdb_overlap_by_split': dict(plinder_overlap),
            'local_train_after_exact_test_pdb_exclusion': sum(split_map.get(s) == 'train' and s.split('__')[0].lower() not in test_pdbs for s in paired_plinder)},
        'limitations': [
            'CrossDocked membership is confirmed against the locally used processed NPZ split, not a newly fetched upstream split revision.',
            'Matching file names are verified alongside source manifests and SHA256; raw source archives are not modified.',
            'Archive structural pairs are not newly generated Router outcome labels.',
            'Exact PDB exclusion alone does not establish protein/pocket/ligand family independence.',
            'SPINDR archive and PLINDER inventory do not certify chemistry/protonation or complete model input readiness.',
            'No GPU training, FM generation, LLM requests, or dataset downloads were run.']}
    (output / 'audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--datasets', type=Path, default=Path('/LocalData1/uslee/datasets'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.datasets, args.output)
