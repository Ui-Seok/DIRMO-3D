"""Visualize saved, paired FM/core -> pocket-aware UFF boundary failures.

No FM/LLM calls, docking, full PB evaluation, or force-field optimization.
Only PB's fast distance-geometry module is rerun to localize bond violations.
The complete PB decisions and Vina scores are read from the original records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from rdkit import Chem, RDLogger
from scipy.spatial import cKDTree

EXP = Path('/LocalData1/uslee/experiments/DIRMO-3D')
STAGES = ('native', 'fm', 'uff')


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(value):
    return value is True or value == 'True'


def molecule(path):
    mol = Chem.MolFromMolFile(str(path), removeHs=False)
    if mol is None or mol.GetNumConformers() != 1:
        raise ValueError(f'Cannot read single-conformer SDF: {path}')
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        raise ValueError(f'Expected original heavy-atom order; explicit H present: {path}')
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError(f'Nonfinite coordinates: {path}')
    return mol


def topology(mol):
    # Older generated SDFs omit atom-map annotations. They retain input atom
    # order; check indexed chemical topology rather than optional map labels.
    return ([(a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope()) for a in mol.GetAtoms()],
            sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                    max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType())) for b in mol.GetBonds()))


def boundary_lengths(mols, fixed):
    coords = [m.GetConformer().GetPositions() for m in mols]
    result = []
    for b in mols[0].GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if bool(fixed[i]) == bool(fixed[j]):
            continue
        lengths = [float(np.linalg.norm(p[i] - p[j])) for p in coords]
        result.append({'indices_0based': [i, j], 'labels_1based': [i + 1, j + 1],
                       'atom_maps': [mols[0].GetAtomWithIdx(k).GetAtomMapNum() for k in (i, j)],
                       'elements': [mols[0].GetAtomWithIdx(k).GetSymbol() for k in (i, j)],
                       'lengths_A': dict(zip(STAGES, lengths)),
                       'fm_minus_native_A': lengths[1] - lengths[0],
                       'uff_minus_native_A': lengths[2] - lengths[0]})
    return result


def collect_candidates(fm_run, refinement_run, min_vina_gap, min_boundary_error):
    from posebusters import PoseBusters
    parameters = next(m['parameters'] for m in PoseBusters(config='dock').config['modules']
                      if m['function'] == 'distance_geometry')
    with (fm_run / 'metrics.csv').open(newline='') as stream:
        records = list(csv.DictReader(stream))
    native = {r['case_id']: r for r in records if r['variant'] == 'original' and r['kind'] == 'native'}
    refined = {(r['case_id'], int(r['draw'])): r
               for path in sorted((refinement_run / 'cases').glob('*/rows.json'))
               for r in load(path) if r['arm'] == 'uff_pkt'}
    counts = dict(fm_slots=0, pb_recovered_graph_held=0, vina_complete=0,
                  native_valid_bad_vina=0, boundary_recovered=0, boundary_pb_violation=0)
    candidates, native_cache = [], {}
    for r in records:
        if r['variant'] != 'original' or r['kind'] != 'fm_core':
            continue
        counts['fm_slots'] += 1
        cid, draw = r['case_id'], int(r['draw'])
        u, n = refined.get((cid, draw)), native[cid]
        if (not u or r['status'] != 'ok' or truth(r['pb_missing']) or truth(r['pb_valid'])
                or not truth(r['graph_preserved']) or not truth(u.get('pb_valid'))
                or not truth(u.get('graph_preserved')) or u.get('status') != 'ok'
                or float(u['held_max_drift']) > .001):
            continue
        counts['pb_recovered_graph_held'] += 1
        if any(x.get('vina_status') != 'ok' or x.get('vina') in (None, '') for x in (r, n, u)):
            continue
        counts['vina_complete'] += 1
        vn, vf, vu = [float(x['vina']) for x in (n, r, u)]
        if not truth(n['pb_valid']) or vf - vn < min_vina_gap:
            continue
        counts['native_valid_bad_vina'] += 1
        paths = {'native': n['sdf'], 'fm': r['sdf'],
                 'uff': str(refinement_run / 'cases' / cid / f'{draw:02d}_uff_pkt.sdf')}
        if cid not in native_cache:
            native_cache[cid] = molecule(paths['native'])
        mols = [native_cache[cid], molecule(paths['fm']), molecule(paths['uff'])]
        if any(topology(m) != topology(mols[0]) for m in mols[1:]):
            raise ValueError(f'Atom order / topology mismatch: {cid}/{draw}')
        condition_path = Path(paths['fm']).parent / 'condition.json'
        condition = load(condition_path)
        fixed = condition['coordinate_fixed_mask']
        if len(fixed) != mols[0].GetNumAtoms() or not all(condition['graph_fixed_mask']):
            raise ValueError(f'Unexpected conditioning contract: {cid}/{draw}')
        bonds = boundary_lengths(mols, fixed)
        if not bonds:
            continue
        worst = max(bonds, key=lambda b: abs(b['fm_minus_native_A']))
        before = max(abs(b['fm_minus_native_A']) for b in bonds)
        after = max(abs(b['uff_minus_native_A']) for b in bonds)
        if before < min_boundary_error or after >= before:
            continue
        counts['boundary_recovered'] += 1
        details = pb_bond_diagnostics(mols[1], parameters)
        actual_violation = any(details[tuple(sorted(b['indices_0based']))]['pb_bond_violation'] for b in bonds)
        counts['boundary_pb_violation'] += int(actual_violation)
        candidates.append({'case_id': cid, 'draw': draw, 'atoms': mols[0].GetNumAtoms(),
                           'boundary_pb_violation': actual_violation,
                           'vina_native': vn, 'vina_fm': vf, 'vina_uff': vu,
                           'vina_fm_minus_native': vf - vn, 'vina_uff_minus_fm': vu - vf,
                           'boundary_before_A': before, 'boundary_after_A': after,
                           'worst_signed_boundary_error_A': worst['fm_minus_native_A'],
                           'fm_failed_checks': r['failed_checks'], 'paths': paths,
                           'condition_path': str(condition_path), 'fixed_mask': fixed,
                           'boundary_bonds': bonds, 'fm_record': r, 'uff_record': u})
    return candidates, counts


def select_cases(candidates, number, requested):
    candidates = [c for c in candidates if c['boundary_pb_violation']]
    if requested:
        if len(requested) not in range(3, 6):
            raise ValueError('--case must specify 3 to 5 distinct pockets')
        selected = []
        for label in requested:
            cid, draw = label.split(':')
            found = next((c for c in candidates if c['case_id'] == cid and c['draw'] == int(draw)), None)
            if found is None:
                raise ValueError(f'Requested case does not satisfy selection filters: {label}')
            selected.append({**found, 'selection_reason': 'explicit_user_selection'})
        if len({c['case_id'] for c in selected}) != len(selected):
            raise ValueError('Choose distinct pockets')
        return selected
    selected, used = [], set()
    profiles = [
        ('stretched_boundary', lambda c: c['worst_signed_boundary_error_A'] > 0,
         lambda c: -c['worst_signed_boundary_error_A']),
        ('compressed_boundary', lambda c: c['worst_signed_boundary_error_A'] < 0,
         lambda c: c['worst_signed_boundary_error_A']),
        ('largest_vina_recovery', lambda c: c['vina_uff_minus_fm'] < 0,
         lambda c: c['vina_uff_minus_fm']),
        ('pb_recovery_without_vina_recovery', lambda c: c['vina_uff_minus_fm'] > 0,
         lambda c: -c['vina_uff_minus_fm']),
        ('additional_vina_recovery', lambda c: c['vina_uff_minus_fm'] < 0,
         lambda c: c['vina_uff_minus_fm']),
    ]
    for reason, predicate, rank in profiles[:number]:
        pool = [c for c in candidates if c['case_id'] not in used and predicate(c)]
        if not pool:
            raise ValueError(f'No distinct-pocket example for profile {reason}; use explicit --case')
        c = min(pool, key=lambda c: (rank(c), c['case_id'], c['draw']))
        selected.append({**c, 'selection_reason': reason})
        used.add(c['case_id'])
    return selected


def pb_bond_diagnostics(mol, parameters):
    from posebusters.modules.distance_geometry import check_geometry
    result = check_geometry(mol, **parameters)
    frame = result['details']['bonds']
    return {tuple(int(i) for i in r['atom_pair']): {
        'lower_bound_A': float(r['lower_bound']), 'upper_bound_A': float(r['upper_bound']),
        'relative_bound_error': float(r['percent_error']),
        'pb_bond_violation': bool(abs(r['percent_error']) > parameters['threshold_bad_bond_length'])}
        for _, r in frame.iterrows()}


def pocket_data(path):
    # Read ATOM/HETATM coordinates directly; never recenter or relax the pocket.
    atoms = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(('ATOM  ', 'HETATM')):
            continue
        element = line[76:78].strip().capitalize()
        if not element:
            element = line[12:16].strip().lstrip('0123456789')[:1].upper()
        if element in ('H', 'D') or line[16:17] not in (' ', 'A'):
            continue
        atoms.append({'xyz': [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                      'element': element, 'residue': f'{line[17:20].strip()} {line[21:22]}{line[22:27].strip()}',
                      'name': line[12:16].strip(), 'serial': int(line[6:11])})
    if not atoms:
        raise ValueError(f'Empty pocket: {path}')
    xyz = np.array([a['xyz'] for a in atoms])
    table = Chem.GetPeriodicTable()
    radii = np.array([table.GetRcovalent(a['element']) for a in atoms])
    # These lines are only a display aid, not a chemistry/evaluation input.
    bonds = [[int(i), int(j)] for i, j in sorted(cKDTree(xyz).query_pairs(2.3))
             if .45 < np.linalg.norm(xyz[i] - xyz[j]) < 1.2 * (radii[i] + radii[j])]
    return {'atoms': atoms, 'bonds': bonds, 'display_bonds': 'distance-inferred only; not evaluation topology'}


def assemble_cases(selected, fm_run, refinement_run, output):
    from posebusters import PoseBusters
    from posebusters.modules import distance_geometry
    parameters = next(m['parameters'] for m in PoseBusters(config='dock').config['modules']
                      if m['function'] == 'distance_geometry')
    cases = []
    for candidate in selected:
        c = dict(candidate)
        cid, draw = c['case_id'], c['draw']
        folder = output / 'cases' / f'{cid}_draw{draw:02d}'
        folder.mkdir(parents=True)
        config_path = fm_run / 'cases' / cid / 'case.json'
        config = load(config_path)
        pocket_path = Path(config['pocket_path'])
        evaluation_path = fm_run / 'evaluations' / cid / 'result.json'
        evaluation = load(evaluation_path)['rows']
        evaluated = {'native': evaluation['native'], 'fm': evaluation[f'original/{draw}/fm_core']}
        coords = {}
        mols = [molecule(c['paths'][s]) for s in STAGES]
        from dirmo.chemistry.graphs import graph_check
        c['graph_validation'] = {s: graph_check(mols[0], m, derive_stereo=True)
                                  for s, m in zip(STAGES, mols)}
        if any(not value['matches'] for value in c['graph_validation'].values()):
            raise ValueError(f'Graph/stereochemistry check failed: {cid}/{draw}')
        c['pb_bond_threshold'] = float(parameters['threshold_bad_bond_length'])
        for stage, mol in zip(STAGES, mols):
            target = folder / f'{stage}.sdf'
            shutil.copyfile(c['paths'][stage], target)
            coords[stage] = mol.GetConformer().GetPositions().tolist()
        c['atoms_data'] = [{'element': a.GetSymbol(), 'map': a.GetAtomMapNum(), 'label': a.GetIdx() + 1,
                            'fixed': bool(c['fixed_mask'][a.GetIdx()])} for a in mols[0].GetAtoms()]
        c['bonds'] = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), float(b.GetBondTypeAsDouble())]
                      for b in mols[0].GetBonds()]
        c['coordinates'] = coords
        c['pocket'] = pocket_data(pocket_path)
        c['pb'] = {stage: evaluated[stage]['pb'] for stage in ('native', 'fm')}
        if (truth(c['pb']['fm']['passes']) or not truth(c['pb']['native']['passes'])
                or c['pb']['fm'].get('missing_checks')):
            raise ValueError(f'PB record conflict: {cid}/{draw}')
        # Saved refined PB cache is indexed by SDF/pocket hashes + evaluator hash.
        # Recover its original record without invoking a scorer or mutating its cache.
        from dirmo.io import digest
        from dirmo.evaluation import pb_guided
        key = digest({'sdf_sha256': sha(c['paths']['uff']), 'pocket_sha256': sha(pocket_path),
                      'diagnose': False, 'evaluator_sha256': sha(pb_guided.__file__)})
        cache_path = refinement_run / 'cache/measurements' / f'{key}.json'
        if not cache_path.exists():
            # Old evaluator revisions may have different hash keys; exact saved
            # SDF content is still checked before reusing an alternative record.
            raise FileNotFoundError(f'Exact saved PB cache not found: {cache_path}')
        c['pb']['uff'] = load(cache_path)
        if not c['pb']['uff']['passes'] or c['pb']['uff'].get('missing_checks'):
            raise ValueError(f'Refined PB cache disagrees with rows.json: {cid}/{draw}')
        c['pb_check_counts'] = {s: {'passed': sum(v is True for v in c['pb'][s]['checks'].values()),
                                    'total': len(c['pb'][s]['checks'])} for s in STAGES}
        diagnostics = {s: pb_bond_diagnostics(m, parameters) for s, m in zip(STAGES, mols)}
        for b in c['boundary_bonds']:
            pair = tuple(sorted(b['indices_0based']))
            b['pb_bond_details'] = {s: diagnostics[s][pair] for s in STAGES}
        if not any(b['pb_bond_details']['fm']['pb_bond_violation'] for b in c['boundary_bonds']):
            raise ValueError(f'No actual PB bond violation at boundary: {cid}/{draw}')
        fixed = np.array(c['fixed_mask'], dtype=bool)
        c['held_max_drift_A'] = {s: float(np.linalg.norm(np.array(coords[s])[fixed] -
                                                           np.array(coords['native'])[fixed], axis=1).max())
                                 for s in STAGES}
        if any(c['held_max_drift_A'][s] > .001 for s in ('fm', 'uff')):
            raise ValueError(f'Saved coordinate fixation contract fails: {cid}/{draw}')
        c['free_rmsd_uff_vs_fm_A'] = float(np.sqrt(np.mean(np.sum(
            (np.array(coords['uff'])[~fixed] - np.array(coords['fm'])[~fixed]) ** 2, axis=1))))
        c['source'] = {'fm_case': str(config_path), 'evaluation': str(evaluation_path),
                       'refinement_rows': str(refinement_run / 'cases' / cid / 'rows.json'),
                       'pb_uff_cache': str(cache_path), 'condition': c['condition_path'],
                       'pocket': str(pocket_path), **c['paths']}
        c['sha256'] = {k: sha(v) for k, v in c['source'].items()}
        shutil.copyfile(pocket_path, folder / 'pocket.pdb')
        shutil.copyfile(c['condition_path'], folder / 'condition.json')
        dump(folder / 'measurements.json', c)
        write_pymol(c, folder)
        cases.append(c)
    return cases, {'parameters': parameters, 'module_path': distance_geometry.__file__,
                   'module_sha256': sha(distance_geometry.__file__)}


def write_pymol(c, folder):
    fixed = '+'.join(str(i + 1) for i, flag in enumerate(c['fixed_mask']) if flag)
    free = '+'.join(str(i + 1) for i, flag in enumerate(c['fixed_mask']) if not flag)
    # index, rather than SDF atom-map labels or ligand IDs, follows stored order.
    script = [
        '# Run from this directory: pymol view.pml', 'reinitialize',
        'bg_color white', 'load pocket.pdb, pocket', 'hide everything',
        'show lines, pocket', 'color gray75, pocket', 'set line_width, 1',
        'set_color held_color, [0.086, 0.55, 0.54]',
        'set_color free_color, [0.92, 0.60, 0.11]',
    ]
    for stage in STAGES:
        script += [f'load {stage}.sdf, {stage}', f'hide everything, {stage}',
                   f'show sticks, {stage}', f'color held_color, {stage} and index {fixed}',
                   f'color free_color, {stage} and index {free}',
                   f'color blue, {stage} and elem N', f'color red, {stage} and elem O',
                   f'color yellow, {stage} and elem S', f'color green, {stage} and elem F+Cl',
                   f'set stick_radius, 0.16, {stage}']
        for k, b in enumerate(c['boundary_bonds']):
            i, j = b['labels_1based']
            script += [f'distance boundary_{stage}_{k}, {stage} and index {i}, {stage} and index {j}',
                       f'set dash_color, magenta, boundary_{stage}_{k}']
    script += ['set label_color, black', 'set label_size, 16', 'zoom fm, 6',
               'disable uff', 'disable native', 'disable boundary_uff_*', 'disable boundary_native_*',
               'scene fm_generated, store', 'disable fm', 'disable boundary_fm_*',
               'enable native', 'enable boundary_native_*', 'scene native_reference, store',
               'disable native', 'disable boundary_native_*', 'enable uff', 'enable boundary_uff_*',
               'scene pocket_uff, store', 'scene fm_generated, recall',
               '# Switch with: scene native_reference, recall / scene pocket_uff, recall']
    (folder / 'view.pml').write_text('\n'.join(script) + '\n')


def write_tables(candidates, cases, output):
    columns = ['case_id', 'draw', 'atoms', 'vina_native', 'vina_fm', 'vina_uff',
               'vina_fm_minus_native', 'vina_uff_minus_fm', 'boundary_before_A',
               'boundary_after_A', 'boundary_pb_violation', 'fm_failed_checks']
    for filename, rows in [('candidate_pool.csv', candidates), ('selected_cases.csv', cases)]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    with (output / 'boundary_bonds.csv').open('w', newline='') as stream:
        keys = ['case_id', 'draw', 'atom_i', 'atom_j', 'stage', 'length_A',
                'native_length_A', 'lower_bound_A', 'upper_bound_A',
                'relative_bound_error', 'pb_bond_violation']
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for c in cases:
            for b in c['boundary_bonds']:
                for stage in STAGES:
                    writer.writerow({'case_id': c['case_id'], 'draw': c['draw'],
                                     'atom_i': b['labels_1based'][0], 'atom_j': b['labels_1based'][1],
                                     'stage': stage, 'length_A': b['lengths_A'][stage],
                                     'native_length_A': b['lengths_A']['native'], **b['pb_bond_details'][stage]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fm-run', type=Path, default=EXP / 'clamp_gpu1_20261021_v2/full')
    parser.add_argument('--refinement-run', type=Path, default=EXP / 'refine_20261022_v1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, choices=(3, 4, 5), default=4)
    parser.add_argument('--case', action='append', help='Explicit eligible CASE_ID:DRAW; repeat 3-5 times')
    parser.add_argument('--min-vina-gap', type=float, default=1.)
    parser.add_argument('--min-boundary-error', type=float, default=.1)
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    now = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    candidates, counts = collect_candidates(args.fm_run, args.refinement_run,
                                            args.min_vina_gap, args.min_boundary_error)
    selected = select_cases(candidates, args.samples, args.case)
    output.mkdir(parents=True, exist_ok=False)
    print(json.dumps({'created_at_kst': now, 'filters': counts,
                      'selected': [(c['case_id'], c['draw'], c['selection_reason']) for c in selected]}), flush=True)
    cases, geometry_config = assemble_cases(selected, args.fm_run, args.refinement_run, output)
    write_tables(candidates, cases, output)
    manifest = {'created_at_kst': now, 'fm_run': str(args.fm_run), 'refinement_run': str(args.refinement_run),
                'source_dataset': 'PDBbind2019 refined-set; original graph/core-conditioned experiments',
                'graph_variant': 'original', 'integration_steps': 64, 'uff_arm': 'uff_pkt',
                'selection_filters': {'native_pb_pass': True, 'fm_pb_fail': True, 'uff_pb_pass': True,
                                      'min_vina_fm_minus_native_kcal_mol': args.min_vina_gap,
                                      'min_boundary_deviation_from_native_A': args.min_boundary_error,
                                      'stereochemistry_preserved': True, 'max_held_drift_A': .001},
                'filter_counts': counts, 'geometry_diagnostic': geometry_config,
                'new_evaluation': 'Only fast PB distance_geometry bond localization; no full PB/Vina/UFF rerun',
                'selection_bias': 'Post-hoc illustrative distinct pockets, not a random or performance-estimation sample',
                'display': 'All stages share original pocket coordinates. No pose alignment or coordinate scaling.',
                'software': {'numpy': np.__version__, 'rdkit': Chem.rdBase.rdkitVersion},
                'fm_kernel': load(args.fm_run / 'kernel.json'),
                'cases': cases, 'script_sha256': sha(__file__),
                'renderer_sha256': sha(Path(__file__).with_name('boundary_visuals.py')),
                'viewer_template_sha256': sha(Path(__file__).with_name('boundary_viewer.html'))}
    dump(output / 'manifest.json', manifest)
    from boundary_visuals import make_figures, make_html, make_report
    make_figures(cases, output)
    make_html(cases, output)
    make_report(manifest, output)
    manifest['completed_at_kst'] = datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    dump(output / 'manifest.json', manifest)
    shutil.copyfile(__file__, output / 'visualize_boundary_refinement.py')
    shutil.copyfile(Path(__file__).with_name('boundary_visuals.py'), output / 'boundary_visuals.py')
    template = Path(__file__).with_name('boundary_viewer.html')
    shutil.copyfile(template, output / template.name)
    shutil.make_archive(str(output) + '-bundle', 'zip', root_dir=output.parent, base_dir=output.name)
    print(json.dumps({'status': 'complete', 'output': str(output), 'cases': len(cases),
                      'completed_at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')}), flush=True)


if __name__ == '__main__':
    main()
