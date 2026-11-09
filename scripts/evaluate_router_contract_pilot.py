"""CPU-only raw/refined/control evaluation of the frozen Router contract pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TZ'] = 'Asia/Seoul'
time.tzset()
from router_contract_pilot import ROOT, now, read, save, log
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from dirmo.chemistry.graphs import graph_check, read_sdf, stereo_from_coordinates, write_sdf
from dirmo.evaluation.edited_graph import graph_from_record
from dirmo.evaluation.pb_worker import bounded_measure
from dirmo.evaluation.fm_quality import vina_for_mol
from dirmo.evaluation.crossdocked_sbdd import prepare_rigid_receptor
from dirmo.io import sha256_file


def configure(root):
    os.environ['DIRMO_PB_MEASURE_CACHE'] = str(root / 'cache' / 'measurements')
    os.environ['DIRMO_PB_ENERGY_CACHE'] = str(root / 'cache' / 'energy')


def refine(source, pocket_path, held):
    """Fixed 6A pocket; keep exactly the same atoms for all arms/outputs."""
    pocket = Chem.MolFromPDBFile(str(pocket_path), sanitize=True, removeHs=True)
    if pocket is None:
        raise ValueError('Pocket sanitization failed; no silent fallback')
    ligand = Chem.AddHs(Chem.Mol(source), addCoords=True)
    complex_mol = Chem.CombineMols(ligand, pocket)
    Chem.GetSymmSSSR(complex_mol)
    if not AllChem.UFFHasAllMoleculeParams(complex_mol):
        raise ValueError('UFF parameters missing for ligand/pocket atoms')
    ff = AllChem.UFFGetMoleculeForceField(complex_mol, ignoreInterfragInteractions=False)
    if ff is None:
        raise ValueError('UFF force-field construction failed')
    for i in list(held) + list(range(ligand.GetNumAtoms(), complex_mol.GetNumAtoms())):
        ff.AddFixedPoint(int(i))
    ff.Initialize()
    before = float(ff.CalcEnergy())
    code = ff.Minimize(maxIts=500)
    after = float(ff.CalcEnergy())
    result = Chem.Mol(source)
    for i in range(source.GetNumAtoms()):
        result.GetConformer().SetAtomPosition(i, complex_mol.GetConformer().GetAtomPosition(i))
    return stereo_from_coordinates(result), dict(minimize_code=code, converged=(code == 0),
        energy_before=before, energy_after=after, pocket_atoms=pocket.GetNumAtoms(),
        energy_decreased=after <= before + 1e-6,
        all_uff_parameters=True, iterations_cap=500, pocket_hydrogens_added=False,
        interfragment_interactions=True)


def assess(path, graph, reference, free_maps, case):
    start = time.monotonic()
    mol = read_sdf(path)
    check = graph_check(graph, mol, derive_stereo=True)
    held = [i for i, a in enumerate(reference.GetAtoms()) if a.GetAtomMapNum() not in free_maps]
    equal_size = mol.GetNumAtoms() == reference.GetNumAtoms()
    drift = (float(np.linalg.norm(mol.GetConformer().GetPositions()[held] -
        reference.GetConformer().GetPositions()[held], axis=1).max()) if held else 0.) if equal_size else None
    # Upstream reference tensorization verifies index identity before any FM call.
    # Do not best-align output to native to hide an actual held-coordinate error.
    pb = bounded_measure(str(path), case['pocket_path'], diagnose=True)
    try:
        vina = vina_for_mol(mol, case)
    except Exception as exc:
        vina = dict(status='missing', reason=f'{type(exc).__name__}: {exc}')
    boundary = []
    if equal_size:
        free = {i for i, a in enumerate(reference.GetAtoms()) if a.GetAtomMapNum() in free_maps}
        xyz = mol.GetConformer().GetPositions()
        for bond in reference.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a in free) != (b in free):
                boundary.append(dict(indices=[a, b], atom_maps=[reference.GetAtomWithIdx(i).GetAtomMapNum() for i in (a, b)],
                    length_angstrom=float(np.linalg.norm(xyz[a] - xyz[b]))))
    missing = bool(pb.get('missing', False) or pb.get('missing_checks'))
    pb['measurement_missing'] = missing
    held_ok = drift is not None and drift <= .001
    physical = None if missing else bool(pb.get('passes') and check['matches'] and held_ok)
    return dict(sdf=str(path), sha256=sha256_file(path), graph_check=check, held_max_drift=drift,
        held_preserved=held_ok, pb=pb, vina=vina, joint_physical=physical,
        boundary_bonds=boundary, measurement_seconds=time.monotonic() - start)


def evaluate_case(root, cid):
    configure(root)
    directory = root / 'evaluations' / cid
    if (directory / 'complete.json').exists():
        return
    case = read(root / 'cases' / cid / 'case.json')
    initial = read_sdf(case['initial_sdf'])
    try:
        receptor, _ = prepare_rigid_receptor(case['pocket_path'], directory)
    except Exception as exc:
        receptor = None
        save(directory / 'receptor_error.json', dict(reason=f'{type(exc).__name__}: {exc}'))
    case['receptor_path'] = receptor
    native_path = directory / 'initial_metrics.json'
    if not native_path.exists():
        save(native_path, assess(case['initial_sdf'], initial, initial, set(), case))
    native = read(native_path)
    parent_vina = native['vina'].get('score')
    rows = []
    for ai, action in enumerate(case['actions']):
        graph = graph_from_record(action['graph'])
        for mode in ('strict', 'relaxed'):
            free_maps = set(action[f'{mode}_free_maps'])
            for prior in action['priors']:
                dest = directory / action['graph']['name'] / mode / f"{prior['draw']:02d}"
                marker = dest / 'outcome.json'
                if marker.exists():
                    rows.append(read(marker))
                    continue
                result = read(root / 'draws' / cid / action['graph']['name'] / mode / f"{prior['draw']:02d}" / 'result.json')
                row = dict(case_id=cid, split=case.get('split', 'train'), target_id=case['target_id'],
                    action=action['graph']['name'], action_kind=action['action_kind'],
                    mode=mode, seed=prior['seed'], draw=prior['draw'],
                    generation_status=result['status'], generation=result,
                    exact_product_properties=action['graph']['properties'],
                    parent_vina=parent_vina, raw=None, refined=None, uff_only=None, time_kst=now())
                if prior['status'] == 'ready':
                    reference = read_sdf(prior['sdf'])
                    held = [i for i, a in enumerate(reference.GetAtoms()) if a.GetAtomMapNum() not in free_maps]
                    raw = result.get('sdf_path')
                    if raw:
                        try:
                            row['raw'] = assess(raw, graph, reference, free_maps, case)
                        except Exception as exc:
                            row['raw_error'] = f'{type(exc).__name__}: {exc}'
                        # Preserve raw failure and refine even a stereo-invalid raw pose;
                        # refined graph/stereo are checked against the target independently.
                        try:
                            start = time.monotonic()
                            refined, detail = refine(read_sdf(raw), case['pocket_path'], held)
                            detail['seconds'] = time.monotonic() - start
                            path = dest / 'refined.sdf'
                            write_sdf(path, refined)
                            row['refined'] = assess(path, graph, reference, free_maps, case)
                            row['refined']['uff'] = detail
                        except Exception as exc:
                            row['refinement_error'] = f'{type(exc).__name__}: {exc}'
                    if case['uff_only_control'] and (case.get('uff_only_all_actions', False) or ai == 0 or action['action_kind'] == 'POSE_RESAMPLE'):
                        try:
                            start = time.monotonic()
                            refined, detail = refine(reference, case['pocket_path'], held)
                            detail['seconds'] = time.monotonic() - start
                            path = dest / 'uff_only.sdf'
                            write_sdf(path, refined)
                            row['uff_only'] = assess(path, graph, reference, free_maps, case)
                            row['uff_only']['uff'] = detail
                        except Exception as exc:
                            row['uff_only_error'] = f'{type(exc).__name__}: {exc}'
                for stage in ('raw', 'refined', 'uff_only'):
                    metric = row[stage]
                    if metric is not None:
                        value = metric['vina'].get('score')
                        metric['vina_improvement_vs_initial'] = parent_vina - value if parent_vina is not None and value is not None else None
                save(marker, row)
                rows.append(row)
                save(directory / 'status.json', dict(status='running', time_kst=now(), completed_rows=len(rows)))
    save(directory / 'rows.json', rows)
    save(directory / 'complete.json', dict(time_kst=now(), rows=len(rows)))


def report(root):
    from collections import Counter
    plan = read(root / 'study.json')
    native, rows = [], []
    completed = []
    for cid in plan['cases']:
        directory = root / 'evaluations' / cid
        if (directory / 'initial_metrics.json').exists():
            native.append(read(directory / 'initial_metrics.json'))
        if (directory / 'complete.json').exists():
            completed.append(cid)
            rows.extend(read(directory / 'rows.json'))
    summary = dict(time_kst=now(), completed_cases=len(completed), planned_cases=len(plan['cases']),
        completed_rows=len(rows), planned_fm_draws=plan['planned_fm_draws'],
        generation_status=dict(Counter(r['generation_status'] for r in rows)),
        initial_pb_measured=sum(not r['pb']['measurement_missing'] for r in native),
        initial_pb_pass=sum(not r['pb']['measurement_missing'] and r['pb'].get('passes', False) for r in native))
    for mode in ('strict', 'relaxed'):
        for stage in ('raw', 'refined', 'uff_only'):
            values = [r[stage] for r in rows if r['mode'] == mode and r[stage] is not None]
            gains = [v['vina_improvement_vs_initial'] for v in values
                     if v['joint_physical'] is True and v['vina_improvement_vs_initial'] is not None]
            summary[f'{mode}_{stage}'] = dict(available=len(values), pb_measured=sum(not v['pb']['measurement_missing'] for v in values),
                pb_pass=sum(not v['pb']['measurement_missing'] and v['pb'].get('passes', False) for v in values),
                joint_physical_pass=sum(v['joint_physical'] is True for v in values),
                graph_preserved=sum(v['graph_check']['matches'] for v in values),
                feasible_vina_pairs=len(gains), feasible_vina_improvement_mean=float(np.mean(gains)) if gains else None)
    save(root / 'summary.json', summary)
    return summary


def watch(args):
    import fcntl
    root = args.output
    lock = (root / 'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(root / 'study.json')
    submitted, futures, errors = set(), {}, {}
    def run(cid):
        directory = root / 'evaluations' / cid
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'evaluation.log').open('a') as stream:
            return subprocess.run([sys.executable, str(Path(__file__).resolve()), 'case',
                '--output', str(root), '--case-id', cid], stdout=stream, stderr=subprocess.STDOUT,
                timeout=21600).returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while True:
            generation = read(root / 'generation_status.json') if (root / 'generation_status.json').exists() else {}
            for cid in plan['cases']:
                if cid in submitted:
                    continue
                if (root / 'cases' / cid / 'generation_done.json').exists():
                    submitted.add(cid)
                    futures[pool.submit(run, cid)] = cid
            for future, cid in list(futures.items()):
                if future.done():
                    try:
                        code = future.result()
                        if code:
                            errors[cid] = f'Evaluation child exited {code}'
                    except Exception as exc:
                        errors[cid] = f'{type(exc).__name__}: {exc}'
                    del futures[future]
                    log('evaluation_case_done', case_id=cid, error=errors.get(cid))
            summary = report(root)
            terminal = generation.get('status') in ('complete', 'failed') and not futures
            save(root / 'evaluation_status.json', dict(time_kst=now(), pid=os.getpid(),
                status=('complete' if len(submitted) == len(plan['cases']) and not errors else 'incomplete') if terminal else 'running',
                submitted_cases=len(submitted), completed_cases=summary['completed_cases'],
                pending_cases=len(futures), errors=errors))
            if terminal:
                break
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['watch', 'case', 'report'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-id')
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == 'case':
        evaluate_case(args.output, args.case_id)
    elif args.command == 'watch':
        watch(args)
    else:
        print(report(args.output))


if __name__ == '__main__':
    main()
