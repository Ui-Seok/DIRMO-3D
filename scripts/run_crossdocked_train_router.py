"""Resumeable full CrossDocked-train outcome collection, then CUDA Router fitting.

The full processed train index is authoritative. Existing test-derived outcomes
are never read as labels. Resident FM/LLM services are reused without lifecycle
changes. SQLite commits a case's rows and completion marker together.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
from dirmo.io import append_jsonl, digest, environment, revision, sha256_file, write_json
from dirmo.schemas import ActionKind, Case


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def log(event, **fields):
    print(json.dumps({'time_kst': now(), 'event': event, **fields}, ensure_ascii=False), flush=True)


def names(root, split):
    with np.load(root / f'{split}.npz') as archive:
        return archive['names'].tolist()


def source_pair(name):
    protein, ligand = name.split('.pdb_', 1)
    return protein + '.pdb', ligand


def pdb_ids(name):
    protein, ligand = source_pair(name)
    return {Path(protein).name.split('_')[0].lower(), Path(ligand).name.split('_')[3].lower()}


def prepare(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'study.json').exists():
        study = json.loads((root / 'study.json').read_text())
        if study['collector_sha256'] != sha256_file(__file__):
            raise ValueError('Collector source changed; use a new output directory')
        return study
    train = names(args.processed, 'train')
    test = names(args.processed, 'test')
    validation = names(args.processed, 'val')
    forbidden = {identifier for n in test for identifier in pdb_ids(n)}
    if set(train) & set(test):
        raise ValueError('Train/test source-pair overlap; refusing to collect')
    if any(pdb_ids(n) & forbidden for n in train):
        raise ValueError('Train/test receptor or ligand source PDB overlap; audit before collection')
    missing = []
    with (root / 'cases.jsonl').open('w') as handle:
        for index, name in enumerate(train):
            pocket, ligand = source_pair(name)
            source_pocket, source_ligand = args.raw / pocket, args.raw / ligand
            absent = [str(p) for p in [source_pocket, source_ligand] if not p.is_file()]
            if absent:
                missing.append({'source_index': index, 'source_name': name, 'missing': absent})
            case = Case(case_id=f'cdtrain-{index:06d}-{digest(name)[:12]}',
                        target_id=Path(pocket).parent.name,
                        series_id='-'.join(Path(ligand).name.split('_')[3:5]),
                        pocket_path=str(source_pocket), initial_ligand_path=str(source_ligand),
                        preparation_notes='CrossDocked train source; fixed heavy coordinates; no ligand relaxation. '
                        'Scoring receptor uses Open Babel add-H/Gasteiger, with heavy coordinates checked.')
            case.budgets.candidate_cap = 2
            case.budgets.fm_draws = 12
            case.budgets.fm_nfe = 4096
            case.budgets.evaluation_calls = 16
            case.budgets.llm_calls = 2
            case.budgets.llm_tokens = 32768
            case.budgets.seeds_per_graph = 3
            handle.write(json.dumps({'source_index': index, 'source_name': name,
                                     'source_split': 'train', 'case': case.model_dump(mode='json')}) + '\n')
    if missing:
        write_json(root / 'missing_sources.json', missing)
        raise ValueError(f'{len(missing)} train entries have missing raw files; full collection not started')
    study = {'created_at_kst': now(), 'source_split': 'train', 'source_entries': len(train),
             'source_train_names_sha256': digest(train), 'source_test_names_sha256': digest(test),
             'source_test_pairs_overlap': 0, 'source_test_pdb_overlap': 0,
             'existing_val_entries_also_in_train': len(set(train) & set(validation)),
             'validation': 'Grouped folds inside train; existing val is not an independent holdout.',
             'manifest_sha256': sha256_file(root / 'cases.jsonl'),
             'collector_sha256': sha256_file(__file__),
             'processed': str(args.processed), 'raw': str(args.raw),
             'seeds': [42, 43, 44], 'steps': 64,
             'actions': ['POSE_RESAMPLE', 'CHEMICAL_EDIT'],
             'candidate_menu': 'Up to two valid LLM edits; fill missing slots with enumerated QED-ranked edits.',
             'planner_pose_gate': 'Skip LLM on parent geometry/PB failure; enumeration remains available.',
             'source_warning': 'CrossDocked train only. Internal grouped validation; no official test evaluation.',
             'final_training_gpu': 4, 'minimum_disk_free_gib': args.minimum_disk_gib,
             'completion': 'Every source receives a recorded terminal outcome; preparation/capability failures are reported, never fabricated labels.'}
    write_json(root / 'study.json', study)
    log('manifest_complete', source_entries=len(train), train_test_overlap=0,
        existing_val_overlap=study['existing_val_entries_also_in_train'])
    return study


def database(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, status TEXT, detail TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS outcomes (case_id TEXT, action_id TEXT, seed INTEGER, '
               'features BLOB, utility REAL, seconds REAL, feasible INTEGER, metadata TEXT, '
               'PRIMARY KEY(case_id, action_id, seed))')
    db.commit()
    return db


def freeze_run_identity(config, health):
    """Cache experiment-wide fingerprints, not per-case input checksums."""
    import dirmo.pipeline.run as module
    source = Path(module.__file__).resolve().parents[1]
    static = {'implementation_sha256': digest({str(p.relative_to(source)): sha256_file(p)
                                               for p in sorted(source.rglob('*.py'))}),
              'config': config.model_dump(mode='json'), 'environment': environment(),
              'checkpoint_sha256': health['model_identity']['sha256'],
              'flowr_revision': revision(config.fm.repo_path),
              'llm_revision': revision(config.llm.model_path)}
    def identity(case, unused_config):
        inputs = {key: sha256_file(value) for key in ['initial_ligand_path', 'pocket_path',
                  'protein_path', 'receptor_path'] if (value := getattr(case, key))}
        return {**static, 'case': case.model_dump(mode='json'), 'input_sha256': inputs}
    module.input_identity = identity


def collect_case(entry, config, active, directory):
    from dirmo.chemistry.graphs import ranked_edits, read_sdf
    from dirmo.evaluation.crossdocked_sbdd import prepare_rigid_receptor
    from dirmo.evaluation.pocket import protein_atoms
    from dirmo.llm.proposals import Planner
    from dirmo.pipeline.run import Executor, Run
    from dirmo.routing.features import FEATURE_NAMES, features
    from dirmo.routing.policies import available_actions
    from dirmo.routing.spatial import region_spatial_features
    from dirmo.chemistry.graphs import execute_edit, graph_id
    case = Case.model_validate(entry['case'])
    directory.mkdir(parents=True, exist_ok=True)
    def progress(stage, **fields):
        write_json(directory / 'progress.json', {'updated_at_kst': now(), 'stage': stage, **fields})
    progress('preparing_receptor')
    if (directory / 'checkpoint.json').exists():
        raise RuntimeError('Uncommitted previous case execution requires inspection; no automatic redraw')
    case.receptor_path, _ = prepare_rigid_receptor(Path(case.pocket_path), directory / 'scoring')
    parent = read_sdf(case.initial_ligand_path)
    if parent.GetNumAtoms() > config.fm.max_atoms:
        raise ValueError(f'Ligand atom count {parent.GetNumAtoms()} exceeds FM limit {config.fm.max_atoms}')
    run = Run(case, config, directory)
    progress('initial_evaluated')
    parent = read_sdf(run.state.work_item.sdf_path)
    protein = protein_atoms(run.fixed_pocket)
    if len(protein[0]) > config.fm.max_pocket_atoms:
        raise ValueError(f'Pocket atom count {len(protein[0])} exceeds FM limit {config.fm.max_pocket_atoms}')
    failure = set(run.state.work_item.observation.failures)
    if not (failure & {'geometry:invalid', 'posebusters:invalid'}):
        progress('llm_proposal')
        planner = Planner(active['llm'], config.llm, directory, run.budget,
                          persist=lambda event: run.save(pending=event))
        run.state.proposal = planner.propose(run.state, run.case)
        run.save()
    else:
        write_json(directory / 'planner_skipped.json', {'time_kst': now(), 'reason': 'parent_pose_invalid'})
    existing = list(run.state.proposal.edit_candidates) if run.state.proposal else []
    graphs = {graph_id(execute_edit(parent, c, parent, case.protected)[0]) for c in existing}
    extra = []
    if len(existing) < 2:
        for candidate in ranked_edits(parent, parent, case.protected, limit=6,
                                      max_sa_loss=case.objective.metrics['sa'].max_loss):
            gid = graph_id(execute_edit(parent, candidate, parent, case.protected)[0])
            if gid not in graphs:
                extra.append(candidate); graphs.add(gid)
            if len(existing) + len(extra) >= 2:
                break
    run.refresh()
    snapshot = run.state.model_copy(deep=True)
    write_json(directory / 'branch_snapshot.json', snapshot)
    parent_metrics = snapshot.work_item.observation.metrics
    parent_maps = {a.GetAtomMapNum() for a in parent.GetAtoms()}
    rows = []
    for seed in [42, 43, 44]:
        executor = Executor(run, active['fm'], seed_base=seed, extra_candidates=extra)
        actions = [a for a in available_actions(snapshot, case, config.fm.parameters,
                   executor.capabilities, executor.nfe_factor, seed, extra)
                   if a.kind in [ActionKind.POSE_RESAMPLE, ActionKind.CHEMICAL_EDIT]]
        for action in actions:
            run.state = snapshot.model_copy(deep=True)
            region = (set(action.candidate.changed_atom_maps) & parent_maps) if action.candidate else parent_maps
            if not region:
                region = parent_maps
            values = features(snapshot, action, case) + region_spatial_features(
                parent, *protein, region, case.protected.atom_maps)
            start = time.monotonic()
            progress('fm_action', action=action.action_id, seed=seed, recorded_outcomes=len(rows))
            outcomes = executor.execute(action)
            for result in outcomes:
                observation = result['resulting_work_item']['observation']
                after = observation.get('utility')
                before = snapshot.work_item.observation.utility
                delta = after - before if after is not None and before is not None else None
                child = observation.get('metrics', {})
                metadata = {'state_id': snapshot.state_id, 'target_id': case.target_id,
                            'series_id': case.series_id, 'parent_graph_id': snapshot.work_item.graph_id,
                            'source_split': 'train', 'source_name': entry['source_name'],
                            'source_index': entry['source_index'], 'case_id': case.case_id,
                            'action_id': action.action_id, 'action_kind': action.kind.value,
                            'region': sorted(region), 'seed': seed,
                            'feasible': result['feasible'], 'parent_feasible': snapshot.work_item.observation.feasible,
                            'utility_delta': delta, 'wall_seconds': time.monotonic() - start,
                            'parent_metrics': parent_metrics, 'child_metrics': child,
                            'metric_deltas': {k: child[k] - parent_metrics[k] for k in ['vina', 'qed', 'sa']
                                              if k in child and k in parent_metrics},
                            'generation_status': result['generation_result']['status'],
                            'failure_reason': result['generation_result'].get('reason'),
                            'generation_protocol_id': executor.generation_kernel_id(action.generation),
                            'objective_sha256': digest(case.objective)}
                rows.append((case.case_id, action.action_id, seed,
                             np.asarray(values, dtype=np.float32).tobytes(), delta,
                             metadata['wall_seconds'], int(result['feasible']), json.dumps(metadata)))
    run.state = snapshot
    run.save()
    progress('case_collected', recorded_outcomes=len(rows))
    return rows


def export_prepared(db, output, study):
    from dirmo.routing.features import FEATURE_NAMES
    from dirmo.routing.spatial import SPATIAL_FEATURE_NAMES
    output.mkdir(parents=True, exist_ok=False)
    metadata, xs, ys = [], [], []
    protocols, objectives = set(), set()
    for i, (blob, utility, seconds, feasible, raw) in enumerate(db.execute(
            'SELECT features,utility,seconds,feasible,metadata FROM outcomes ORDER BY case_id,action_id,seed')):
        item = json.loads(raw); item['row_id'] = i
        if item['source_split'] != 'train':
            raise ValueError('Non-train outcome in training database')
        metadata.append(item)
        xs.append(np.frombuffer(blob, dtype=np.float32).copy())
        ys.append([np.nan if utility is None else utility, seconds, feasible])
        protocols.add(item['generation_protocol_id']); objectives.add(item['objective_sha256'])
    if len(protocols) != 1 or len(objectives) != 1:
        raise ValueError('Mixed generation/evaluation protocols; refusing to pool labels')
    parents = list(range(len(metadata))); seen = {}
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    for i, row in enumerate(metadata):
        for name in ['state_id', 'target_id', 'series_id', 'parent_graph_id']:
            key = (name, row[name])
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = {}
    for i, row in enumerate(metadata):
        root = find(i)
        if root not in groups:
            groups[root] = f'group_{len(groups):06d}'
        row['group_id'] = groups[root]
    x, y = np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite features')
    np.savez_compressed(output / 'arrays.npz', x=x, y=y,
                        groups=np.array([r['group_id'] for r in metadata]))
    write_json(output / 'rows.json', metadata)
    audit = {'created_at_kst': now(), 'baseline_features': FEATURE_NAMES, 'spatial_features': SPATIAL_FEATURE_NAMES,
             'generation_protocol_id': protocols.pop(), 'objective_sha256': objectives.pop(),
             'states': len({r['state_id'] for r in metadata}), 'groups': len(groups), 'rows': len(metadata),
             'known_utility_rows': int(np.isfinite(y[:, 0]).sum()), 'feasible_rows': int(y[:, 2].sum()),
             'source_warning': study['source_warning'], 'source_split': 'train',
             'benchmark_test_derived_rows': 0, 'all_source_entries': study['source_entries'],
             'scope': 'full_crossdocked_train',
             'source_train_names_sha256': study['source_train_names_sha256'],
             'supported_action_kinds': sorted({r['action_kind'] for r in metadata})}
    write_json(output / 'audit.json', audit)


def run(args):
    from dirmo.config import load_config
    from dirmo.services import clients
    study = prepare(args)
    if args.phase == 'prepare':
        return
    config = load_config(args.config)
    if config.fm.inpaint_mode != 'graph' or config.fm.parameters.integration_steps != 64:
        raise ValueError('This collection protocol requires graph mode and 64 integration steps')
    active = clients(config)
    health = {role: client.health() for role, client in active.items()}
    freeze_run_identity(config, health['fm'])
    write_json(args.output / 'worker_identity.json', health)
    db = database(args.output / 'outcomes.sqlite')
    finished = {r[0] for r in db.execute('SELECT case_id FROM cases')}
    counts = dict(db.execute('SELECT status,COUNT(*) FROM cases GROUP BY status'))
    row_count = db.execute('SELECT COUNT(*) FROM outcomes').fetchone()[0]
    start = time.monotonic(); initially_finished = len(finished); consecutive = 0
    status = {'status': 'collecting', 'started_or_resumed_at_kst': now(), 'source_entries': study['source_entries'],
              'completed_cases': len(finished), 'outcome_rows': row_count, 'source_split': 'train',
              'benchmark_test_derived_rows': 0, 'gpu_fm': 2, 'gpu_llm': 3, 'gpu_router_training': 4,
              'training_status': 'queued_after_full_collection', 'pid': os.getpid()}
    try:
        for line in (args.output / 'cases.jsonl').open():
            entry = json.loads(line); case_id = entry['case']['case_id']
            if case_id in finished:
                continue
            if shutil.disk_usage(args.output).free < args.minimum_disk_gib * 2**30:
                raise RuntimeError('Disk reserve reached; preserving collected data and stopping safely')
            status.update(current_case=case_id, current_source_index=entry['source_index'], updated_at_kst=now())
            write_json(args.output / 'status.json', status)
            directory = args.output / 'cases' / f"{entry['source_index'] // 1000:03d}" / case_id
            status['current_case_progress'] = str(directory / 'progress.json')
            write_json(args.output / 'status.json', status)
            try:
                rows = collect_case(entry, config, active, directory)
                case_status, detail = ('collected', f'{len(rows)} outcomes') if rows else ('no_actions', 'No executable action')
                consecutive = 0 if rows else consecutive + 1
            except Exception as exc:
                if (directory / 'checkpoint.json').exists():
                    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
                    if checkpoint.get('pending_action'):
                        raise RuntimeError(f'Uncertain in-flight case {case_id}; inspect checkpoint before continuing') from exc
                rows = []; case_status = 'preparation_failed'; detail = f'{type(exc).__name__}: {exc}'
                consecutive += 1
            with db:
                db.executemany('INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?)', rows)
                db.execute('INSERT INTO cases VALUES (?,?,?)', (case_id, case_status, detail))
            finished.add(case_id); counts[case_status] = counts.get(case_status, 0) + 1; row_count += len(rows)
            append_jsonl(args.output / 'case_events.jsonl', {'time_kst': now(), 'case_id': case_id,
                         'status': case_status, 'detail': detail, 'source_index': entry['source_index']})
            completed_now = len(finished) - initially_finished
            seconds_per_case = (time.monotonic() - start) / max(1, completed_now)
            status.update(completed_cases=len(finished), outcome_rows=row_count, case_status_counts=counts,
                          updated_at_kst=now(), elapsed_seconds=time.monotonic()-start,
                          measured_seconds_per_case=seconds_per_case,
                          estimated_remaining_seconds=seconds_per_case*(study['source_entries']-len(finished)))
            write_json(args.output / 'status.json', status)
            log('case_complete', **{k: status[k] for k in ['completed_cases', 'outcome_rows', 'current_case']},
                case_status=case_status, detail=detail)
            if consecutive >= 5:
                raise RuntimeError('Five consecutive cases without outcomes; inspect recorded failures')
            if completed_now % 100 == 0:
                current = {role: client.health()['model_identity'] for role, client in active.items()}
                if any(digest(current[r]) != digest(health[r]['model_identity']) for r in current):
                    raise RuntimeError('Resident model identity changed during collection')
        if len(finished) != study['source_entries']:
            raise AssertionError('Full train manifest was not traversed')
        status.update(status='preparing_training', updated_at_kst=now())
        write_json(args.output / 'status.json', status)
        prepared = args.output / 'prepared'
        if prepared.exists():
            previous = json.loads((prepared / 'audit.json').read_text())
            if previous['rows'] != row_count or previous['source_train_names_sha256'] != study['source_train_names_sha256']:
                raise ValueError('Existing prepared data do not match the completed collection')
        else:
            export_prepared(db, prepared, study)
        status.update(status='training', training_status='running_on_gpu4', updated_at_kst=now())
        write_json(args.output / 'status.json', status)
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = 'GPU-a9f4e995-14aa-3b83-ffaf-0f5e0172f422'
        training_output = args.output / 'training'
        if not (training_output / 'summary.json').exists():
            if training_output.exists():
                number = 1
                while (args.output / f'training_attempt_{number}').exists():
                    number += 1
                training_output = args.output / f'training_attempt_{number}'
            status.update(training_output=str(training_output))
            write_json(args.output / 'status.json', status)
            command = [sys.executable, str(Path(__file__).with_name('train_router_ablation.py')),
                       '--prepared', str(prepared), '--output', str(training_output), '--physical-gpu', '4']
            with (args.output / f'{training_output.name}.log').open('w') as stream:
                subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        status.update(status='complete', training_status='complete', finished_at_kst=now(), updated_at_kst=now())
        write_json(args.output / 'status.json', status)
    except Exception as exc:
        status.update(status='failed', updated_at_kst=now(), reason=f'{type(exc).__name__}: {exc}')
        write_json(args.output / 'status.json', status)
        raise
    finally:
        db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--processed', type=Path, default=Path('/LocalData1/uslee/datasets/processed_crossdock_noH_full_temp'))
    parser.add_argument('--raw', type=Path, default=Path('/LocalData1/uslee/datasets/crossdocked_pocket10'))
    parser.add_argument('--config', type=Path, default=Path('/LocalData1/uslee/experiments/DIRMO-3D/worker-runtime/config.json'))
    parser.add_argument('--phase', choices=['prepare', 'run'], default='run')
    parser.add_argument('--minimum-disk-gib', type=int, default=15)
    args = parser.parse_args()
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.collection.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)
