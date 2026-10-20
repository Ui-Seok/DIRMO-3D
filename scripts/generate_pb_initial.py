"""Prepare a fixed pocket cohort and generate raw de novo starting ligands."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from dirmo.gpu import inventory, resolve_device, worker_environment
from dirmo.io import append_jsonl, sha256_file, write_json


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def prepare(args):
    if (args.output / 'study.json').exists():
        return read(args.output / 'study.json')
    source = read(args.source)
    cases = list(source['cases'])
    np.random.default_rng(args.seed).shuffle(cases)
    selected, proteins = [], set()
    for case in cases:
        if case['target_id'] in proteins:
            continue
        proteins.add(case['target_id'])
        selected.append({key: case[key] for key in ('case_id', 'target_id', 'scaffold', 'native_sdf',
                         'pocket_path', 'protein_path', 'atoms', 'pocket_atoms', 'source_dataset')})
        if len(selected) == args.cases:
            break
    if len(selected) != args.cases:
        raise ValueError('Insufficient distinct exact proteins in the supplied pool')
    plan = {'cases': selected, 'cases_requested': args.cases, 'initial_draws': args.draws,
            'repair_draws': 8, 'steps': args.steps, 'seed': args.seed, 'physical_gpu': args.gpu,
            'checkpoint': 'base', 'source_study': str(args.source), 'source_sha256': sha256_file(args.source),
            'selection': 'Seed-shuffled existing eligible complex pool; one complex per exact protein sequence; no PB outcome selection',
            'initial_generation': 'Pocket-conditioned de novo, graph/coordinates generated; reference atom count retained',
            'repair': 'Pose-only; original generated graph/stereo and held XYZ preserved; free initial coordinate noise sigma=0.3 A',
            'arms': ['targeted', 'random', 'whole', 'UFF_targeted'], 'rounds': 1,
            'primary': 'Pocket PoseBusters dock all checks + generated graph/stereo + held drift <=0.001 A',
            'localization': 'Exact configured PoseBusters module details; global-only failures explicitly ineligible',
            'llm_calls': 0, 'training': False, 'pretraining_overlap': 'unknown',
            'sampling_budget': '8 FM attempts per eligible failed initial ligand per arm; UFF is a separate CPU control'}
    write_json(args.output / 'study.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', type=Path, default=ROOT / 'runs/fm_pose_gpu4_2000_batch32_v1/steps_64/study.json')
    parser.add_argument('--gpu', default='4')
    parser.add_argument('--cases', type=int, default=100)
    parser.add_argument('--draws', type=int, default=16)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = prepare(args)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared', 'cases': len(plan['cases'])}))
        return
    with (args.output / 'initial.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = resolve_device(args.gpu, inventory())
        os.environ.update(worker_environment(device, str(ROOT / 'src')))
        from dirmo.config import FMConfig
        from dirmo.evaluation.fm_quality import CHECKPOINTS
        from dirmo.generation.de_novo import DeNovoBackend
        filename, checksum = CHECKPOINTS['base']
        config = FMConfig(gpu=args.gpu, port=8876, python=sys.executable, repo_path='/home/uslee/code/flowr_root',
                          checkpoint_path=str(Path('/LocalData1/uslee/checkpoints/FLOWR_root') / filename),
                          checkpoint_sha256=checksum)
        state = {'status': 'loading', 'pid': os.getpid(), 'gpu': device}
        write_json(args.output / 'initial_worker.json', state)
        backend = DeNovoBackend(config)
        for case in plan['cases']:
            directory = args.output / 'cases' / case['case_id'] / 'initial'
            pending = [i for i in range(plan['initial_draws']) if not (directory / f'{i:03d}' / 'result.json').exists()]
            capacity = min(16, len(pending))
            while pending:
                indices = pending[:capacity]
                try:
                    stats = backend.generate(case, indices, directory, plan['steps'],
                                             seed=plan['seed'] + int(case['case_id'], 36) % 1000000 + indices[0])
                    append_jsonl(args.output / 'initial_batches.jsonl', {'case_id': case['case_id'], **stats})
                    pending = pending[len(indices):]
                    state.update(status='running', current_case=case['case_id'], remaining_in_case=len(pending))
                    write_json(args.output / 'initial_worker.json', state)
                    print(json.dumps({'status': 'initial_batch_complete', 'case_id': case['case_id'], **stats}), flush=True)
                except backend.torch.OutOfMemoryError:
                    backend.torch.cuda.empty_cache()
                    append_jsonl(args.output / 'initial_batches.jsonl', {'status': 'oom', 'case_id': case['case_id'], 'batch_size': capacity})
                    if capacity > 1:
                        capacity = max(1, capacity // 2)
                        continue
                    write_json(directory / f'{indices[0]:03d}' / 'result.json', {'status': 'oom'})
                    pending = pending[1:]
                except Exception as exc:
                    traceback.print_exc()
                    for i in indices:
                        write_json(directory / f'{i:03d}' / 'result.json', {'status': 'failed', 'reason': f'{type(exc).__name__}: {exc}'})
                    pending = pending[len(indices):]
                backend.torch.cuda.empty_cache()
            write_json(directory / 'complete.json', {'case_id': case['case_id'], 'draws': plan['initial_draws']})
        state.update(status='complete')
        write_json(args.output / 'initial_worker.json', state)


if __name__ == '__main__':
    main()
