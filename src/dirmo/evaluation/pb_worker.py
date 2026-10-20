"""Isolate slow PB conformer calculations without changing the PB checks."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from ..io import digest, sha256_file, write_json


def bounded_measure(path, pocket, *, diagnose=False):
    from . import pb_guided
    budget = int(os.environ.get('DIRMO_PB_MEASUREMENT_TIMEOUT', '600'))
    cache = Path(os.environ.get('DIRMO_PB_MEASURE_CACHE', '/tmp/dirmo_pb_measurements'))
    identity = {'sdf_sha256': sha256_file(path), 'pocket_sha256': sha256_file(pocket),
                'diagnose': diagnose, 'evaluator_sha256': sha256_file(pb_guided.__file__)}
    cache.mkdir(parents=True, exist_ok=True)
    key = digest(identity)
    target = cache / f'{key}.json'
    with (cache / f'{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            saved = json.loads(target.read_text())
            if not saved.get('measurement_timeout') or saved.get('timeout_seconds', 0) >= budget:
                return {**saved, 'sdf': str(path)}
        pending = cache / f'{key}.{os.getpid()}.pending.json'
        command = [sys.executable, '-m', 'dirmo.evaluation.pb_worker', '--sdf', str(path),
                   '--pocket', str(pocket), '--output', str(pending)]
        if diagnose:
            command.append('--diagnose')
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1',
                       'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parents[2]) + os.pathsep + environment.get('PYTHONPATH', '')
        log = cache / f'{key}.log'
        try:
            with log.open('a') as stream:
                process = subprocess.run(command, env=environment, stdout=stream,
                                         stderr=subprocess.STDOUT, timeout=budget)
            if process.returncode or not pending.exists():
                raise RuntimeError(f'PB child exited with status {process.returncode}; see {log}')
            result = json.loads(pending.read_text())
        except subprocess.TimeoutExpired:
            # An unavailable evaluation is neither a measured physical failure nor
            # a pass. It remains explicit and stays in the scheduled denominator.
            result = {'passes': False, 'missing': True, 'checks': {}, 'measurement_timeout': True,
                      'timeout_seconds': budget, 'sdf': str(path),
                      'error': f'PB measurement exceeded {budget} seconds; conformer count and thresholds were unchanged',
                      'log': str(log)}
        except Exception as exc:
            result = {'passes': False, 'missing': True, 'checks': {}, 'sdf': str(path),
                      'error': f'{type(exc).__name__}: {exc}', 'log': str(log)}
        finally:
            pending.unlink(missing_ok=True)
        write_json(target, result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdf', required=True)
    parser.add_argument('--pocket', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--diagnose', action='store_true')
    args = parser.parse_args()
    from .pb_guided import measure
    write_json(args.output, measure(args.sdf, args.pocket, diagnose=args.diagnose))


if __name__ == '__main__':
    main()
