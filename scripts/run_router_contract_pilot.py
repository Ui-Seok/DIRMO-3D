"""Supervise only this pilot's FM process and CPU evaluator; no server lifecycle changes."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from router_contract_pilot import ROOT, now, read, save
from dirmo.io import sha256_file, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    root = args.output.resolve()
    lock = (root / 'supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read(root / 'study.json')
    if sha256_file(ROOT / 'scripts/router_contract_pilot.py') != plan['collector_sha256']:
        raise ValueError('Collector changed since preparation')
    code = root / 'code_snapshot'
    code.mkdir(exist_ok=True)
    paths = [ROOT / p for p in ('scripts/router_contract_pilot.py',
        'scripts/evaluate_router_contract_pilot.py', 'scripts/run_router_contract_pilot.py',
        'src/dirmo/generation/pose_repair.py', 'src/dirmo/generation/flowr.py',
        'src/dirmo/evaluation/edited_graph.py', 'src/dirmo/evaluation/pb_guided.py',
        'src/dirmo/evaluation/pb_worker.py', 'src/dirmo/evaluation/pose_repair_checks.py',
        'src/dirmo/evaluation/fm_quality.py', 'src/dirmo/evaluation/crossdocked_sbdd.py',
        'src/dirmo/chemistry/graphs.py')]
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}
    record = root / 'execution_protocol.json'
    if record.exists() and read(record)['code_hashes'] != code_hashes:
        raise ValueError('Evaluation/generation implementation changed; use a new run')
    for p in paths:
        dest = code / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    save(record, dict(time_kst=now(), protocol_id=plan['protocol_id'], code_hashes=code_hashes,
        python=sys.executable, environment=environment(), workers=args.workers,
        training=False, llm_calls=0))
    env = {**os.environ, 'TZ': 'Asia/Seoul', 'PYTHONPATH': str(ROOT / 'src')}
    state = dict(started_at_kst=now(), updated_at_kst=now(), status='starting',
        supervisor_pid=os.getpid(), gpu=plan['physical_gpu'], cases=len(plan['cases']),
        planned_fm_draws=plan['planned_fm_draws'], llm_calls=0, training=False)
    children = []
    def stop(signum, frame):
        state.update(status='stopped', updated_at_kst=now(), signal=signum)
        save(root / 'status.json', state)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (root / 'generation.log').open('a') as genlog, (root / 'evaluation.log').open('a') as evallog:
        generator = subprocess.Popen([sys.executable, str(ROOT / 'scripts/router_contract_pilot.py'),
            'generate', '--output', str(root)], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=genlog, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(generator)
        evaluator = subprocess.Popen([sys.executable, str(ROOT / 'scripts/evaluate_router_contract_pilot.py'),
            'watch', '--output', str(root), '--workers', str(args.workers)], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=evallog, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(evaluator)
        state.update(status='running', generation_pid=generator.pid, evaluation_pid=evaluator.pid)
        while True:
            gen_code, eval_code = generator.poll(), evaluator.poll()
            for name in ('generation', 'evaluation'):
                path = root / f'{name}_status.json'
                if path.exists():
                    state[name] = read(path)
            if gen_code is not None and state.get('generation', {}).get('status') not in ('complete', 'failed'):
                # Includes CUDA/import failures before the backend's inner try block.
                failed = dict(status='failed', time_kst=now(), returncode=gen_code,
                    reason='Generation exited before terminal status; see generation.log')
                save(root / 'generation_status.json', failed)
                state['generation'] = failed
            state.update(updated_at_kst=now(), generation_returncode=gen_code, evaluation_returncode=eval_code)
            if gen_code is not None and eval_code is not None:
                ok = gen_code == eval_code == 0 and state.get('evaluation', {}).get('status') == 'complete'
                state['status'] = 'complete' if ok else 'incomplete'
            elif gen_code == 0:
                state['status'] = 'evaluating'
            save(root / 'status.json', state)
            if gen_code is not None and eval_code is not None:
                break
            time.sleep(5)


if __name__ == '__main__':
    main()
