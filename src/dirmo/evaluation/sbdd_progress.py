"""Read-only observation of benchmark outputs, with KST progress timestamps.

The observer writes progress.json only. The active benchmark owns status.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
from zoneinfo import ZoneInfo

KST = ZoneInfo('Asia/Seoul')


def kst_now():
    return datetime.now(KST).isoformat(timespec='seconds')


def process_alive(pid):
    if not pid:
        return False
    try:
        state = Path(f'/proc/{int(pid)}/stat').read_text().rpartition(') ')[2].split()[0]
        return state != 'Z'
    except (OSError, ValueError, IndexError):
        return False


def collect_progress(root, plan, controller):
    root = Path(root)
    count, generated, complete, generated_cases, references, latest = 0, 0, 0, 0, 0, None
    active = []
    for case in plan['cases']:
        folder = root / 'cases' / case['case_id']
        files = list((folder / 'generated').glob('*/evaluation.json'))
        count += len(files)
        generated += len(list((folder / 'generated').glob('*/result.json')))
        done = (folder / 'evaluation_complete.json').exists()
        has_reference = (folder / 'reference_evaluation.json').exists()
        complete += done
        references += has_reference
        generated_cases += (folder / 'generated/complete.json').exists()
        for path in files:
            timestamp = path.stat().st_mtime
            latest = timestamp if latest is None else max(latest, timestamp)
        if not done and (files or has_reference):
            active.append(dict(case_id=case['case_id'], evaluated_draws=len(files),
                               scheduled_draws=plan['settings']['draws']))
    scheduled = plan['scheduled_draws']
    alive = process_alive(controller.get('pid'))
    status = controller.get('status', 'unknown')
    if not alive and status not in ('complete', 'failed'):
        status = 'stopped'
    stage = ('complete' if status == 'complete' else
             'stopped' if not alive else
             'aggregating' if count == scheduled and references == len(plan['cases']) else
             'evaluating' if generated_cases == len(plan['cases']) else 'generating_and_evaluating')
    return dict(status=status, stage=stage, updated_kst=kst_now(), timezone='Asia/Seoul',
                controller_pid=controller.get('pid'), controller_alive=alive,
                total_cases=len(plan['cases']), generated_cases=generated_cases,
                generated_draws=generated, scheduled_draws=scheduled,
                evaluated_cases=complete, evaluated_draws=count,
                evaluation_percent=round(100 * count / scheduled, 2) if scheduled else 0.,
                evaluated_reference_ligands=references, active_cases=active,
                latest_evaluation_kst=datetime.fromtimestamp(latest, KST).isoformat(timespec='seconds') if latest else None,
                seconds_since_last_evaluation=round(max(0., time.time() - latest), 1) if latest else None,
                failed_evaluations=controller.get('failed_evaluations', []),
                count_definition='evaluated_cases: pockets with all draws processed; evaluated_draws: processed attempts including generation failures and explicitly missing metrics')


def save_progress(path, value):
    """Use a unique temporary file so observers cannot collide on .tmp names."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--interval', type=float, default=15.)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be at least one second')
    root = args.output.resolve()
    plan = json.loads((root / 'study.json').read_text())
    with (root / 'progress_watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_progress(root / 'progress_watcher.json', dict(pid=os.getpid(), started_kst=kst_now(), interval_seconds=args.interval))
        previous = None
        while True:
            controller = json.loads((root / 'status.json').read_text())
            snapshot = collect_progress(root, plan, controller)
            save_progress(root / 'progress.json', snapshot)
            key = snapshot['evaluated_draws'], snapshot['evaluated_cases'], snapshot['status']
            if key != previous:
                print(json.dumps({k: snapshot[k] for k in ('updated_kst', 'status', 'evaluated_draws', 'evaluated_cases', 'evaluation_percent')}), flush=True)
                previous = key
            if not snapshot['controller_alive'] or snapshot['status'] in ('complete', 'failed'):
                break
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
