"""Give a saved GPU4 job compute time while a priority job waits for CPU work."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal

from .io import append_jsonl, write_json


def worker_state(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]


def worker_command(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')


def set_saved_workers_paused(pause_file, paused, experiment, reason):
    path = Path(pause_file)
    record = json.loads(path.read_text())
    # Validate every identity before sending any signal. A stale/reused PID
    # must never affect an unrelated process.
    workers = []
    for info in record['workers'].values():
        pid = info['pid']
        if worker_command(pid) != info['command']:
            raise RuntimeError(f'Saved GPU worker PID changed identity: {pid}')
        if '--gpu' not in info['command'] or info['command'][info['command'].index('--gpu') + 1] != '4':
            raise RuntimeError('Only the saved GPU4 workers may be paused/resumed')
        workers.append((pid, worker_state(pid)))
    changed = []
    for pid, state in workers:
        if paused and state != 'T':
            os.kill(pid, signal.SIGSTOP); changed.append(pid)
        elif not paused and state == 'T':
            os.kill(pid, signal.SIGCONT); changed.append(pid)
    record.update(status='gpu_paused' if paused else 'gpu_resumed',
                  priority_experiment=str(experiment), last_switch_reason=reason,
                  last_switch_utc=datetime.now(timezone.utc).isoformat())
    write_json(path, record)
    if changed:
        append_jsonl(Path(experiment) / 'gpu_switches.jsonl',
                     {'paused': paused, 'pids': changed, 'reason': reason, 'utc': record['last_switch_utc']})
    return changed
