"""Collect missing measured outcomes, then train the Router on physical GPU 1."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from router_contract_pilot import ROOT, now, read, save, sha256_file


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu',default='1');args=ap.parse_args();root=args.output.resolve()
    lock=(root/'training_workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(root/'study.json');assert plan['physical_gpu']==args.gpu
    assert sha256_file(root/'split.json')==plan['split_sha256']
    snapshot=root/'training_code_snapshot';snapshot.mkdir(exist_ok=True)
    paths=[ROOT/'scripts'/n for n in ('prepare_router100.py','train_contract_router.py','run_router100_training.py')]
    paths.append(ROOT/'src/dirmo/routing/spatial.py')
    hashes={str(p.relative_to(ROOT)):sha256_file(p) for p in paths}
    for p in paths:shutil.copy2(p,snapshot/p.name)
    state=dict(status='running',stage='collection',started_at_kst=now(),updated_at_kst=now(),pid=os.getpid(),
        gpu=args.gpu,roots=100,split=read(root/'split.json')['root_counts'],llm_calls=0,
        fm_frozen=True,training_model='Action Router only',code_hashes=hashes,
        training_output=str(root/'router_training'))
    child=None
    def stop(sig,frame):
        state.update(status='stopped',updated_at_kst=now());save(root/'training_workflow.json',state)
        if child and child.poll() is None:child.send_signal(signal.SIGTERM)
        raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env={**os.environ,'TZ':'Asia/Seoul','PYTHONPATH':str(ROOT/'src')}
    try:
        commands=[('collection',[sys.executable,str(ROOT/'scripts/run_router_contract_pilot.py'),
            '--output',str(root),'--workers','6']),
            ('router_training',[sys.executable,str(ROOT/'scripts/train_contract_router.py'),
                '--source',str(root),'--output',str(root/'router_training'),'--gpu',args.gpu])]
        for stage,command in commands:
            if stage=='collection' and (root/'status.json').exists() and read(root/'status.json')['status']=='complete':continue
            if stage=='router_training':
                assert read(root/'status.json')['status']=='complete'
                for p,sha in hashes.items():assert sha256_file(ROOT/p)==sha,'Training code changed in flight'
            state.update(stage=stage,updated_at_kst=now());save(root/'training_workflow.json',state)
            with (root/f'{stage}_workflow.log').open('a') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid
                while child.poll() is None:
                    state['updated_at_kst']=now();save(root/'training_workflow.json',state);time.sleep(5)
                if child.returncode:raise RuntimeError(f'{stage} exited {child.returncode}; see {stage}_workflow.log')
        assert read(root/'router_training/status.json')['status']=='complete'
        state.update(status='complete',stage='complete',updated_at_kst=now())
        save(root/'training_workflow.json',state)
    except Exception as exc:
        state.update(status='failed',updated_at_kst=now(),reason=f'{type(exc).__name__}: {exc}')
        save(root/'training_workflow.json',state);raise


if __name__=='__main__':main()
