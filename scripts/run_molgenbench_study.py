"""Resumable stage supervisor for the approved external SBMO experiment."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from router_contract_pilot import ROOT,now,read,save,sha256_file

def verify_completion(root):
    execution=read(root/'status.json')
    if execution['status']!='complete':
        raise RuntimeError('Collector/evaluator did not complete; see status.json')
    study=read(root/'study.json');result=read(root/'external_results.json')
    if result['completed_targets']!=len(study['cases']) or result['evaluated_rows']!=study['planned_fm_draws']:
        raise RuntimeError('Final report does not cover all scheduled cases/slots')
    if sha256_file(root/'frozen_router.json')!=study['router_sha256']:
        raise RuntimeError('Frozen Router changed')
    save(root/'completion_verification.json',dict(time_kst=now(),status='complete',
        completed_targets=result['completed_targets'],evaluated_rows=result['evaluated_rows']))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--gpu',default='3')
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--verify-terminal',action='store_true',help='One-shot completion guard for an already running workflow')
    args=ap.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.verify_terminal:
        while read(root/'workflow_status.json')['status']=='running':time.sleep(10)
        state=read(root/'workflow_status.json')
        if state['status']=='complete':
            try:verify_completion(root)
            except Exception as exc:
                state.update(status='failed',updated_at_kst=now(),reason=str(exc))
                save(root/'workflow_status.json',state);raise
        return
    lock=(root/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(status='running',stage='starting',started_at_kst=now(),updated_at_kst=now(),
        pid=os.getpid(),gpu=args.gpu,llm_calls=0,fm_training=False,router_training=False)
    status=root/'workflow_status.json';save(status,state);child=None
    def stop(signum,frame):
        if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
        state.update(status='stopped',updated_at_kst=now(),signal=signum);save(status,state)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    paths=['scripts/'+n for n in ('molgenbench_data.py','molgenbench_study.py','molgenbench_report.py',
        'molgenbench_activity.py','run_molgenbench_study.py','fetch_molgenbench.py')]
    paths += ['scripts/router_contract_pilot.py','scripts/evaluate_router_contract_pilot.py',
              'scripts/run_router_contract_pilot.py','scripts/train_contract_router.py']
    hashes={p:sha256_file(ROOT/p) for p in paths}
    saved=root/'workflow_code.json'
    if saved.exists() and read(saved)['hashes']!=hashes:raise ValueError('Workflow changed after launch; review before resuming')
    save(saved,dict(time_kst=now(),hashes=hashes))
    commands=[('audit',[sys.executable,str(ROOT/'scripts/molgenbench_data.py'),'audit','--output',str(root)]),
              ('evidence',[sys.executable,str(ROOT/'scripts/molgenbench_data.py'),'evidence','--output',str(root)]),
              ('prepare',[sys.executable,str(ROOT/'scripts/molgenbench_study.py'),'prepare','--output',str(root),'--gpu',args.gpu]),
              ('generate_evaluate_report',[sys.executable,str(ROOT/'scripts/molgenbench_study.py'),'run','--output',str(root),'--workers',str(args.workers)])]
    try:
        for stage,command in commands:
            state.update(stage=stage,updated_at_kst=now());save(status,state)
            with (root/f'{stage}.log').open('a') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'TZ':'Asia/Seoul'},
                    stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                state['child_pid']=child.pid
                while child.poll() is None:
                    state['updated_at_kst']=now();save(status,state);time.sleep(10)
                if child.returncode:raise RuntimeError(f'{stage} failed, exit={child.returncode}; see {stage}.log')
        verify_completion(root)
        state.update(status='complete',updated_at_kst=now());save(status,state)
    except Exception as exc:
        state.update(status='failed',updated_at_kst=now(),reason=str(exc));save(status,state);raise

if __name__=='__main__':main()
