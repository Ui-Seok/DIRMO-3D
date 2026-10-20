"""Retry explicit incomplete CPU evaluations and refresh their original report."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

from benchmark_fm_pose_repair import aggregate, read, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve()
    cases=read(root/'summary.json',{}).get('evaluation_incomplete_cases',[])
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    save(root/'evaluation-recovery.json',{'status':'running','pid':os.getpid(),'cases':cases})
    def evaluate(case_id):
        command=[sys.executable,str(Path(__file__).with_name('benchmark_fm_pose_repair.py')),
                 '--evaluate-case','--case-id',case_id,'--output',str(root)]
        try:
            with (root/'evaluations'/f'{case_id}.recovery.log').open('a') as log:
                result=subprocess.run(command,env=env,stdout=log,stderr=log,timeout=1800)
            return {'case_id':case_id,'returncode':result.returncode}
        except subprocess.TimeoutExpired:
            return {'case_id':case_id,'status':'timeout'}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(evaluate,cases))
    aggregate(root)
    status=read(root/'summary.json')['status']
    save(root/'evaluation-recovery.json',{'status':status,'pid':os.getpid(),'results':results})
    controller=read(root/'controller.json',{})
    controller.update(status=status,evaluation_recovery_pid=os.getpid())
    save(root/'controller.json',controller)


if __name__=='__main__':
    main()
