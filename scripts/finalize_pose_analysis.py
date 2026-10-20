"""Run final CPU-only analysis once the original experiment evaluation ends."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();output=args.output.resolve()
    state=output.parent/(output.name+'.status.json')
    def save(status,**extra):
        temporary=state.with_suffix('.tmp')
        temporary.write_text(json.dumps({'status':status,'pid':os.getpid(),'run':str(root),'output':str(output),
            'at_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),**extra},indent=2)+'\n')
        temporary.replace(state)
    save('waiting_for_all_case_evaluations')
    while True:
        summary=json.loads((root/'summary.json').read_text())
        if summary.get('status') in ('complete','complete_with_missing_evaluation') and summary['evaluation_complete_cases']==summary['cases']:
            break
        time.sleep(60)
    save('analyzing',source_status=summary['status'])
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MPLCONFIGDIR':'/tmp/dirmo_pose_analysis_mpl'}
    command=[sys.executable,str(Path(__file__).with_name('analyze_pose_repair.py')),
             '--run',str(root),'--output',str(output)]
    result=subprocess.run(command,env=env)
    save('complete' if result.returncode==0 else 'analysis_failed',returncode=result.returncode)


if __name__=='__main__':
    main()
