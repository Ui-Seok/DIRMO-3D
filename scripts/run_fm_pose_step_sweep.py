"""Detached step comparison with independent GPU slots and live-job adoption."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dirmo.chemistry.graphs import graph_check, read_sdf, stereo_from_coordinates, write_sdf
from dirmo.io import sha256_file
from benchmark_fm_pose_repair import read, save
from probe_fm_pose_batch import distort


def extend_plan(source,destination,seeds=16,steps=64,baseline=None):
    """Preserve existing paired inputs; add deterministic independent draws."""
    plan=copy.deepcopy(read(source))
    if not plan:
        raise ValueError(f'No source study: {source}')
    for case in plan['cases']:
        by_key={(i['difficulty'],i['seed']):i for i in case['inputs']}
        missing=any((difficulty,4242+level*10000+offset) not in by_key
                    for level,difficulty in enumerate(('small','large')) for offset in range(seeds))
        native=read_sdf(case['native_sdf']) if missing else None
        for level,difficulty in enumerate(('small','large')):
            for offset in range(seeds):
                seed=4242+level*10000+offset
                if (difficulty,seed) in by_key:
                    continue
                rng=np.random.default_rng(seed)
                angle=float(rng.uniform(20,40) if level==0 else rng.uniform(50,70))*int(rng.choice([-1,1]))
                path=destination/'cases'/case['case_id']/'inputs'/difficulty/f'{seed}.sdf'
                write_sdf(path,stereo_from_coordinates(distort(native,case['region'],angle)))
                if not graph_check(native,read_sdf(path),True)['matches']:
                    raise ValueError(f'Additional input stereo failure: {path}')
                by_key[difficulty,seed]={'case_id':case['case_id'],'input_sdf':str(path),'seed':seed,
                    'difficulty':difficulty,'angle_degrees':angle,'free_maps':case['region']['free_maps'],
                    'pocket_path':case['pocket_path']}
        case['inputs']=[by_key[difficulty,4242+level*10000+offset]
                        for level,difficulty in enumerate(('small','large')) for offset in range(seeds)]
        save(destination/'cases'/case['case_id']/'case.json',case)
    plan.update(seeds=seeds,steps=steps,batch_cap=32,
                planned_outputs=len(plan['cases'])*seeds*2*2,
                source_study=str(source),input_pairing='Exactly the same saved corrupted poses across checkpoints/steps',
                homogeneous_batches=True,backend_sha256=sha256_file(ROOT/'src/dirmo/generation/pose_repair.py'))
    if baseline:
        plan['baseline_output']=str(baseline)
    save(destination/'study.json',plan)
    return plan


def refresh(root,stages,children,status,gpus=None):
    gpus=gpus or ['4']
    rows=[]
    for name,path in stages:
        plan=read(path/'study.json',{})
        summary=read(path/'summary.json',{})
        gpu_done={cp:(path/'phases'/cp/'complete.json').exists() for cp in ('base','v2')}
        batches={}
        for cp in ('base','v2'):
            history=path/'phases'/cp/'batches.jsonl'
            records=[]
            if history.exists():
                for line in history.read_text().splitlines():
                    try: records.append(json.loads(line))
                    except json.JSONDecodeError: pass
            valid=[r for r in records if r.get('status')=='complete']
            batches[cp]={'completed_batches':len(valid),'batch_sizes':dict(Counter(r['batch_size'] for r in valid)),
                         'inference_seconds':sum(r['inference_seconds'] for r in valid),
                         'sample_nfe':sum(r['sample_nfe'] for r in valid),
                         'oom_retries':sum(r.get('status')=='oom' for r in records),
                         'peak_device_used_mib':max((r.get('peak_shared_device_used_mib') or 0 for r in valid),default=0)}
        rows.append({'stage':name,'output':str(path),'steps':plan.get('steps'),'cases':plan.get('actual_cases'),
                     'physical_gpu':str(plan.get('physical_gpu','4')),
                     'gpu_complete':gpu_done,'evaluation_complete_cases':summary.get('evaluation_complete_cases',0),
                     'generation_outputs':summary.get('generation_outputs',{}),'planned_outputs':plan.get('planned_outputs'),
                     'summary_status':summary.get('status'),'batches':batches})
    prepared=read(root/'steps_64'/'preparation.json',{})
    progress={'status':status,'pid':os.getpid(),'physical_gpu':','.join(gpus),'physical_gpus':gpus,'requested_cases':2000,
              'steps':[64,128,256],'maximum_batch_size':32,'seeds_per_difficulty':16,
              'preparation':prepared,'controller_pids':{name:p.pid for name,p in children},'stages':rows,
              'mixed_target_probe':'Rejected: up to 0.112 Angstrom difference; use homogeneous batches'}
    save(root/'progress.json',progress)
    lines=[f"# GPU {', '.join(gpus)} pose 복구: 배치 32 / 2,000 복합체 / step 비교",'',f'상태: {status}.','',
           '각 step의 base와 v2는 같은 GPU에서 병행하고, 서로 다른 step을 GPU별로 나눠 실행한다. 한 복합체에 small/large 각 16개 입력을 만들고 같은 입력으로 64·128·256 step을 비교한다. 실제 배치는 큰 pocket이나 OOM에서 자동으로 낮아진다. 추가 학습과 LLM 호출은 없다.','',
           f"준비 완료: {prepared.get('prepared',0)}/2000. 실제 적격 사례 수는 각 study.json에 기록된다.",'',
           '| 단계 | GPU | 복합체 | base 생성 | v2 생성 | 물리 평가 완료 | GPU 완료(base/v2) |',
           '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in rows:
        lines.append(f"| {row['stage']} | {row['physical_gpu']} | {row['cases']} | {row['generation_outputs'].get('base',0)} | {row['generation_outputs'].get('v2',0)} | {row['evaluation_complete_cases']} | {row['gpu_complete']['base']}/{row['gpu_complete']['v2']} |")
    lines+=['','## 동일 입력에서의 step 비교','',
            '진행 중 수치는 부분 평가 결과다. 성공률 분모는 평가된 결과만이 아니라 전체 예정 입력 수다.','',
            '| Step | 모델 | 오차 | 평가/예정 | Pocket 통과 | Graph 보존 | RMSD 개선 | 종합 성공 |',
            '| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for name,path in stages:
        if not name.startswith('steps_'):
            continue
        summary=read(path/'summary.json',{})
        for group in summary.get('groups',[]):
            if group['method'] not in ('base','v2'):
                continue
            lines.append(f"| {name.removeprefix('steps_')} | {group['method']} | {group['difficulty']} | {group['evaluated']}/{group['expected']} | {group['pocket'].get('valid',0)} | {group['graph_preserved']} | {group['free_rmsd_improved']} | {group['primary_success']} |")
    lines+=['','GPU 생성이 끝나면 CPU 물리 평가를 기다리지 않고 다음 step 생성으로 넘어간다. 최종 완료는 모든 CPU 평가와 대조군 집계가 끝난 시점이다.','',
            '각 단계의 REPORT.md/summary.json 및 evaluations/*.json에 원본·흐트러뜨린 입력·FM 출력의 물리 검사가 기록된다. 동일 입력과 UFF 대조군 측정은 이전 단계 결과를 재사용한다.','',
            '2,000개 확대에서는 서로 다른 복합체의 동일 protein/scaffold를 허용하고 관련 ID를 기록한다. Refined와 other-PL 출처를 사례별로 기록한다. 원본에서 미지정된 stereo는 실제 원본 XYZ로 정의하며 명시된 원본 stereo와 충돌하는 사례는 제외한다. 출력에는 같은 엄격한 stereo 보존 검사를 적용한다.','',
            '혼합 크기 배치 검증에서 단독 생성 대비 최대 0.112 Å 차이를 관측했다. 대규모 결과에는 혼합 크기 배치를 사용하지 않는다.','']
    (root/'REPORT.md').write_text('\n'.join(lines))


def is_live_controller(pid,path):
    """Only adopt the expected benchmark controller, never a reused PID."""
    try:
        command=Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode().split('\0')
        state=Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')',1)[1].strip().split()[0]
        return (state!='Z' and any(Path(a).name=='benchmark_fm_pose_repair.py' for a in command)
                and '--output' in command and Path(command[command.index('--output')+1]).resolve()==path.resolve())
    except (OSError,TypeError,ValueError,IndexError):
        return False


class AdoptedController:
    def __init__(self,path):
        self.path=path
        self.pid=read(path/'controller.json',{}).get('pid')

    def poll(self):
        if is_live_controller(self.pid,self.path):
            return None
        status=read(self.path/'controller.json',{}).get('status')
        return 0 if status in ('complete','complete_with_missing_evaluation') else 1

    @property
    def returncode(self):
        return self.poll()


def gpu_complete(path):
    return all((path/'phases'/cp/'complete.json').exists() for cp in ('base','v2'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-pid',type=int)
    parser.add_argument('--gpus',nargs='+',default=['4'])
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if len(set(args.gpus))!=len(args.gpus) or not set(args.gpus)<={'3','4'}:
        parser.error('Only the authorized distinct GPU selectors 3 and 4 are supported')
    with (root/'sweep.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stages=[];children=[];streams=[]
        def register(name,path,process):
            stages.append((name,path));children.append((name,process))
            return process
        def start(name,path,plan,workers,gpu):
            plan['physical_gpu']=gpu
            save(path/'study.json',plan)
            stream=(root/f'{name}.controller.log').open('a');streams.append(stream)
            command=[sys.executable,str(ROOT/'scripts/benchmark_fm_pose_repair.py'),'--resume',
                     '--cases',str(plan['cases_requested']),'--seeds',str(plan['seeds']),
                     '--batch-size','32','--steps',str(plan['steps']),'--workers',str(workers),
                     '--gpu',gpu,'--output',str(path)]
            process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=stream)
            return register(name,path,process)
        try:
            warmup=root/'batch32_50_case_confirmation'
            warmup_state=read(warmup/'controller.json',{})
            if is_live_controller(warmup_state.get('pid'),warmup) or warmup_state.get('status') in ('complete','complete_with_missing_evaluation'):
                warmup_process=register('batch32_50_case_confirmation',warmup,AdoptedController(warmup))
            else:
                plan=extend_plan(ROOT/'runs/fm_pose_gpu4_50_v1/study.json',warmup)
                warmup_process=start('batch32_50_case_confirmation',warmup,plan,2,args.gpus[0])
            while not gpu_complete(warmup):
                if warmup_process.poll() not in (None,0):
                    raise RuntimeError('Batch-32 confirmation controller failed')
                refresh(root,stages,children,'running',args.gpus)
                time.sleep(20)
            source=root/'steps_64'/'study.json'
            deadline=time.monotonic()+7200
            while not source.exists():
                refresh(root,stages,children,'preparing_2000_cases',args.gpus)
                if args.prepare_pid is None:
                    raise RuntimeError('Prepared study or live --prepare-pid is required')
                try: os.kill(args.prepare_pid,0)
                except ProcessLookupError:
                    raise RuntimeError('Dataset preparation stopped before saving study.json')
                if time.monotonic()>deadline:
                    raise RuntimeError('Dataset preparation exceeded two hours')
                time.sleep(20)
            # Preserve running controllers/outputs when changing only the
            # scheduler. CPU-only controllers do not occupy a GPU slot.
            pending=[]
            for steps in (64,128,256):
                path=root/f'steps_{steps}'
                controller=read(path/'controller.json',{})
                if is_live_controller(controller.get('pid'),path) or controller.get('status') in ('complete','complete_with_missing_evaluation'):
                    register(f'steps_{steps}',path,AdoptedController(path))
                else:
                    pending.append(steps)
            while pending or any(p.poll() is None for _,p in children):
                failed={name:p.returncode for name,p in children if p.poll() not in (None,0)}
                if failed:
                    raise RuntimeError(f'Controllers failed: {failed}')
                occupied={str(read(path/'study.json',{}).get('physical_gpu','4'))
                          for _,path in stages if not gpu_complete(path)}
                for gpu in args.gpus:
                    if not pending or gpu in occupied:
                        continue
                    steps=pending.pop(0);path=root/f'steps_{steps}'
                    plan=extend_plan(source,path,steps=steps,baseline=root/'steps_64' if steps!=64 else None)
                    start(f'steps_{steps}',path,plan,2 if steps==64 else 3,gpu)
                    occupied.add(gpu)
                refresh(root,stages,children,'running' if pending or any(not gpu_complete(p) for _,p in stages)
                        else 'cpu_evaluation',args.gpus)
                time.sleep(20)
            failed={name:p.returncode for name,p in children if p.returncode}
            if failed:
                raise RuntimeError(f'Controllers failed: {failed}')
            statuses=[read(path/'summary.json',{}).get('status') for _,path in stages]
            final='complete_with_missing_evaluation' if 'complete_with_missing_evaluation' in statuses else 'complete'
            refresh(root,stages,children,final,args.gpus)
        except Exception as exc:
            refresh(root,stages,children,'failed',args.gpus)
            save(root/'error.json',{'error':f'{type(exc).__name__}: {exc}','pid':os.getpid()})
            raise
        finally:
            for stream in streams:
                stream.close()


if __name__=='__main__':
    main()
