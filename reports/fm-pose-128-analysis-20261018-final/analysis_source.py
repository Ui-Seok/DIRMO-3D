"""Snapshot completed, matched pose-repair cases without changing live runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

METHODS=('input','UFF','base','v2')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def physical(m,scope):
    if m.get('build_failed'):
        return 'missing'
    pb=m.get(scope,{})
    if pb.get('passes') is True:
        return 'valid'
    return 'invalid' if any(v is False for v in pb.get('checks',{}).values()) else 'missing'


def finite(value):
    return value is not None and np.isfinite(value)


def scalar_row(case,difficulty,seed,method,m,source,native,status='control'):
    rmsd=m.get('free_rmsd');before=source.get('free_rmsd');drift=m.get('held_max_drift')
    graph=bool(m.get('graph_preserved'));held=finite(drift) and drift<=.001
    improved=bool(graph and finite(rmsd) and finite(before) and rmsd<before)
    pocket=physical(m,'pocket');full=physical(m,'full_protein')
    return {'case_id':case['case_id'],'target_id':case['target_id'],
            'scaffold':case['scaffold'] or 'acyclic:'+case['case_id'],
            'dataset':Path(case.get('source_dataset','unknown')).name,
            'atoms':case['atoms'],'pocket_atoms':case['pocket_atoms'],
            'free_atoms':len(case['region']['free_indices']),'difficulty':difficulty,'seed':seed,'method':method,
            'native_pocket_valid':physical(native,'pocket')=='valid',
            'native_full_valid':physical(native,'full_protein')=='valid',
            'input_pocket_category':physical(source,'pocket'),'pocket_category':pocket,'full_category':full,
            'pocket_valid':pocket=='valid','full_valid':full=='valid',
            'built':not m.get('build_failed',False) and bool(m.get('sdf')),
            'generation_status':status,'graph_preserved':graph,'held_preserved':held,
            'held_max_drift':drift if finite(drift) else None,
            'free_rmsd':rmsd if finite(rmsd) else None,
            'input_free_rmsd':before if finite(before) else None,
            'rmsd_improved':improved,'primary_success':bool(graph and held and pocket=='valid' and improved),
            'full_success':bool(graph and held and full=='valid' and improved),
            'material_success_0p1A':bool(graph and held and pocket=='valid' and finite(rmsd) and finite(before) and before-rmsd>=.1),
            'valid_pose_within_1A':bool(graph and held and pocket=='valid' and finite(rmsd) and rmsd<=1.),
            'sdf':m.get('sdf'),'missing_reason':m.get('reason',m.get('error',''))}


def group_summary(frame):
    rows=[]
    for (difficulty,method),g in frame.groupby(['difficulty','method'],sort=True):
        r=g.loc[g.graph_preserved & g.held_preserved & g.free_rmsd.notna()]
        invalid=g[g.input_pocket_category=='invalid'];valid=g[g.input_pocket_category=='valid']
        rows.append({'difficulty':difficulty,'method':method,'n':len(g),
            'pocket_valid_pct':100*float(g.pocket_valid.mean()),'full_valid_pct':100*float(g.full_valid.mean()),
            'pocket_missing':int((g.pocket_category=='missing').sum()),'built':int(g.built.sum()),
            'graph_preserved_pct':100*float(g.graph_preserved.mean()),
            'held_preserved_pct':100*float(g.held_preserved.mean()),
            'max_held_drift':float(g.held_max_drift.max()) if g.held_max_drift.notna().any() else None,
            'conditional_rmsd_n':len(r),'conditional_median_free_rmsd':float(r.free_rmsd.median()) if len(r) else None,
            'paired_median_input_rmsd':float(r.input_free_rmsd.median()) if len(r) else None,
            'median_paired_rmsd_delta':float((r.free_rmsd-r.input_free_rmsd).median()) if len(r) else None,
            'rmsd_improved_pct':100*float(g.rmsd_improved.mean()),
            'primary_success_pct':100*float(g.primary_success.mean()),'full_success_pct':100*float(g.full_success.mean()),
            'material_success_0p1A_pct':100*float(g.material_success_0p1A.mean()),
            'valid_pose_within_1A_pct':100*float(g.valid_pose_within_1A.mean()),
            'invalid_input_n':len(invalid),'invalid_to_valid_pct':100*float(invalid.pocket_valid.mean()) if len(invalid) else None,
            'valid_input_n':len(valid),'valid_to_not_valid_pct':100*float((~valid.pocket_valid).mean()) if len(valid) else None})
    return rows


def paired_bootstrap(rates,metadata,repetitions=2000):
    comparisons=[];rng=np.random.default_rng(20261015)
    for difficulty in ('small','large'):
        group=rates[rates.difficulty==difficulty]
        for metric in ('pocket_valid','full_valid','primary_success'):
            wide=group.pivot(index='case_id',columns='method',values=metric)
            for other in ('v2','UFF'):
                values=(wide['base']-wide[other]).rename('difference').to_frame().join(metadata[['target_id','scaffold']])
                for cluster in ('target_id','scaffold'):
                    pooled=values.groupby(cluster).difference.agg(['sum','count'])
                    sums=pooled['sum'].to_numpy();counts=pooled['count'].to_numpy();samples=[]
                    for start in range(0,repetitions,100):
                        indices=rng.integers(0,len(pooled),size=(min(100,repetitions-start),len(pooled)))
                        samples.extend((sums[indices].sum(1)/counts[indices].sum(1)).tolist())
                    comparisons.append({'difficulty':difficulty,'metric':metric,'comparison':'base-'+other,
                        'cluster':cluster,'clusters':len(pooled),'difference_pp':100*float(values.difference.mean()),
                        'bootstrap95_pp':(100*np.quantile(samples,[.025,.975])).tolist()})
    return comparisons


def audit_generation(root,plan):
    from rdkit import Chem
    results={}
    for cp in ('base','v2'):
        records=[json.loads(line) for line in (root/'phases'/cp/'batches.jsonl').read_text().splitlines()]
        good=[r for r in records if r['status']=='complete']
        statuses=Counter();counts=Counter();bad_counts=Counter()
        for r in good:
            statuses.update(r['statuses']);counts.update(r['case_ids'])
            if r['statuses'].get('contract_failed'):
                bad_counts[r['case_ids'][0]]+=r['statuses']['contract_failed']
        failed=Counter();failure_cases=defaultdict(set);contract_types=Counter();worst_drift=0.
        for case in plan['cases']:
            cid=case['case_id']
            if counts[cid]==len(case['inputs']) and cid not in bad_counts:
                continue
            for item in case['inputs']:
                f=root/'draws'/cp/cid/item['difficulty']/str(item['seed'])/'result.json'
                record=json.loads(f.read_text())
                if record['status']=='failed':
                    reason=record.get('reason','unknown');failed[reason]+=1;failure_cases[reason].add(cid)
                if record.get('status')=='contract_failed':
                    check=record.get('graph_check',{});drift=record.get('held_max_drift',0.)
                    worst_drift=max(worst_drift,drift)
                    a=Chem.MolFromSmiles(check.get('expected_smiles',''));b=Chem.MolFromSmiles(check.get('actual_smiles',''))
                    if check.get('matches') is False and a is not None and b is not None:
                        same=Chem.MolToSmiles(a,isomericSmiles=False)==Chem.MolToSmiles(b,isomericSmiles=False)
                        contract_types['stereo_only' if same else 'nonstereo_graph_difference']+=1
                    if drift>.001:contract_types['held_coordinate_violation']+=1
        statuses['failed']=sum(failed.values())
        assert sum(statuses.values())==plan['planned_outputs']//2,(cp,statuses)
        results[cp]={'scheduled':plan['planned_outputs']//2,'statuses':dict(statuses),
            'raw_sdf_count':sum(r['batch_size']-r['statuses'].get('build_failed',0) for r in good),
            'preprocessing_failures':[{'reason':reason,'samples':n,'cases':sorted(failure_cases[reason])} for reason,n in failed.items()],
            'contract_failure_types':dict(contract_types),'max_failed_contract_held_drift':worst_drift,
            'max_held_tensor_drift':max(r['held_tensor_max_drift'] for r in good),
            'forward_calls':sum(r['forward_calls'] for r in good),'sample_nfe':sum(r['sample_nfe'] for r in good),
            'inference_hours':sum(r['inference_seconds'] for r in good)/3600,
            'peak_shared_gpu_gib':max(r.get('peak_shared_device_used_mib') or 0 for r in good)/1024,
            'oom_retries':sum(r['status']=='oom' for r in records)}
    return results


def figures(output,summary,frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'input':'#a7adb5','UFF':'#c9a24c','base':'#2478ab','v2':'#b46886'}
    labels={'input':'Corrupted input','UFF':'UFF','base':'FLOWR base','v2':'FLOWR v2'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    for ax,field,title in zip(axes,['pocket_valid_pct','primary_success_pct'],['Pocket physical validity','Combined repair success']):
        for j,method in enumerate(METHODS):
            if field=='primary_success_pct' and method=='input':continue
            values=[next(g[field] for g in summary['groups'] if g['method']==method and g['difficulty']==d) for d in ['small','large']]
            ax.bar(np.arange(2)+(j-1.5)*.18,values,width=.17,label=labels[method],color=colors[method])
        ax.set(xticks=[0,1],xticklabels=['Small distortion','Large distortion'],ylim=(0,100),ylabel='Percent of scheduled inputs in completed cases',title=title)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].axhline(summary['native']['pocket_valid_pct'],color='#435847',ls='--',lw=1,label='Native ligand')
    axes[0].legend(fontsize=8,loc='lower left');axes[1].legend(fontsize=8,loc='upper right')
    fig.suptitle(f"64-step pose repair: {summary['cohort']['completed_cases']:,} completed complexes ({summary['status']})",fontsize=13)
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/'overview.png',dpi=180);fig.savefig(output/'overview.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,difficulty in zip(axes,['small','large']):
        for method in METHODS:
            g=frame[(frame.difficulty==difficulty)&(frame.method==method)]
            rmsd=g.loc[g.graph_preserved & g.held_preserved & g.free_rmsd.notna(),'free_rmsd'].to_numpy()
            x=np.sort(rmsd[rmsd<=4.])
            ax.step(x,100*np.arange(1,len(x)+1)/len(g),where='post',label=labels[method],color=colors[method])
        ax.set(xlim=(0,4),ylim=(0,100),xlabel='Free-region RMSD to native (Å)',ylabel='Percent of all scheduled inputs',title=difficulty.capitalize()+' distortion')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('RMSD recovery with graph and held-coordinate preservation; failures remain in denominator')
    fig.savefig(output/'rmsd_ecdf.png',dpi=180);fig.savefig(output/'rmsd_ecdf.pdf');plt.close(fig)


def report(output,summary):
    n=summary['cohort']['completed_cases'];den=n*summary['protocol']['seeds_per_level']
    groups={(g['difficulty'],g['method']):g for g in summary['groups']}
    base_leads=all(groups[d,'base'][metric]>groups[d,m][metric]
                   for d in ['small','large'] for m in ['v2','UFF']
                   for metric in ['pocket_valid_pct','primary_success_pct'])
    finding=('현재 완료 사례에서는 FLOWR base가 v2 및 ligand-only UFF보다 물리 검사 통과와 종합 복구 성공에서 우세하다.'
             if base_leads else '현재 완료 사례의 모델 차이는 아래 물리 검사·RMSD·종합 성공 결과를 구분해서 판단해야 한다.')
    label='중간' if summary['status']=='interim' else '최종'
    lines=['# FLOWR 64-step 부분 pose 복구 결과 분석','',
       f"**{label} 분석: 평가가 종료된 {n:,}/{summary['cohort']['planned_cases']:,}개 복합체를 고정하여 비교했다.**",'',
       finding+' 이는 화학 구조 편집이나 결합 친화도 개선에 대한 결과가 아니다.','',
       f"분석 시작: {summary['snapshot_started_kst']}; 스캔 완료: {summary['snapshot_finished_kst']}.",'',
       '## 실험과 분모','',
       '전체 원소·전하·결합 그래프를 조건으로 고정하고, 결정적으로 선택한 2–8개 말단 원자의 좌표만 바꿨다. 원본에 torsion 오차 small 20–40도 / large 50–70도를 주고 복구했다. 각 수준의 16개는 서로 다른 흐트러뜨린 입력이며, 같은 입력의 독립적인 16회 생성이 아니다. 두 모델은 정확히 같은 입력을 받았다. 64는 ODE integration step 수다.','',
       '입체화학은 별도 고정 tensor가 없으므로 생성 XYZ에서 계산해 원본 stereo와 일치하는지 검사한다. LLM 호출·추가 학습·Vina/친화도 목표는 사용하지 않았다. base는 FLOWR checkpoint 이름이며 initial ligand가 아니다.','',
       f"표의 각 방법·오차 수준 분모는 {den:,}개다. 실패·누락을 분모에서 빼지 않았다. 진행 중 사례의 일부 결과는 배제해 같은 완료 복합체·같은 입력으로 비교한다. 완료 여부에 따른 선택 편향은 남는다.",'',
       '## 전체 2,000개 생성 처리 점검','',
       '| 모델 | 예정 | SDF 저장 | 계약 통과 | 계약 불일치 | 전처리 실패 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for cp,a in summary['generation_audit'].items():
        s=a['statuses'];lines.append(f"| {cp} | {a['scheduled']:,} | {a['raw_sdf_count']:,} | {s.get('ok',0):,} | {s.get('contract_failed',0):,} | {s.get('failed',0):,} |")
    base_failures=summary['generation_audit']['base']['preprocessing_failures']
    total_failed_cases=len(set(cid for f in base_failures for cid in f['cases']))
    failure_description=' / '.join(f"{len(f['cases'])}개: {f['reason']}" for f in base_failures)
    lines+=['',f"두 모델의 공통 전처리 실패는 {total_failed_cases}개 복합체다. {failure_description}. `128,000 처리 완료`는 `128,000개 정상 분자 생성`을 뜻하지 않는다. 실패 기록은 지우거나 쉬운 사례로 교체하지 않았다.",'',
            f"전체 생성 배치의 보호 좌표 tensor 최대 drift: base {summary['generation_audit']['base']['max_held_tensor_drift']} Å, v2 {summary['generation_audit']['v2']['max_held_tensor_drift']} Å. Stereo를 제외한 graph 변경 여부 및 저장 SDF의 drift 검사는 summary.json에 별도 기록했다.",'',
            '## 원본·입력·복구 후 물리 검사','',
            f"원본 ligand: pocket {summary['native']['pocket_valid_pct']:.2f}%, 전체 protein {summary['native']['full_valid_pct']:.2f}% 통과. 원본부터 물리 검사를 통과하지 못한 사례도 포함했다.",'',
            '| 오차 | 방법 | Pocket 통과 | 전체 protein 통과 | 종합 성공 |','| --- | --- | ---: | ---: | ---: |']
    for d in ['small','large']:
        for method in METHODS:
            g=next(g for g in summary['groups'] if g['method']==method and g['difficulty']==d)
            success='—' if method=='input' else f"{g['primary_success_pct']:.2f}%"
            lines.append(f"| {d} | {method} | {g['pocket_valid_pct']:.2f}% | {g['full_valid_pct']:.2f}% | {success} |")
    lines+=['','종합 성공 = 원본 stereo까지 graph 보존 + 보호 좌표 drift ≤0.001 Å + pocket PoseBusters 통과 + 입력보다 자유 영역 RMSD 감소. 입력 자체는 자기 자신보다 개선될 수 없어 종합 성공 비교에서 기준선으로만 쓴다. 이것은 논문에서 가져온 단일 utility 점수가 아니라 이 pose 복구 실험의 명시적인 조건 결합이다.','',
            '전체 protein 검사는 물·보조인자까지 포함하여 고정 pocket 검사보다 엄격하다. UFF는 ligand-only이며 보호 좌표는 고정하지만 protein 에너지 항은 사용하지 않는다.','',
            '## RMSD: 모든 방법이 graph/보호 좌표를 보존한 공통 입력에서 비교','',
            '| 오차 | 공통 입력 수 | input | UFF | base | v2 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for d,r in summary['common_rmsd'].items():
        lines.append(f"| {d} | {r['n']:,} | "+' | '.join(f"{r['median'][m]:.3f} Å" for m in METHODS)+' |')
    lines+=['','RMSD는 protein frame을 유지한 자유 영역 값이다. 전체 ligand를 정답에 정렬하지 않는다. 보호 원자의 대응을 유지하는 graph symmetry를 고려한다. 이 표는 공통으로 비교 가능한 입력에 조건부인 값이므로, graph 실패까지 포함한 전체 복구율은 종합 성공률로 함께 판단해야 한다.','',
            '## 입력 상태별 복구와 훼손','',
            '| 오차 | 방법 | 물리 검사 실패 입력 → 통과 | 통과 입력 → 실패/누락 |','| --- | --- | ---: | ---: |']
    for d in ['small','large']:
        for method in ['UFF','base','v2']:
            g=next(g for g in summary['groups'] if g['method']==method and g['difficulty']==d)
            lines.append(f"| {d} | {method} | {g['invalid_to_valid_pct']:.2f}% | {g['valid_to_not_valid_pct']:.2f}% |")
    lines+=['','## 같은 복합체에서의 차이와 불확실성','',
            '각 복합체의 16개 입력을 먼저 평균하고 exact protein sequence ID로 묶어 paired bootstrap을 수행했다. 32개 입력을 독립적인 단백질처럼 세지 않았다. 아래 구간은 현재 완료 cohort에 조건부인 탐색적 95% bootstrap 구간이다. Scaffold grouping 결과도 JSON에 저장했다.','',
            '| 오차 | 비교 | 지표 | 차이(%p) | Protein cluster bootstrap 95% |','| --- | --- | --- | ---: | --- |']
    for c in summary['paired_comparisons']:
        if c['cluster']=='target_id' and c['metric'] in ['pocket_valid','primary_success']:
            a,b=c['bootstrap95_pp'];lines.append(f"| {c['difficulty']} | {c['comparison']} | {c['metric']} | {c['difference_pp']:+.2f} | [{a:+.2f}, {b:+.2f}] |")
    lines+=['','## 잘 안 되는 부분','',
            'Physical validity 통과와 원본 pose 복구는 다르다. 유효한 다른 conformation을 만들면 물리 검사에는 통과해도 RMSD 복구에 실패할 수 있다. 반대로 RMSD가 줄어도 잘못된 stereo 또는 원자 겹침이 있으면 종합 성공이 아니다.','',
            ('v2는 graph/stereo 계약 위반은 적지만 이번 결과의 종합 복구 성공률은 base보다 낮다.'
             if all(groups[d,'base']['primary_success_pct']>groups[d,'v2']['primary_success_pct'] for d in ['small','large'])
             else 'Graph/stereo 보존율과 종합 pose 복구 성공률은 별도로 판단해야 한다.')+
            ' Base도 모든 입력을 개선하지 않으므로 통과 입력을 나쁘게 만드는 비율과 UFF 대비 차이를 함께 확인해야 한다.','',
            'Base의 물리 검사 통과율 향상은 주로 protein과의 접촉 충돌 감소와 함께 나타난다. 그러나 자유 영역의 결합 길이·각도·내부 에너지 오류가 늘어난 경우도 있다. Graph의 결합 종류를 고정하는 것과 3D 결합 길이를 올바르게 유지하는 것은 서로 다른 조건이다. 검사별 수치는 FAILURE_ANALYSIS.md에서 확인한다.','',
            '## 해석의 한계','',
            (f"미완료 {summary['cohort']['pending_cases']}개 복합체는 이 비교에 포함하지 않았다. GPU는 큰 pocket부터 처리하고 CPU도 완료 순서에 영향을 받으므로 완료 cohort를 무작위 표본으로 해석하면 안 된다."
             if summary['cohort']['pending_cases'] else '예정한 2,000개 복합체의 평가가 종료됐다. 평가 누락 및 전처리 실패는 summary.json에서 별도로 확인해야 한다.')+
            ' 구조·단백질·scaffold 분포는 cohort JSON에 저장했다.','',
            '동일 protein/scaffold의 서로 다른 복합체가 포함돼 있다. Exact sequence grouping은 서열 유사도 clustering과 다르며, FM 사전학습 데이터 중복도 미확인이다. 독립적인 unseen-target 일반화나 ligand affinity 개선을 주장할 수 없다.','',
            ('현재 조건의 작은 말단 영역 pose 복구에서는 base를 우선 후보로 둘 근거가 있다. ' if base_leads else '')+
            '128·256-step에서도 같은 완료 복합체를 맞춰 물리 검사·stereo 보존·종합 성공·NFE를 비교한 뒤 step 수의 효과를 판단해야 한다. 이 결과만으로 FM 재학습이 필요하거나 불필요하다고 단정할 수 없다.','',
            '## 산출물','',
            '- `summary.json`: 집계, 분모, 공통 RMSD, paired bootstrap, 생성 실패 감사 결과.','- `cohort.json`: 완료/대기 사례와 읽은 평가 파일 SHA-256.','- `metrics.csv.gz`: 현재 cohort의 입력별 요약 측정값.','- `case_rates.csv`: 복합체별·오차별·방법별 성공 비율.','- `failure_checks.json`: 검사별 실패 수(한 분자가 여러 검사에 실패할 수 있음).','- `figures/overview.{png,pdf}` 및 `figures/rmsd_ecdf.{png,pdf}`: 고정된 snapshot의 그림.','']
    lines+=['분석 코드: `scripts/analyze_pose_repair.py`. 이 분석은 저장된 JSON/SDF만 읽으며 실행 중인 생성·평가 프로세스나 조건을 변경하지 않는다.','']
    lines+=['## 저장된 출력의 stereo 검사','']
    for cp,a in summary['generation_audit'].items():
        stereo=a['contract_failure_types'].get('stereo_only',0)
        lines.append(f"- {cp}: stereo만 불일치한 저장 출력 {stereo:,}/{a['raw_sdf_count']:,}개 ({100*stereo/a['raw_sdf_count']:.2f}%).")
    lines+=['','원소·연결·결합 종류와 보호 좌표 조건을 유지해도 자유 영역의 3D stereo는 자동 보장되지 않는다. 잘못된 stereo 출력은 종합 성공으로 세지 않았다.','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    failures=json.loads((output/'failure_checks.json').read_text())
    lines=['# 검사별 실패율','',
           f"각 방법·오차 수준의 분모는 {den:,}개다. 한 입력이 여러 검사에 실패할 수 있다. 생성 전처리 실패는 no_molecule로 별도 집계한다.",'',
           '| 오차 | 방법 | Protein 최소거리 | 결합 길이 | 내부 에너지 | 내부 원자 겹침 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for d in ['small','large']:
        for method in METHODS:
            checks=failures[f'{d}:{method}:pocket']
            keys=['minimum_distance_to_protein','bond_lengths','internal_energy','internal_steric_clash']
            lines.append(f'| {d} | {method} | '+' | '.join(f"{100*checks.get(k,0)/checks['scheduled']:.2f}%" for k in keys)+' |')
    lines+=['','이 집계는 접촉 충돌 감소와 내부 geometry 손상이 함께 관찰되는지 비교하는 자료다. 두 현상의 인과관계를 직접 입증하지는 않는다.','']
    (output/'FAILURE_ANALYSIS.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Choose a new analysis output to preserve its cohort snapshot')
    (out/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    started=datetime.now(ZoneInfo('Asia/Seoul')).isoformat();plan_bytes=(root/'study.json').read_bytes();plan=json.loads(plan_bytes)
    records=[];natives=[];manifest=[];pending=[];checks=defaultdict(Counter);terminal_missing=[]
    for i,case in enumerate(plan['cases']):
        f=root/'evaluations'/f"{case['case_id']}.json"
        if not f.exists():pending.append(case['case_id']);continue
        raw=f.read_bytes();data=json.loads(raw)
        if not data.get('complete'):pending.append(case['case_id']);continue
        if data.get('evaluation_incomplete'):terminal_missing.append(case['case_id'])
        manifest.append({'case_id':case['case_id'],'sha256':sha256(raw).hexdigest(),'bytes':len(raw)})
        native=data.get('native',{});nr=scalar_row(case,'native',0,'native',native,native,native)
        natives.append(nr)
        for item in case['inputs']:
            d=item['difficulty'];seed=item['seed'];key=f'{d}/{seed}';source=data.get('inputs',{}).get(key,{}).get('metrics',{'build_failed':True})
            values={'input':(source,'control'),'UFF':(data.get('relaxed',{}).get(key,{}).get('metrics',{'build_failed':True}),'control')}
            for cp in ['base','v2']:
                g=data.get('generated',{}).get(cp,{}).get(key,{})
                values[cp]=(g.get('metrics',{'build_failed':True}),g.get('generation_status','missing'))
            for method,(m,status) in values.items():
                records.append(scalar_row(case,d,seed,method,m,source,native,status))
                for scope in ['pocket','full_protein']:
                    counter=checks[f'{d}:{method}:{scope}'];counter['scheduled']+=1
                    if m.get('build_failed'):counter['no_molecule']+=1
                    for check,value in m.get(scope,{}).get('checks',{}).items():
                        if value is not True:counter[check+(':missing' if value is None else '')]+=1
        if len(natives)%200==0:print(json.dumps({'loaded_completed_cases':len(natives)}),flush=True)
    if not natives:raise ValueError('No terminal case evaluations to analyze')
    frame=pd.DataFrame(records);native_frame=pd.DataFrame(natives)
    assert len(frame)==len(natives)*plan['seeds']*2*len(METHODS)
    assert not frame.duplicated(['case_id','difficulty','seed','method']).any()
    flags=['pocket_valid','full_valid','graph_preserved','held_preserved','rmsd_improved','primary_success','full_success','material_success_0p1A','valid_pose_within_1A']
    rates=frame.groupby(['case_id','difficulty','method'])[flags].mean().reset_index()
    metadata=native_frame.set_index('case_id')
    comparisons=paired_bootstrap(rates,metadata)
    common={}
    for d in ['small','large']:
        part=frame[frame.difficulty==d].copy()
        part['eligible']=part.graph_preserved & part.held_preserved & part.free_rmsd.notna()
        wide=part.pivot(index=['case_id','seed'],columns='method',values='free_rmsd')
        keep=part.pivot(index=['case_id','seed'],columns='method',values='eligible').all(axis=1)
        eligible=wide[keep]
        common[d]={'n':len(eligible),'scheduled_per_method':len(part)//4,
            'median':{m:float(eligible[m].median()) for m in METHODS},
            'median_paired_delta':{m:float((eligible[m]-eligible['input']).median()) for m in METHODS}}
    selected=set(native_frame.case_id)
    def shape_stats(group):
        return {'n':len(group),'median_ligand_atoms':float(np.median([c['atoms'] for c in group])),
                'median_pocket_atoms':float(np.median([c['pocket_atoms'] for c in group])),
                'mean_total_atoms':float(np.mean([c['atoms']+c['pocket_atoms'] for c in group]))} if group else {'n':0}
    cohort={'planned_cases':len(plan['cases']),'completed_cases':len(natives),'pending_cases':len(pending),
            'terminal_incomplete_cases':terminal_missing,'unique_exact_proteins':native_frame.target_id.nunique(),
            'completed_shape':shape_stats([c for c in plan['cases'] if c['case_id'] in selected]),
            'pending_shape':shape_stats([c for c in plan['cases'] if c['case_id'] not in selected]),
            'source_datasets':native_frame.dataset.value_counts().to_dict()}
    summary={'status':'interim' if pending else 'complete','snapshot_started_kst':started,
        'snapshot_finished_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'run':str(root),
        'plan_sha256':sha256(plan_bytes).hexdigest(),'analysis_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol':{'steps':plan['steps'],'seeds_per_level':plan['seeds'],'levels':['small','large'],
                    'unit':'different corrupted input per seed; not repeated random generations of one input',
                    'missing_outputs_retained_in_denominator':True},'cohort':cohort,
        'native':{'n':len(natives),'pocket_valid_pct':100*float(native_frame.pocket_valid.mean()),
                  'full_valid_pct':100*float(native_frame.full_valid.mean())},
        'groups':group_summary(frame),'native_valid_groups':group_summary(frame[frame.native_pocket_valid]),
        'source_groups':{s:group_summary(frame[frame.dataset==s]) for s in native_frame.dataset.unique()},
        'common_rmsd':common,'paired_comparisons':comparisons,'generation_audit':audit_generation(root,plan)}
    frame.to_csv(out/'metrics.csv.gz',index=False,compression='gzip');rates.to_csv(out/'case_rates.csv',index=False)
    native_frame.to_csv(out/'native_cases.csv',index=False)
    save(out/'cohort.json',{'started_kst':started,'finished_kst':summary['snapshot_finished_kst'],'evaluated_files':manifest,'pending_case_ids':pending})
    save(out/'failure_checks.json',{key:dict(value) for key,value in checks.items()});save(out/'summary.json',summary)
    figures(out/'figures',summary,frame);report(out,summary)
    print(json.dumps({'status':summary['status'],'cohort_cases':len(natives),'report':str(out/'REPORT.md')}),flush=True)


if __name__=='__main__':
    main()
