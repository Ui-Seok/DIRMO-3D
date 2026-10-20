"""Separate explicit graph-edit effects from pose repair in the completed cohort."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from dirmo.chemistry.graphs import read_sdf,smiles,stereo_from_coordinates
from dirmo.evaluation.edited_graph import graph_from_record
from dirmo.io import sha256_file
from analyze_edited_graph_study import index_signature

VARIANTS=['C_to_N','C_to_O','C_to_F']


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def ci(values,median=True):
    values=np.asarray(values,float)
    rng=np.random.default_rng(20261018)
    sample=values[rng.integers(0,len(values),(5000,len(values)))]
    return np.quantile(np.median(sample,axis=1) if median else sample.mean(axis=1),[.025,.975]).tolist()


def table(rows):
    frame=pd.DataFrame(rows)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join('---' for _ in frame.columns)+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in frame.itertuples(index=False,name=None)])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/fm_edited_graph_gpu3_100_20261016_v1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.run.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'study.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert summary['evaluation_complete_cases']==100 and summary['fm_terminal_slots']==6400
    frame=pd.read_csv(root/'metrics.csv');assert len(frame)==16100
    cases={c['case_id']:c for c in plan['cases']}
    chemical=pd.DataFrame([dict(case_id=c['case_id'],variant=g['name'],**g['properties']) for c in plan['cases'] for g in c['graphs']])
    changes=[]
    for prop in ['logp','qed','sa','mw','hbd','hba','tpsa']:
        wide=chemical.pivot(index='case_id',columns='variant',values=prop)
        for v in VARIANTS:
            delta=wide[v]-wide.original
            changes.append({'variant':v,'property':prop,'cases':len(delta),'mean_delta':float(delta.mean()),
                            'median_delta':float(delta.median()),'mean_bootstrap95':ci(delta,False),
                            'increased':int((delta>1e-9).sum()),'decreased':int((delta < -1e-9).sum())})
    save(out/'chemical_changes.json',changes);chemical.to_csv(out/'target_graph_properties.csv',index=False)
    core=frame[frame.kind=='fm_core'].copy();assert len(core)==3200
    eligible=core[core.success & (core.vina_status=='ok')]
    means=eligible.groupby(['case_id','variant']).vina.mean().unstack()
    counts=eligible.groupby(['case_id','variant']).size().unstack()
    comparisons=[];paired=[]
    native=frame[(frame.kind=='native') & frame.success & (frame.vina_status=='ok')].set_index('case_id').vina
    for v in ['original']+VARIANTS:
        reference=means[v].dropna().to_frame('generated').join(native.rename('native'),how='inner')
        reference['delta_vina']=reference.generated-reference.native
        reference.to_csv(out/f'{v}_vs_native_vina.csv')
    for v in VARIANTS:
        rows=means[['original',v]].dropna();delta=rows[v]-rows.original
        comparisons.append({'variant':v,'pockets':len(rows),'median_delta_vina':float(delta.median()),
                            'mean_delta_vina':float(delta.mean()),'lower_score':int((delta<0).sum()),
                            'higher_score':int((delta>0).sum()),'median_bootstrap95':ci(delta)})
        for cid,row in rows.iterrows():
            paired.append({'case_id':cid,'variant':v,'original_vina':row.original,'edited_vina':row[v],
                           'delta_vina':row[v]-row.original,'original_valid_poses':int(counts.loc[cid,'original']),
                           'edited_valid_poses':int(counts.loc[cid,v])})
    pd.DataFrame(paired).to_csv(out/'paired_graph_vina.csv',index=False)
    save(out/'paired_graph_vina_summary.json',comparisons)
    prior=frame[frame.kind=='prior'][['case_id','variant','draw','vina','success']]
    same=core.merge(prior,on=['case_id','variant','draw'],suffixes=('','_prior'))
    pose=[]
    for name in ['all_scored','both_success']:
        rows=same[same.vina.notna() & same.vina_prior.notna()]
        if name=='both_success':rows=rows[rows.success & rows.success_prior]
        delta=rows.vina-rows.vina_prior
        pose.append({'subset':name,'pairs':len(rows),'pockets':rows.case_id.nunique(),
                     'median_delta_vina':float(delta.median()),'lower_score':int((delta<0).sum())})
        rows.to_csv(out/f'pose_repair_{name}.csv',index=False)
    save(out/'pose_repair_summary.json',pose)
    # Independent verification of all saved core-conditioned outputs.
    graphs={(cid,g['name']):graph_from_record(g) for cid,c in cases.items() for g in c['graphs']}
    audit=[]
    for row in core.itertuples(index=False):
        if pd.isna(row.sdf):continue
        case=cases[row.case_id];graph=graphs[row.case_id,row.variant]
        record=next(g for g in case['graphs'] if g['name']==row.variant)
        mol=read_sdf(row.sdf);xyz=mol.GetConformer().GetPositions()
        max_drift=max(float(np.linalg.norm(xyz[int(i)]-position)) for i,position in case['held_xyz'].items())
        changed=record['changed_index']
        audit.append({'case_id':row.case_id,'variant':row.variant,'draw':row.draw,
                      'index_graph_preserved':index_signature(graph)==index_signature(mol),
                      'target_element_preserved':True if changed is None else graph.GetAtomWithIdx(changed).GetAtomicNum()==mol.GetAtomWithIdx(changed).GetAtomicNum(),
                      'stereo_preserved':smiles(stereo_from_coordinates(mol))==smiles(graph),
                      'core_max_drift':max_drift})
    audit=pd.DataFrame(audit);audit.to_csv(out/'core_output_audit.csv',index=False)
    assert len(audit)==2913 and audit.index_graph_preserved.all() and audit.target_element_preserved.all()
    assert audit.core_max_drift.le(.001).all() and int(audit.stereo_preserved.sum())==2703
    first=plan['cases'][0]['case_id']
    example=[r for r in paired if r['case_id']==first]
    result={'status':'complete','analysis_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
            'run':str(root),'cases':100,'receptor_available_cases':int((frame[(frame.kind=='native')].vina_status=='ok').sum()),
            'saved_core_outputs':len(audit),'primitive_graph_preserved':int(audit.index_graph_preserved.sum()),
            'specified_edit_delivered':int(audit[audit.variant!='original'].target_element_preserved.sum()),
            'edited_saved_outputs':int((audit.variant!='original').sum()),'max_core_drift':float(audit.core_max_drift.max()),
            'stereo_preserved':int(audit.stereo_preserved.sum()),'chemical_changes':changes,
            'edited_vs_original_fm':comparisons,'pose_repair':pose,'first_case_example':example,
            'analysis_sha256':sha256_file(__file__),'metrics_sha256':sha256_file(root/'metrics.csv')}
    save(out/'summary.json',result)
    plot(out,comparisons,changes,pd.DataFrame(paired))
    chem_rows=[]
    for v in VARIANTS:
        item={'편집':v.replace('C_to_','C→')}
        for prop in ['logp','qed','sa']:
            value=next(x for x in changes if (x['variant'],x['property'])==(v,prop))['mean_delta']
            item[f'평균 Δ{prop}']=f'{value:+.3f}'
        chem_rows.append(item)
    vina_rows=[{'편집':x['variant'],'비교 pocket':x['pockets'],'ΔVina 중앙값':f"{x['median_delta_vina']:+.3f}",
                '낮은 점수':x['lower_score'],'높은 점수':x['higher_score'],
                '탐색적 95% CI':f"[{x['median_bootstrap95'][0]:+.3f}, {x['median_bootstrap95'][1]:+.3f}]"} for x in comparisons]
    first_rows=[{'편집':x['variant'],'원본 graph FM 평균 Vina':f"{x['original_vina']:.3f}",
                 '편집 graph FM 평균 Vina':f"{x['edited_vina']:.3f}",'ΔVina':f"{x['delta_vina']:+.3f}"} for x in example]
    lines=['# GPU3: 화학 편집과 core 고정 생성의 성능 변화 — 최종 100개','',
           '**지정한 화학 교체와 core 좌표 고정은 작동했다. 물성은 실제로 변했지만 전반적인 성능 향상은 확인되지 않았다. 목표 기반 편집 최적화 실험은 아니었다.**','',
           f"분석 시각: {result['analysis_kst']}. 원본: `{root}`.",'',
           '## 수행한 작업','',
           '기존 말단 영역에서 원자 하나를 정해 C→N/O/F로 미리 교체했다. RDKit이 target graph를 만들고 FM에는 이미 편집된 전체 graph·pocket·core 좌표·시작 pose를 전달했다. FM은 target graph를 유지하면서 자유 영역 좌표를 생성했다. 원자 수와 결합 연결/차수는 그대로이며 fragment 크기 변경, LLM 편집 선택, QED/SA/Vina 목표 조건은 없었다. 편집 위치와 원소를 성능으로 순위 매기지 않았다. 원본 graph의 FM 생성도 대조군으로 만들었다.','',
           '## 실제 조건 이행','',
           f"Core 조건의 예정 결과 3,200개 중 SDF 2,913개가 생성됐다. 초기화 실패 255개와 지원 불가 32개도 분모에 남겼다. 독립 검사에서 2,913개 모두 원자 순서별 비입체 graph를 유지했고, 편집 graph의 2,188개 모두 요청한 N/O/F 원소를 유지했다. 보호 좌표 최대 이동은 {result['max_core_drift']:.6f} Å였다. 다만 입체화학까지 보존한 것은 2,703/2,913개였다.",'',
           '## 화학 graph 자체의 물성 변화','',table(chem_rows),'',
           '100개 pocket의 target graph에서 원본 대비 평균 변화다. 일부 target은 생성 실패/지원 불가였으며 이 표는 설계 graph 자체의 계산 성질을 말한다. LogP 감소는 덜 지용성이 되는 방향이지만 무조건 성능 향상은 아니다. QED는 높을수록, SA는 낮을수록 해당 지표상 유리하다. N/O 교체는 LogP를 약 1 낮췄지만 QED와 SA는 평균적으로 나빠졌다. 이 변화는 RDKit graph 편집의 효과이며 FM 좌표 생성이 만든 변화가 아니다. pKa·실측 용해도는 측정하지 않았다.','',
           '## 편집이 원본 graph보다 결합 점수를 개선했는가','',
           '각 pocket에서 PB·graph/입체화학·core 보존을 모두 통과한 pose의 평균 Vina를 graph별로 계산했다. 편집과 원본 graph 양쪽에 유효 후보와 점수가 있는 pocket끼리 비교한 후 Δ=편집 평균−원본 평균의 pocket 중앙값을 구했다. 음수가 낮은 점수다. Vina receptor 준비는 58/100개 성공했으며 양쪽 유효 후보 조건으로 비교 대상은 더 줄었다. 모든 결과를 섞은 비대응 중앙값을 빼지 않았다.','',table(vina_rows),'',
           'N/O는 이 조건부 비교에서 점수가 나빠진 pocket이 더 많았다. F는 평균 Δ와 중앙 Δ의 방향이 달라 일부 큰 개선 사례에 민감했고, 중앙 변화가 매우 작았다. Graph 간 seed와 성공 pose 수가 달라 이 차이를 원자 교체만의 인과 효과로 단정하지 않는다. CI는 pocket 재표집 5,000회의 탐색적 범위이며 사후 성공 subset·여러 편집 비교의 영향을 제거하지 않는다. Vina는 score-only 대리 지표이며 실험 affinity가 아니다.','',
           '## FM의 좌표 생성 자체는 효과가 있었는가','',
           '같은 graph·같은 draw의 시작 pose와 core 고정 FM 출력을 비교했다. 전체 점수 쌍 1,736개/57 pockets에서 ΔVina 중앙값 −2.573 kcal/mol, 1,594개 쌍이 낮아졌다. 시작과 출력이 모두 종합 성공인 535쌍/39 pockets에서도 중앙값 −0.191, 435쌍이 낮아졌다. 이는 초기화 pose를 다듬은 효과이며 화학 편집의 이득과 구분해야 한다.','',
           '원본 실험 pose와 비교한 결과도 각 *_vs_native_vina.csv에 저장했다. 원본 graph FM과 PB 통과 native를 비교할 수 있는 47 pockets에서 평균 점수 차이의 중앙값은 +0.144 kcal/mol, 낮아진 pocket은 13개였다. 초기화 pose보다 개선됐다는 사실로 실험 원본보다 좋아졌다고 주장할 수 없다.','',
           f'## 개별 예시: 선정 순서 첫 pocket {first}','',table(first_rows),'',
           '이 예시는 cohort 첫 사례로 고정해 제시했다. 개별 개선 사례가 있더라도 전체 표의 결과와 함께 해석해야 한다. 분자 원자 대응과 정확한 교체 위치는 원본 cases/*/case.json의 changed_index와 atom_maps에 기록돼 있다.','',
           f"![Chemical edit and Vina changes]({out/'chemical_edit_effects.png'})",'',
           '## 해석','',
           '이 실험은 명시한 원자로 교체한 graph를 받아 core 좌표를 유지하며 pose를 생성할 수 있는지 확인했다. 어떤 부분이 나빠서 무엇으로 바꾸면 목표가 좋아지는지 선택하는 최적화 정책은 시험하지 않았다. 다음 목표 기반 실험에서는 목표 범위·보호 조건을 먼저 정하고 편집 후보를 제안한 후 같은 PB·graph·좌표·Vina·물성 검사로 후보를 평가해야 한다.','',
           '## 데이터','',
           '- summary.json: 완료 100개 기준 요약·입력 hash.','- target_graph_properties.csv / chemical_changes.json: graph별 물성과 평균 변화.',
           '- paired_graph_vina.csv: 같은 pocket에서 원본/편집 graph의 성공 pose 평균 점수와 후보 수.',
           '- core_output_audit.csv: 저장 SDF의 graph·지정 원소·입체화학·core 좌표 독립 검사.',
           '- pose_repair_*.csv: 같은 graph/시작 pose와 FM 출력의 비교.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'complete','report':str(out/'REPORT.md'),'saved_core_outputs_checked':len(audit),'comparisons':comparisons},ensure_ascii=False))


def plot(out,comparisons,changes,paired):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
    fig,axes=plt.subplots(1,4,figsize=(15,4.2),layout='constrained')
    colors=['#7567a8','#269b9a','#c69c43']
    for ax,prop,title in zip(axes[:3],['logp','qed','sa'],['LogP','QED (higher is favorable)','SA (lower is favorable)']):
        entries=[next(x for x in changes if (x['variant'],x['property'])==(v,prop)) for v in VARIANTS]
        ys=np.array([x['mean_delta'] for x in entries]);intervals=np.array([x['mean_bootstrap95'] for x in entries])
        ax.bar(range(3),ys,color=colors,alpha=.85)
        ax.errorbar(range(3),ys,yerr=np.maximum(0,[ys-intervals[:,0],intervals[:,1]-ys]),fmt='none',ecolor='#333',capsize=3)
        ax.axhline(0,color='gray',lw=.8)
        ax.set(xticks=range(3),xticklabels=['C to N','C to O','C to F'],ylabel='Mean edited minus original',title=title)
    ax=axes[3]
    for i,(v,color) in enumerate(zip(VARIANTS,colors)):
        values=paired[paired.variant==v].delta_vina.to_numpy()
        ax.scatter(i+np.random.default_rng(i).uniform(-.11,.11,len(values)),values,s=15,color=color,alpha=.65)
        median=next(x['median_delta_vina'] for x in comparisons if x['variant']==v)
        ax.plot([i-.14,i+.14],[median,median],color='black',lw=2)
    ax.axhline(0,color='gray',lw=.9,ls='--')
    ax.set(xticks=range(3),xticklabels=['C to N','C to O','C to F'],ylabel='Delta Vina (kcal/mol)',title='Core-fixed successful poses\nOne paired difference per pocket')
    fig.suptitle('Explicit chemical edits: property changes and paired Vina | 100-pocket completed study',fontsize=12)
    fig.savefig(out/'chemical_edit_effects.png');fig.savefig(out/'chemical_edit_effects.pdf');plt.close(fig)


if __name__=='__main__':main()
