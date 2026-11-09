"""Report all intended external policies, preserving missingness and failures."""
from collections import Counter,defaultdict
import csv
from pathlib import Path
import subprocess
import numpy as np
from router_contract_pilot import ROOT, read, save, now
from molgenbench_data import identity
from dirmo.chemistry.graphs import read_sdf,map_indices
from dirmo.evaluation.edited_graph import graph_from_record

def core_drift(parent,target_graph,actual,core):
    """FLOWR omits atom-map metadata; use its verified fixed tensor atom order.

    Never remap a symmetric core to minimize movement or align to the parent.
    Reject changes to indexed connectivity rather than invent correspondence.
    """
    def indexed_graph(m):
        atoms=[(a.GetAtomicNum(),a.GetIsotope()) for a in m.GetAtoms()]
        bonds=sorted((min(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),max(b.GetBeginAtomIdx(),b.GetEndAtomIdx()),
                      str(b.GetBondType())) for b in m.GetBonds())
        return atoms,bonds
    if indexed_graph(target_graph)!=indexed_graph(actual):
        raise ValueError('Output no longer has the fixed target tensor graph order')
    pl=map_indices(parent);tl=map_indices(target_graph)
    px=parent.GetConformer().GetPositions();ax=actual.GetConformer().GetPositions()
    distances=[float(np.linalg.norm(ax[tl[m]]-px[pl[m]])) for m in core]
    return dict(mcs_max_drift=max(distances),mcs_rmsd=float(np.sqrt(np.mean(np.square(distances)))))

def payoff(row,stage='refined'):
    if row is None:return None
    result=row.get(stage)
    if result is None:
        # An observed execution failure returns the parent with realized gain 0.
        # Missing evaluation/timeout is NOT imputed as an affinity measurement.
        if row['generation_status']=='initialization_failed':return 0.
        if row['generation_status'] in ('generation_error','build_failed') and stage!='uff_only':return 0.
        return None
    if result['joint_physical'] is False:return 0.
    if result['joint_physical'] is None:return None
    delta=result.get('vina_improvement_vs_initial')
    return max(delta,0.) if delta is not None else None

def mean_complete(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None

def bootstrap(values,seed=20261105):
    if len(values)<3:return None
    x=np.asarray(values);rng=np.random.default_rng(seed)
    return np.quantile([rng.choice(x,len(x),replace=True).mean() for _ in range(2000)],[.025,.975]).tolist()

def report(root):
    root=Path(root);plan=read(root/'study.json');split=read(root/'target_split.json')
    cases=[];rows=[];queries={};bycase={};initials={};drifts=[]
    for cid in plan['cases']:
        directory=root/'evaluations'/cid
        if not (directory/'complete.json').exists():continue
        case=read(root/'cases'/cid/'case.json');bycase[cid]=case
        native=read(directory/'initial_metrics.json');initials[cid]=native
        cr=read(directory/'rows.json');rows+=cr
        parent=read_sdf(case['initial_sdf']);parent_smi=identity(parent)
        queries[(cid,parent_smi)]=dict(case_id=cid,target=case['target_id'],series=case['series'],smiles=parent_smi,is_parent=True)
        action_lookup={a['graph']['name']:a for a in case['actions']}
        for row in cr:
            for stage in ('raw','refined','uff_only'):
                result=row.get(stage)
                if not result:continue
                smi=result['graph_check']['actual_smiles']
                queries[(cid,smi)]=dict(case_id=cid,target=case['target_id'],series=case['series'],smiles=smi,is_parent=smi==parent_smi)
                try:
                    mol=read_sdf(result['sdf']);core=case['chemical_protected_maps']
                    target_graph=graph_from_record(action_lookup[row['action']]['graph'])
                    drifts.append(dict(case_id=cid,action=row['action'],mode=row['mode'],draw=row['draw'],stage=stage,
                        **core_drift(parent,target_graph,mol,core),
                        attachment_is_mcs=action_lookup[row['action']]['attachment_map'] in core))
                except Exception as exc:
                    drifts.append(dict(case_id=cid,action=row['action'],mode=row['mode'],draw=row['draw'],stage=stage,error=str(exc)))
        decision=read(root/'decisions'/f'{cid}.json')
        indexed={(r['action'],r['mode'],r['draw']):r for r in cr}
        menu={tuple(o['key']):o for o in decision['options']}
        gains={key:mean_complete([payoff(indexed.get((*key,i))) for i in range(3)]) for key in menu}
        controls={key:mean_complete([payoff(indexed.get((*key,i)),'uff_only') for i in range(3)]) for key in menu}
        def chosen(name,source):
            key=decision[name]
            return 0. if key=='STOP' else source[tuple(key)]
        selected=decision['learned']; key=None if selected=='STOP' else tuple(selected)
        options=decision['options'];option=menu.get(key)
        cost=[];vinas=0;fm_calls=0
        if key:
            for i in range(3):
                r=indexed.get((*key,i))
                if r is None:cost.append(None);continue
                stats=r.get('generation',{}).get('batch_stats')
                fm_calls+=int(stats is not None)
                vinas+=sum(r.get(s) is not None and r[s].get('vina',{}).get('status')=='ok' for s in ('raw','refined'))
                if stats and r.get('raw') and r.get('refined'):
                    cost.append(stats['total_seconds']/stats['batch_size']+r['raw']['measurement_seconds']+
                                r['refined']['measurement_seconds']+r['refined']['uff']['seconds'])
                else:cost.append(None)
        gain=chosen('learned',gains);seconds=mean_complete(cost) if key else 0.
        cases.append(dict(case_id=cid,target=case['target_id'],series=case['series'],initial_pb=native['pb'].get('passes')
                          if not native['pb'].get('measurement_missing') else None,
            candidates=len(menu),selected=selected,learned=gain,property_rule=chosen('property_rule',gains),
            random=mean_complete(list(gains.values())) if gains else 0.,stop=0.,
            learned_uff_only=chosen('learned',controls),
            oracle_observed=max((v for v in gains.values() if v is not None),default=None),
            incomplete_candidate_modes=sum(v is None for v in gains.values()),
            delta_qed=option['delta_qed'] if option else 0.,delta_sa=option['delta_sa'] if option else 0.,
            selected_fm_draws=fm_calls,selected_successful_vina_scores=vinas,
            mean_amortized_seconds=seconds,gain_per_second=gain/seconds if gain is not None and seconds else None))
    save(root/'activity_queries.json',list(queries.values()))
    if queries:
        subprocess.run(['/LocalData1/uslee/envs/molgenbench-eval/bin/python',str(ROOT/'scripts/molgenbench_activity.py'),
                        '--output',str(root)],check=True)
    matches=read(root/'activity_matches.json')['matches'] if queries else []
    active_lookup={(r['case_id'],r['smiles']):r for r in matches}
    stage_results={}
    for stage in ('raw','refined','uff_only'):
        bymode={}
        for mode in ('strict','relaxed'):
            scheduled=[r for r in rows if r['mode']==mode];values=[r[stage] for r in scheduled if r.get(stage)]
            rediscovered=0
            for row in scheduled:
                v=row.get(stage)
                if v:
                    match=active_lookup.get((row['case_id'],v['graph_check']['actual_smiles']),{})
                    rediscovered+=int(match.get('known_active_rediscovered',False) and not match.get('is_parent',False))
            deltas=[v['vina_improvement_vs_initial'] for v in values if v['joint_physical'] is True and v['vina_improvement_vs_initial'] is not None]
            bymode[mode]=dict(scheduled=len(scheduled),available=len(values),
                pb_pass=sum(v['pb'].get('passes',False) and not v['pb']['measurement_missing'] for v in values),
                pb_missing=sum(v['pb']['measurement_missing'] for v in values),
                joint_physical_pass=sum(v['joint_physical'] is True for v in values),
                exact_stereograph_preserved=sum(v['graph_check']['matches'] for v in values),
                held_preserved=sum(v['held_preserved'] for v in values),
                observed_physical_vina_pairs=len(deltas),mean_signed_vina_improvement=float(np.mean(deltas)) if deltas else None,
                improved=sum(d>0 for d in deltas),worsened=sum(d<0 for d in deltas),
                nonparent_active_rediscovery_draws=rediscovered)
        stage_results[stage]=bymode
    policy={}
    for name in ('learned','random','property_rule','stop','learned_uff_only'):
        values=[c[name] for c in cases if c[name] is not None]
        valid=[c[name] for c in cases if c['initial_pb'] is True and c[name] is not None]
        policy[name]=dict(measured_targets=len(values),mean_gain=float(np.mean(values)) if values else None,
            initial_pb_valid_targets=len(valid),mean_gain_pb_valid=float(np.mean(valid)) if valid else None,
            target_bootstrap_95ci=bootstrap(values))
    contrasts={}
    for name in ('random','property_rule','learned_uff_only'):
        values=[c['learned']-c[name] for c in cases if c['learned'] is not None and c[name] is not None]
        contrasts[name]=dict(paired_targets=len(values),mean_difference=float(np.mean(values)) if values else None,
                             target_bootstrap_95ci=bootstrap(values))
    result=dict(time_kst=now(),requested_targets=plan['requested_targets'],locked_targets=len(split['selected']),
        completed_targets=len(cases),prepared_targets=len(plan['cases']),preparation_failures=plan['preparation_failures'],
        scheduled_fm_draws=plan['planned_fm_draws'],evaluated_rows=len(rows),
        generation_status=dict(Counter(r['generation_status'] for r in rows)),
        initial_pb_pass=sum(c['initial_pb'] is True for c in cases),
        initial_active_self_matches=sum(r['known_active_rediscovered'] and r['is_parent'] for r in matches),
        stages=stage_results,policies=policy,learned_minus=contrasts,cases=cases,
        activity_caveat='Nonmatch means unknown activity. Raw draw rediscovery is not successful optimization; must also satisfy graph/physics. Parent identity reported separately.',
        policy_caveat='Scores fixed before FM outcomes. Missing measurements remain null. Preparation failures and missing targets are reported, not replaced. Mean seed payoff, never best-of-three.',
        efficiency_caveat='Per-choice amortized counterfactual cost from exhaustive collection; excludes startup/preparation/initial metrics and is not measured online speedup.')
    save(root/'external_results.json',result);save(root/'core_coordinate_drift.json',drifts)
    if cases:
        with (root/'per_target.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    evidence=read(root/'evidence_complete.json') if (root/'evidence_complete.json').exists() else {}
    lines=['# MolGenBench full-ligand SBMO 외부 평가',f"작성: {result['time_kst']}",'',
        '공식 scaffold-only H2L leaderboard 재현이 아닌, 완전한 초기 ligand를 편집하는 별도 설정이다. LLM 호출과 추가 모델 학습은 없다.',
        f"타깃 잠금 {len(split['selected'])}, 준비 {len(plan['cases'])}, 평가 완료 {len(cases)}. 계획 FM 생성 {plan['planned_fm_draws']}회, 평가 행 {len(rows)}개.",
        '선정 이후 후보 부족·초기화 실패·평가 결측으로 다른 타깃을 보충하지 않았다.', '',
        '| 정책 | 결과가 측정된 타깃 | 평균 수락 Vina 개선 (kcal/mol) |','|---|---:|---:|']
    for name,p in policy.items():lines.append(f"| {name} | {p['measured_targets']} | {p['mean_gain']} |")
    lines+=['','수락 이득은 유효한 생성물에서 Vina가 개선될 때 얻으며, 실패/악화 시 원본을 유지한다. 미측정 affinity를 0으로 채우지 않는다. signed Vina 변화와 QED/SA 변화는 JSON/CSV에 따로 기록했다.',
        f"원본 PB 통과 {result['initial_pb_pass']}/{len(cases)}. 원본의 알려진 활성 자기 일치 {result['initial_active_self_matches']}건은 편집 성공에서 제외한다.",
        '', '## 조건과 한계',
        '- 두 모드 모두 편집된 전체 화학 그래프를 고정한다. strict는 유지 원자 좌표를 고정한다. relaxed는 편집 연결 원자 한 개의 좌표도 풀며, 이 원자가 MCS에 포함될 수 있다. MCS 화학 구조는 양쪽 모두 보호한다.',
        '- UFF-only는 동일한 편집 그래프·ETKDG 초기 좌표·고정 mask에서 FM을 생략한 대조군이다.',
        '- 평가 활성 목록은 생성기와 Router에 전달하지 않았다. 개발 타깃의 구조만 사용해 새 MMP 규칙을 만들었다.',
        '- CrossDocked 공식 타깃 중복 및 로컬 source PDB/정확한 초기 그래프 중복을 점검했다. 유사 서열/동족체 분리와 FM 사전학습 비중복을 입증한 것은 아니다.',
        '- 동결 Router의 과거 MMP 학습 출처 중복은 미확인이다. 새로운 규칙을 정리했다고 기존 가중치의 과거 노출까지 제거되는 것은 아니다.',
        '- PB는 기존 DIRMO dock20 계약, Vina는 score_only이다. 공식 MolGenBench 전체 지표 재현 또는 실험 결합력 검증으로 해석하지 않는다.',
        '- 알려진 활성 분자와 일치하지 않는 생성물의 실험 활성은 미확인이다. 활성값은 endpoint별로 분리하며 원본 자기 일치와 비원본 재발견을 구분한다.',
        '', '## 학습 데이터 검토',
        f"개발 타깃에서 ChEMBL37의 동일 assay 측정으로 확인한 MMP 쌍: {evidence.get('same_assay_mmp_pairs','pending')}개.",
        '이는 MolGenBench 원본 ChEMBL33 assay를 복원했다는 뜻이 아니라, 동일 분자·타깃·endpoint·값과 일치하는 별도 ChEMBL37 근거다. 같은 series라는 이유만으로 assay를 같다고 간주하지 않았다.',
        '후속 학습에는 개발 타깃을 다시 train/validation으로 나누어야 한다. 이 쌍은 화학 편집 활성 변화의 근거이며 FM 모드/좌표 고정 정답은 아니다. 이번에는 재학습하지 않았다.',
        '', '파일: target_split.json, inventory.json, evidence_complete.json, activity_pairs.jsonl, pair_examples.json, external_results.json, per_target.csv, core_coordinate_drift.json, activity_matches.json.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result
