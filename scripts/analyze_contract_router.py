"""Read-only analysis of frozen internal Router test predictions; no retraining."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from router_contract_pilot import now, read, save, sha256_file


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--training',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();root=args.training.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    metrics=read(root/'metrics.json');rows=read(root/'data/rows.json');y=np.load(root/'data/arrays.npz')['y']
    prediction=np.load(root/'predictions.npy');policy=metrics['test_policy']['rows']
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test']);fit=np.array([i for i,r in enumerate(rows) if r['split']!='test'])
    comparison={}
    for j,name in enumerate(('physical','delta_vina','accepted_gain','log_cost')):
        ti=test[np.isfinite(y[test,j])];fi=fit[np.isfinite(y[fit,j])]
        target=y[ti,j].astype(float);pred=prediction[ti,j];median=float(np.median(y[fi,j]));mean=float(y[fi,j].mean())
        d=dict(n=len(ti),fit_median=median,fit_mean=mean,observed_test_mean=float(target.mean()),
            predicted_test_mean=float(pred.mean()),model_mae=float(np.abs(target-pred).mean()),
            fit_median_baseline_mae=float(np.abs(target-median).mean()),zero_baseline_mae=float(np.abs(target).mean()))
        if j==0:d.update(model_brier=float(((target-pred)**2).mean()),
            fit_prevalence_baseline_brier=float(((target-mean)**2).mean()),
            threshold_05_accuracy=float(((pred>=.5)==target).mean()))
        comparison[name]=d
    by=defaultdict(lambda:defaultdict(list))
    for i in test:
        r=rows[i];by[r['case_id']][r['action'],r['mode']].append(i)
    detailed=[];selected_ids=[];within_auc=[];within_rank=[]
    for item in policy:
        cid=item['case_id'];actions=by[cid];eligible=[]
        allids=np.array([i for ids in actions.values() for i in ids if np.isfinite(y[i,0])])
        auc=float(roc_auc_score(y[allids,0],prediction[allids,0])) if len(set(y[allids,0]))>1 else None
        if auc is not None:within_auc.append(auc)
        for key,ids in actions.items():
            if len(ids)==3 and np.isfinite(y[ids,2]).all():
                eligible.append(dict(key=key,ids=ids,gain=float(y[ids,2].mean()),pred=float(prediction[ids,2].mean()),
                    physical=float(y[ids,0].mean()),positive=float((y[ids,2]>0).mean())))
        selected=next(v for v in eligible if list(v['key'])==item['selected']) if item['selected']!='STOP' else None
        corr=float(spearmanr([v['gain'] for v in eligible],[v['pred'] for v in eligible]).statistic) if len(set(v['gain'] for v in eligible))>1 else None
        if corr is not None:within_rank.append(corr)
        ids=selected['ids'] if selected else [];selected_ids+=ids
        base=rows[ids[0]] if ids else None
        detail={**item,'within_root_physical_auc':auc,'within_root_gain_spearman':corr,
            'selected_predicted_gain':selected['pred'] if selected else 0.,
            'selected_positive_seeds':int(sum(y[i,2]>0 for i in ids)),
            'selected_delta_qed':base['delta_qed'] if base else 0.,'selected_delta_sa':base['delta_sa'] if base else 0.,
            'random_physical':float(np.mean([v['physical'] for v in eligible])),
            'random_positive_seed_rate':float(np.mean([v['positive'] for v in eligible]))}
        detailed.append(detail)
    means=metrics['test_policy']['means']
    gains=sorted([r['learned'] for r in policy],reverse=True)
    cmp={}
    for baseline in ('random','property_rule'):
        dif=np.array([r['learned']-r[baseline] for r in policy])
        cmp[baseline]=dict(mean_difference=float(dif.mean()),wins=int((dif>1e-12).sum()),
            ties=int((abs(dif)<=1e-12).sum()),losses=int((dif< -1e-12).sum()))
        rng=np.random.default_rng(20261105)
        # Frozen split has 15 distinct test groups, one root per group here.
        assert len({r['group'] for r in policy})==len(policy)
        boot=rng.choice(dif,size=(10000,len(dif)),replace=True).mean(1)
        cmp[baseline]['diagnostic_bootstrap_95ci']=np.quantile(boot,[.025,.975]).tolist()
    cov=metrics['test_policy']['coverage'];gi=test[np.isfinite(y[test,2])]
    result=dict(time_kst=now(),test_roots=len(policy),test_slots=len(test),
        generation_status=dict(Counter(rows[i]['generation_status'] for i in test)),
        common_menu_candidate_modes=cov['total_candidate_modes']-cov['incomplete_candidate_modes_excluded'],
        all_candidate_modes=cov['total_candidate_modes'],head_comparisons=comparison,policy_means=means,
        policy_comparisons=cmp,original_saved_random_ci=metrics['test_policy']['learned_minus_random_group_bootstrap_95ci'],
        roots_with_selected_improvement=sum(r['learned']>0 for r in policy),roots_with_oracle_improvement=sum(r['oracle']>0 for r in policy),
        selected_draws=len(selected_ids),selected_physical_success=int(sum(y[i,0]==1 for i in selected_ids)),
        selected_positive_draws=int(sum(y[i,2]>0 for i in selected_ids)),
        positive_gain_labels=int((y[gi,2]>0).sum()),observed_gain_labels=len(gi),
        top_two_share=float(sum(gains[:2])/sum(gains)),oracle_gain_fraction=means['learned']/means['oracle'],
        average_regret=means['oracle']-means['learned'],
        within_root_physical_auc_mean=float(np.mean(within_auc)),within_root_gain_spearman_mean=float(np.mean(within_rank)),
        within_root_rank_defined_roots=len(within_rank),
        random_common_menu_physical=float(np.mean([r['random_physical'] for r in detailed])),
        improving_selected_roots_qed_decrease=sum(r['learned']>0 and r['selected_delta_qed']<0 for r in detailed),
        improving_selected_roots_sa_increase=sum(r['learned']>0 and r['selected_delta_sa']>0 for r in detailed),
        cases=detailed,source_hashes={n:sha256_file(root/n) for n in ('metrics.json','data/rows.json','predictions.npy','data/arrays.npz')},
        analyst_script_sha256=sha256_file(__file__))
    save(out/'analysis.json',result)
    with (out/'per_case.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(detailed[0]));writer.writeheader();writer.writerows(detailed)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(13,9),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1,1.25])
    ax=fig.add_subplot(gs[0,0]);labels=['STOP','Random','QED rule','Router','Oracle']
    values=[means[k] for k in ('stop','random','property_rule','learned','oracle')]
    bars=ax.bar(labels,values,color=['#aab2bd','#8b9fb3','#dfaa48','#266d9b','#61a58a'])
    ax.bar_label(bars,fmt='%.3f',padding=3);ax.set_ylim(0,.24);ax.set_ylabel('Accepted Vina gain (kcal/mol)');ax.set_title('A. Mean gain per draw (15 roots)');ax.tick_params(axis='x',rotation=25)
    ax=fig.add_subplot(gs[0,1]);d=comparison['delta_vina'];b=ax.bar(['Fit median','Router'],[d['fit_median_baseline_mae'],d['model_mae']],color=['#8b9fb3','#266d9b']);ax.bar_label(b,fmt='%.3f',padding=3);ax.set_ylim(0,1.28);ax.set_ylabel('MAE (kcal/mol; lower is better)');ax.set_title('B. Feasible Vina-change prediction')
    ax=fig.add_subplot(gs[0,2]);d=comparison['accepted_gain'];b=ax.bar(['Always zero','Router'],[d['zero_baseline_mae'],d['model_mae']],color=['#8b9fb3','#266d9b']);ax.bar_label(b,fmt='%.3f',padding=3);ax.set_ylim(0,.075);ax.set_ylabel('MAE (kcal/mol; lower is better)');ax.set_title('C. Accepted-gain prediction')
    ax=fig.add_subplot(gs[1,:]);xx=np.arange(len(policy));width=.25
    for shift,key,label,color in [(-width,'random','Random','#8b9fb3'),(0,'learned','Router','#266d9b'),(width,'oracle','Observed menu oracle','#61a58a')]:
        ax.bar(xx+shift,[r[key] for r in policy],width,label=label,color=color)
    ax.set_xticks(xx,[str(i+1).zfill(2) for i in xx]);ax.set_xlabel('Test complex (mapping in per_case.csv)');ax.set_ylabel('Accepted Vina gain (kcal/mol)');ax.set_title('D. Per-root 3-seed means: improvement in 4/15 roots; top two provide 86.5% of Router gain');ax.legend(frameon=False)
    fig.suptitle('Action Router: internal 15-complex holdout\nReject invalid / non-improving outputs; complete measured menus only',fontsize=14)
    fig.savefig(out/'test_analysis.png',dpi=180);fig.savefig(out/'test_analysis.pdf');plt.close(fig)
    report=f'''# Action Router 100-root 실험: 고정 test 결과 분석

작성: {now()}. 모델이나 라벨을 수정하지 않고 저장된 test 예측을 재분석했다. 초기 데이터는 CrossDocked train 출처이며 70/15/15 내부 분할, 최종 모델은 validation으로 설정을 선택한 후 85 roots에서 refit했다. 공식 CrossDocked test 결과가 아니다.

## 판단

물리적으로 유효한 결과를 구분하는 신호는 학습했다. 하지만 실제 개선 후보 선택 능력은 아직 제한적이다. 랜덤 대비 평균 개선은 작고 불확실하며, 단순 QED 규칙과 거의 같다. 전체 파이프라인의 성능 향상 또는 최소 편집 최적화를 입증한 결과가 아니다.

## 평가량 정의

한 초기 상태에서 후보(site+target graph+mode) 하나를 선택한 뒤 세 seed의 실제 결과를 평균한다. 각 seed는 graph/stereo·held 좌표·PB를 통과하고 Vina가 개선될 때에만 `initial Vina - final Vina`를 얻는다. 실패 또는 악화는 부모를 유지하므로 **정책 이득**이 0이다. affinity 회귀의 결측을 0으로 바꾼 것이 아니다. best-of-three 성능도 아니다. Oracle도 같은 메뉴의 후보별 3-seed 평균 중 최대를 사용하며, 사후 관측 상한 참조일 뿐 배포 가능한 모델이 아니다.

| 방법 | 평균 수락 Vina 개선, kcal/mol |
|---|---:|
| STOP | 0 |
| Random | {means['random']:.5f} |
| QED 변화 규칙 | {means['property_rule']:.5f} |
| Learned Router | {means['learned']:.5f} |
| 관측 메뉴 Oracle | {means['oracle']:.5f} |

랜덤 대비 차이 {cmp['random']['mean_difference']:.5f}; 원래 저장된 group bootstrap 95% 구간 {result['original_saved_random_ci']}. 0을 포함한다. 상대 증가율 약 {(means['learned']/means['random']-1)*100:.1f}%만 강조하면 절대 효과와 불확실성을 숨기게 된다. 규칙 대비 차이는 {cmp['property_rule']['mean_difference']:.5f}다. 추가 진단 bootstrap 설정/결과는 analysis.json에 있으며 원래 평가를 덮어쓰지 않았다.

## 사례별 결과

- Router 개선: {result['roots_with_selected_improvement']}/15 roots. 관측 메뉴에는 개선 가능한 후보가 {result['roots_with_oracle_improvement']}/15 roots에 있었다.
- 랜덤의 기대 이득과 root별 비교: {cmp['random']['wins']}승/{cmp['random']['ties']}동률/{cmp['random']['losses']}패. 규칙 대비 {cmp['property_rule']['wins']}승/{cmp['property_rule']['ties']}동률/{cmp['property_rule']['losses']}패.
- 두 root의 기여가 전체 Router 이득의 {result['top_two_share']*100:.1f}%. 전체 root에 고르게 개선되는 모델은 아니다.
- 선택한 45 draws 중 물리 통과 {result['selected_physical_success']}, 실제 양의 개선 {result['selected_positive_draws']}. 물리 통과와 Vina 개선은 다른 사건이다.
- 같은 공통 메뉴의 random 기대 물리 통과율 {result['random_common_menu_physical']*100:.1f}%, Router 80.0%.
- Oracle 이득의 {result['oracle_gain_fraction']*100:.1f}%를 회수. 평균 관측 regret {result['average_regret']:.5f}.
- STOP 선택 0/15. 개선 가능한 메뉴가 전혀 없었던 root에서도 실행했다. 비용을 선택 점수에 반영하지 않았으므로 STOP의 적절한 비용-이득 판단은 입증되지 않았다.

## 예측 head

- Physical AUROC 0.7984, AP 0.8829 (test 양성 비율 0.6827), Brier 0.1839. fit 부분의 일정 성공률 예측 Brier {comparison['physical']['fit_prevalence_baseline_brier']:.4f}보다 낫다. AUROC는 성공/실패 결과를 점수로 구분하는 순위 지표이며 실제 생성 성공률이나 정확도 79.8%라는 뜻이 아니다.
- 동일 root 내부 AUROC 단순 평균 {result['within_root_physical_auc_mean']:.4f}. pooled 결과에 target별 난이도 차이도 포함된다.
- 유효 결과 ΔVina MAE 0.9831, fit-median baseline {comparison['delta_vina']['fit_median_baseline_mae']:.4f}. 약 1 kcal/mol 예측 오차로 작은 개선을 정밀 구분하는 데 한계가 있다.
- accepted-gain MAE 0.0591, 항상 0 baseline {comparison['accepted_gain']['zero_baseline_mae']:.4f}. 개선 라벨 {result['positive_gain_labels']}/{result['observed_gain_labels']}로 희소하다. 작은 MAE 자체가 선택 능력을 보장하지 않으며, 0 baseline이 MAE에서 낫다고 해서 0 예측을 유용한 정책이라고 부를 수도 없다.
- root별 후보 gain 순위 Spearman 평균 {result['within_root_gain_spearman_mean']:.4f}, 변동이 있어 계산 가능한 {result['within_root_rank_defined_roots']} roots. 개선량을 정확히 순위화하는 능력은 제한적이다.
- cost head는 log1p(amortized seconds) MAE 0.1712, fit-median baseline {comparison['log_cost']['fit_median_baseline_mae']:.4f}. 현재 정책은 비용을 감점하지 않으므로 실제 비용 절감의 증거로 읽지 않는다.

## 물성 및 평가 범위

Vina 개선이 나온 선택 4개 그래프 중 QED 감소는 {result['improving_selected_roots_qed_decrease']}개, SA 증가(통상 합성 용이성 관점에서 불리한 방향)는 {result['improving_selected_roots_sa_increase']}개다. 이번 선택 목표에 QED/SA 제약을 넣지 않았으므로 다목적 성질 개선 성공이라고 부를 수 없다. per_case.csv에 각 변화량을 저장했다.

Test slots 594개 중 physical 측정 479개, feasible Vina 회귀 317개다. 초기화 실패 114개와 raw contract 실패 11개를 성공으로 처리하지 않는다. 정책 비교는 198 candidate-mode 중 완전한 세 seed 결과가 있는 153개(77.3%)의 공통 메뉴에 한정한다. 45개 제외를 숨기면 배포 범위 성능을 과대해석하게 된다. pose-only baseline은 14 roots에서만 존재하므로 위 15-root 표와 직접 비교하지 않았다.

이것은 15개 초기 복합체의 오프라인 평가다. 479행을 479개 독립 단백질로 해석하면 안 된다. 실제 Router 선택을 FM에 전달하는 새 실행이나 여러 round 최적화는 아직 수행하지 않았다. target-family/exact scaffold grouping은 full sequence/pocket clustering을 대체하지 않으며 MMP source-assay provenance 및 FM pretraining 중복도 아직 불명확하다.

## 다음 단계

1. 물리 feasibility 예측을 기준선으로 유지하고 후보 ranking/accepted-gain 목표를 보완한다. 같은 root 내 후보 비교 loss 또는 개선 사건과 개선 크기를 분리하는 모델을 검토한다. 원인 확정이 아닌 후속 실험 가설이다.
2. validation에서 목적과 선택 규칙을 동결한다. Vina뿐 아니라 QED/SA 허용 범위, 실패·비용에 대한 STOP 조건을 명시한다. 현재 test를 보고 threshold를 맞춘 뒤 같은 test에서 향상을 주장하지 않는다.
3. 초기화 실패와 미완전 메뉴 coverage를 개선하고, 새 train roots를 확장한다. 단순 epoch 증가보다 개선/비개선 후보가 함께 있는 다양한 초기 상태를 확보한다.
4. 수정 모델을 validation에서 고른 뒤 새로 잠근 holdout에 실제 선택 후보를 실행하는 전향적 한-step 평가를 한다. 그 다음 닫힌 루프를 검증한다.

파일: analysis.json(추가 집계), per_case.csv(15개 root별 선택), test_analysis.png/PDF(그림). 원본 metrics.json과 모델은 변경하지 않았다.
'''
    (out/'REPORT.md').write_text(report)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','source_hashes')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
