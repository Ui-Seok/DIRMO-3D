"""Render the predeclared ranking ablation and stored paired UFF comparison."""
import argparse
from pathlib import Path
import numpy as np
from router_contract_pilot import read,now


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();r=a.run.resolve()
    d=read(r/'results.json');u=read(r/'uff_grouped_analysis.json')['results']['all'];design=read(r/'design.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    b=d['results']['0.0']['policy']['means']
    vals=[b['random'],b['property_rule'],b['learned'],d['results']['0.1']['policy']['means']['learned'],d['results']['0.3']['policy']['means']['learned']]
    bars=ax[0].bar(['Random','QED rule','Base','Rank 0.1','Rank 0.3'],vals,color=['#adbac7','#d7a04b','#285f8c','#4b94b4','#86bac6']);ax[0].bar_label(bars,fmt='%.3f',padding=3);ax[0].set_ylim(0,.14);ax[0].tick_params(axis='x',rotation=30);ax[0].set_ylabel('Mean accepted Vina gain (kcal/mol)');ax[0].set_title('Validation: 15 roots, same candidate menus')
    base={x['case_id']:x for x in d['results']['0.0']['policy']['rows']};rank={x['case_id']:x for x in d['results']['0.1']['policy']['rows']};ids=sorted(base)
    dif=[rank[c]['learned']-base[c]['learned'] for c in ids];ax[1].bar(range(15),dif,color=['#56a486' if v>0 else '#d17a68' for v in dif]);ax[1].axhline(0,color='gray',lw=.7);ax[1].set_xticks(range(15),[str(i+1) for i in range(15)],fontsize=8);ax[1].set_ylabel('Rank 0.1 minus base (kcal/mol)');ax[1].set_xlabel('Validation root (sorted case ID)');ax[1].set_title('Primary contrast: 1 win / 12 ties / 2 losses')
    pv=[u['uff_physical_pass']/u['paired_draws']*100,u['fm_physical_pass']/u['paired_draws']*100];bars=ax[2].bar(['UFF only','FM + UFF'],pv,color=['#adbac7','#285f8c']);ax[2].bar_label(bars,fmt='%.1f%%',padding=3);ax[2].set_ylim(0,100);ax[2].set_ylabel('Joint physical pass (%)');ax[2].set_title('Paired controls: 192 draws / 18 roots')
    fig.suptitle('GPU 3 Router ranking-loss ablation | saved frozen-FM outcomes, no LLM',fontsize=13)
    fig.savefig(r/'comparison.png',dpi=180);fig.savefig(r/'comparison.pdf');plt.close(fig)
    table=[]
    for lam in ['0.0','0.1','0.3']:
        v=d['results'][lam];m=v['policy']['means'];table.append(f"| {lam} | {m['learned']:.5f} | {m['learned_physical']*100:.1f}% | {sum(x['learned']>0 for x in v['policy']['rows'])}/15 | {', '.join(str(s['epoch']) for s in v['seeds'])} |")
    cp=d['comparisons']['0.1'];cs=d['comparisons']['0.3']
    report=f'''# GPU 3: Router ranking-loss 비교 결과

작성: {now()}. 실험 종료: {read(r/'status.json')['time_kst']}. 모든 모델은 GPU UUID `{d['actual_gpu_uuid']}`에서 CUDA로 학습했다. 9개 fit 및 정책 평가 시간 {d['elapsed_seconds']:.1f}초, peak allocated {d['peak_memory_bytes']/1024**2:.1f} MiB. 작은 MLP의 full-batch 학습이므로 FM 생성보다 훨씬 짧다.

## 실험 범위와 모델

모델 구성은 고정된 FLOWR.root FM과 Action Router다. 이번에는 저장된 FM 결과를 재사용하여 **Router만 새로 학습**했다. LLM 호출, FM 가중치 학습, 새 ligand 생성은 없다. UFF-only 비교도 이미 저장된 결과를 읽은 것이다. 기존 학습 70 roots와 validation 15 roots만 사용했다. 기존 test 15개는 학습·scoring·통계·모델 선택에서 제외했고 test 예측 파일을 만들지 않았다. 배열 컨테이너는 기존 파일을 읽지만 split 필터로 test 행을 즉시 제거한다.

## 사전 고정한 설계

- 기존 385 features, MLP 128→64→4, LR 3e-4, AdamW weight decay .01, 같은 3개 초기화 seed.
- 최대 200 epochs, 최소 40 epochs 이후 patience 25. 모든 설정에서 동일한 **기존 base validation loss**로 checkpoint를 선택했다. 서로 다른 augmented loss를 직접 비교하지 않는다.
- λ=0: 기존 masked/root-weighted BCE+SmoothL1. λ=.1: 주 비교. λ=.3: 민감도 확인. **λ는 ranking loss의 가중치이며 FM guidance strength가 아니다.**
- 같은 부모 상태의 완전한 3-seed 후보 결과만 비교한다. 각 후보의 3회 결과를 복원추출한 평균 27개씩을 정확히 열거해 soft preference를 만들고, 우열이 불확실할수록 가중치를 낮춘다. 정확한 tie는 제외한다. 세 반복이 9개나 729개의 독립 관측으로 늘어나는 것이 아니다. bootstrap soft target도 보정된 성공 확률/통계적 확신이 아니다.
- train의 informative pair 1,195개, 기여 root 46개. root별 pair 가중치 합을 정규화하여 후보가 많은 root가 loss를 지배하지 않도록 했다. pair에는 validation/test 행이 없다.
- QED/SA 제약, 비용 penalty, STOP threshold는 이번에 바꾸지 않았다. 학습 목표 추가의 효과만 비교한다.

## Validation 결과

| Ranking λ | 평균 수락 Vina 개선 kcal/mol | 선택 결과 물리 통과 | 개선을 얻은 root | seed별 선택 epoch |
|---|---:|---:|---:|---|
{chr(10).join(table)}

동일 메뉴의 random 평균 {b['random']:.5f}, QED 규칙 {b['property_rule']:.5f}, 관측 oracle {b['oracle']:.5f}다. 세 seed 결과 평균이며 best seed 선택이 아니다. 유효하지 않거나 Vina가 악화된 결과는 거절하고 부모를 유지해 정책 이득 0을 얻는다. affinity의 결측값을 0으로 넣은 것은 아니다.

주 비교 λ=.1은 기존 대비 **{cp['mean']:.5f}**, paired group bootstrap 95% 구간 **{cp['ci95']}**다. root별 {cp['wins']}승/{cp['ties']}동률/{cp['losses']}패이며, 개선 root 수는 7개로 같다. λ=.3은 기존 대비 {cs['mean']:.5f}, 구간 {cs['ci95']}다. 두 설정 모두 개선 근거가 없다. λ=.1에서 physical pass는 약 2.2%p 높지만 Vina 선택 이득은 높지 않다.

각 rank 설정은 개별 초기화 seed 3개 중 2개에서 baseline seed보다 나았지만, 3개 모델의 예측을 평균해 후보를 선택한 ensemble 결과는 더 낮았다. 후보 argmax는 비선형이므로 개별 정책 성능의 평균과 ensemble 정책 성능이 같지 않다.

이전 test 평균 0.0672와 이번 baseline validation 0.1114를 직접 비교하면 안 된다. 다른 15 roots이고, 이번 모델은 validation을 학습에 포함한 최종 refit 모델이 아닌 train 70개로 학습한 모델이다. 이번 λ=0의 validation head 지표와 선택 epoch는 원래 동일 설정 trial을 재현한다.

## 확대 기준과 결정

학습 전에 design.json에 고정한 조건: 주 비교 λ=.1의 validation 이득이 baseline·QED rule·random보다 높고, baseline 대비 bootstrap 구간 하한 >0, 동일 seed 비교 2/3 이상 승리, 물리 통과율 하락 ≤5%p. λ=.3의 유리한 결과를 골라 확대를 정당화하지 않는다.

결과: **기준 미통과**. 새 50-root FM 확인 실험이나 대규모 수집으로 확대하지 않았다. 이 제한된 구조·데이터·loss·checkpoint 조건에서 효과가 확인되지 않았다는 뜻이며 모든 ranking 방법이 효과 없다는 증명은 아니다.

## FM+UFF 대 UFF-only 저장 대조

Train/validation의 사전에 지정한 매 다섯 번째 root, 첫 화학 편집과 pose control을 사용한다. 동일한 입력 SDF와 고정 mask에서 FM+UFF 또는 UFF-only를 비교한다. 물리 평가가 양쪽 모두 있는 18 roots/192 draws에서:

- UFF-only: **{u['uff_physical_pass']}/{u['paired_draws']} = {u['uff_physical_pass']/u['paired_draws']*100:.1f}%** 물리 통과.
- FM+UFF: **{u['fm_physical_pass']}/{u['paired_draws']} = {u['fm_physical_pass']/u['paired_draws']*100:.1f}%** 물리 통과.
- PB-valid parent와 이득 측정이 가능한 17 roots/182 draws에서 root 평균 Vina 정책 이득 차이(FM+UFF − UFF-only)는 **{u['gain_difference_root_mean']:.5f} kcal/mol**.
- group bootstrap 95% 구간 **{u['gain_difference_group_bootstrap95ci']}**로 0을 포함한다.

이 부분 표본에서는 FM을 추가했을 때 물리 통과율이 높았으나, Vina 개선 이득까지 확실하다고 결론 내리기 어렵다. strict/relaxed 부분 분석은 탐색적이며 다중 비교 보정이 없다. 전체 100 roots 또는 임의 pocket, native SBDD mode에 일반화하지 않는다. 두 팔 모두 관측된 결과에 대한 paired 분석이므로 탈락률도 원본 rows에서 확인해야 한다.

## 후속 판단

기존 회귀 Router를 유지하고, 현재 rank=.1/.3을 더 좋은 모델로 교체하지 않는다. 다음 우선 후보는 3-seed 라벨의 안정성 진단이다. Train/validation의 후보 우열이 seed에 따라 흔들리는 상태를 정해 반복 생성을 늘리고, 평균 개선과 우열이 안정되는지를 확인하는 편이 근거가 있다. Validation에서 효과가 검증되기 전 test 재평가나 대규모 수집을 반복하지 않는다. 추가 실험은 이번에 실행하지 않았다.

## 파일

- `design.json`: 결과를 보기 전에 저장한 설계·확대 조건·hash.
- `pair_audit.json`: pair 수·root별 기여·split 검증.
- `models/`: λ×seed별 9개 모델과 epoch history. `code.py`: 실행 코드 사본.
- `results.json`: validation head·정책·seed별 결과 및 대비 통계.
- `uff_paired_controls.json`, `uff_grouped_analysis.json`: 대응 UFF 대조와 group bootstrap.
- `comparison.png`, `comparison.pdf`: 결과 그림.
- 실행 코드: `/home/uslee/code/DIRMO-3D/scripts/router_ranking_ablation.py`.
- 보고서 코드: `/home/uslee/code/DIRMO-3D/scripts/report_router_ranking.py`.

기존 모델·데이터·test 결과를 덮어쓰지 않았다. 이 모델들은 오프라인 후보 scorer이며 전체 optimize 파이프라인에 자동 배포하지 않았다.
'''
    (r/'REPORT.md').write_text(report)
    print(str(r/'REPORT.md'))


if __name__=='__main__':main()
