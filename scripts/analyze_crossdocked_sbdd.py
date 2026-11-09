"""Analyze completed raw SBDD draws without running new generation or scoring."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
from rdkit import Chem
from dirmo.evaluation.crossdocked_sbdd import canonical, load_mol
from dirmo.io import sha256_file, write_json


def cluster_interval(values, clusters):
    """Bootstrap source pocket groups; every case keeps equal point weight."""
    frame = pd.DataFrame({'value': values, 'cluster': clusters}).dropna()
    grouped = frame.groupby('cluster').value.agg(['sum', 'count'])
    rng = np.random.default_rng(20261020)
    draws = rng.integers(0, len(grouped), size=(20000, len(grouped)))
    means = grouped['sum'].to_numpy()[draws].sum(axis=1) / grouped['count'].to_numpy()[draws].sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def comparison(left, right, clusters):
    values = np.asarray(right) - np.asarray(left)
    return dict(n=len(values), clusters=len(set(clusters)), reference_mean=float(np.mean(left)),
                generated_mean=float(np.mean(right)), mean_delta=float(values.mean()),
                median_delta=float(np.median(values)), mean_delta_cluster_bootstrap95=cluster_interval(values, clusters),
                lower=int((values < 0).sum()), higher=int((values > 0).sum()), equal=int((values == 0).sum()))


def missing_category(result):
    reason = str(result.get('reason', result.get('error', 'unknown')))
    if 'TimeoutExpired' in reason or 'exceeded 600' in reason:
        return 'timeout_600_seconds'
    if 'fragments' in reason:
        return 'disconnected_ligand'
    if 'non finite charge' in reason:
        return 'ligand_charge_nonfinite'
    if 'None type' in reason:
        return 'unsupported_atom_type'
    if 'parameters unavailable' in reason:
        return 'force_field_parameters_unavailable'
    return reason[:200]


def plot_results(out, summary, result, paired, generated, reference):
    os.environ.setdefault('MPLCONFIGDIR', str(out / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 11, 'figure.dpi': 150, 'pdf.fonttype': 42})
    blue, orange = '#3573A1', '#D88734'
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3), layout='constrained')
    ax = axes[0, 0]
    vals = [96., 96.82, 100 * summary['generated']['pb_pass_per_rdkit_valid']]
    bars = ax.bar(['Reference\n100 ligands', 'Generated\nall 10,000 attempts', 'Generated\n9,883 RDKit-valid'], vals,
                  color=[orange, blue, '#6C9FBD'], width=.65)
    ax.bar_label(bars, labels=[f'{value:.2f}%' for value in vals], padding=3)
    ax.set_ylim(0, 110); ax.set_ylabel('PoseBusters pass (%)'); ax.set_title('A. Physical validity: keep denominators explicit')
    ax = axes[0, 1]
    modes = ['score', 'minimize', 'dock']; x = np.arange(3); width = .34
    native = [result['paired_vina'][key]['reference_mean'] for key in modes]
    output = [result['paired_vina'][key]['generated_mean'] for key in modes]
    ax.bar(x - width / 2, native, width, color=orange, label='Reference')
    ax.bar(x + width / 2, output, width, color=blue, label='Generated mean per pocket')
    ax.set_xticks(x, ['Original pose', 'Local minimize', 'Redock'])
    ax.set_ylabel('Vina (kcal/mol; lower is better)'); ax.set_title('B. Same 97 pockets, equal pocket weight')
    ax.legend(fontsize=8, loc='lower left')
    for index, (prop, label) in enumerate([('qed', 'QED (higher is better)'), ('sa_raw_1_to_10', 'SA, 1-10 (lower is easier)')]):
        ax = axes[1, index]
        values = [result['paired_properties'][prop]['reference_mean'], result['paired_properties'][prop]['generated_mean']]
        bars = ax.bar(['Reference', 'Generated mean\nper pocket'], values, color=[orange, blue], width=.55)
        ax.bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=4)
        ax.set_ylim(0, 1 if prop == 'qed' else 5.0); ax.set_ylabel(label)
        ax.set_title(('C. QED: small positive mean shift' if prop == 'qed' else 'D. Synthetic accessibility gets worse'))
        estimate = result['paired_properties'][prop]
        low, high = estimate['mean_delta_cluster_bootstrap95']
        ax.text(.5, .94, f"Paired delta {estimate['mean_delta']:+.3f}; 95% CI [{low:+.3f}, {high:+.3f}]",
                transform=ax.transAxes, ha='center', va='top', fontsize=8)
    fig.suptitle('FLOWR.root SPINDR base on CrossDocked2020: 100 pockets x 100 raw draws', fontsize=13)
    fig.savefig(out / 'overview.png', bbox_inches='tight'); fig.savefig(out / 'overview.pdf', bbox_inches='tight'); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    ax = axes[0]
    rows = paired[paired.metric == 'score']
    ax.scatter(rows.reference, rows.generated, c=np.where(rows.delta < 0, blue, orange), s=24, alpha=.8)
    limits = [min(rows.reference.min(), rows.generated.min()) - .4, max(rows.reference.max(), rows.generated.max()) + .4]
    ax.plot(limits, limits, color='gray', linestyle='--', linewidth=1)
    ax.set(xlim=limits, ylim=limits, xlabel='Reference Vina score', ylabel='Mean generated Vina score',
           title='Same-pocket scores: 52/97 improve')
    ax = axes[1]
    for data, color, label in [(reference.strain_kcal_mol.dropna(), orange, 'Reference (100)'),
                               (generated.strain_kcal_mol.dropna(), blue, 'Generated (9,856)')]:
        data = np.sort(data.to_numpy())
        ax.step(data, np.arange(1, len(data) + 1) / len(data), where='post', color=color, label=label)
    ax.set_xscale('symlog', linthresh=1)
    ax.set(xlabel='MMFF94s strain (kcal/mol; symlog)', ylabel='Cumulative fraction', title='Lower median, but a severe upper tail')
    ax.legend(fontsize=8); ax.grid(alpha=.15)
    ax = axes[2]
    reasons = result['generated_pb_failed_checks']
    selected = sorted(reasons, key=reasons.get, reverse=True)[:6]
    labels = {'internal_energy': 'Internal energy', 'double_bond_flatness': 'Double-bond flatness',
              'internal_steric_clash': 'Internal clash', 'all_atoms_connected': 'Disconnected graph',
              'bond_angles': 'Bond angles', 'bond_lengths': 'Bond lengths'}
    ax.barh([labels.get(key, key) for key in selected][::-1], [reasons[key] for key in selected][::-1], color=blue)
    ax.set(xlabel='Failing outputs (checks can overlap)', title='What still fails PB?')
    fig.savefig(out / 'diagnostics.png', bbox_inches='tight'); fig.savefig(out / 'diagnostics.pdf', bbox_inches='tight'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'analyze_crossdocked_sbdd.py').write_text(Path(__file__).read_text())
    summary = json.loads((root / 'summary.json').read_text())
    plan = json.loads((root / 'study.json').read_text())
    assert summary['status'] == 'complete' and summary['evaluated_draws'] == 10000
    frame = pd.read_csv(root / 'molecule_metrics.csv')
    generated, reference = frame[frame.draw >= 0].copy(), frame[frame.draw < 0].copy()
    assert len(generated) == 10000 and len(reference) == 100
    assert not generated.duplicated(['case_id', 'draw']).any()
    cases = {c['case_id']: c for c in plan['cases']}
    native = {cid: json.loads((root / 'cases' / cid / 'reference_evaluation.json').read_text()) for cid in cases}
    clusters = {cid: Path(c['source_protein']).parent.name for cid, c in cases.items()}
    property_pairs, vina_pairs, all_pairs, per_case = {}, {}, [], []
    for metric in ('score', 'minimize', 'dock'):
        rows = []
        improved, scored, pb_improved, pb_scored = 0, 0, 0, 0
        for c in summary['per_case']:
            cid = c['case_id']; a = native[cid].get('vina', {}).get(metric); b = c['vina'][metric]['mean']
            if a is None or b is None:
                continue
            rows.append(dict(case_id=cid, cluster=clusters[cid], metric=metric,
                             reference=a, generated=b, delta=b - a, generated_n=c['vina'][metric]['n']))
            improved += c['high_affinity'][metric]['improved']; scored += c['high_affinity'][metric]['n']
        data = pd.DataFrame(rows)
        vina_pairs[metric] = comparison(data.reference, data.generated, data.cluster)
        vina_pairs[metric].update(improved_ligands=improved, scored_ligands=scored,
                                  ligand_improved_fraction=improved / scored,
                                  macro_improved_fraction=float(np.mean([c['high_affinity'][metric]['fraction'] for c in summary['per_case']
                                                                       if c.get('high_affinity', {}).get(metric, {}).get('fraction') is not None])))
        all_pairs.extend(rows)
    for prop in ('qed', 'sa_raw_1_to_10', 'logp', 'molecular_weight', 'lipinski_pass'):
        rows = [(native[c['case_id']]['properties'][prop], c['properties'][prop]['mean'], clusters[c['case_id']]) for c in summary['per_case']]
        property_pairs[prop] = comparison([x[0] for x in rows], [x[1] for x in rows], [x[2] for x in rows])
    pb_pairs = {}
    for denom in ('attempt', 'rdkit_valid'):
        a = [int(native[c['case_id']]['pb']['passes']) for c in summary['per_case']]
        b = [c['pb_pass_per_' + denom] for c in summary['per_case']]
        pb_pairs[denom] = comparison(a, b, [clusters[c['case_id']] for c in summary['per_case']])
    failed_checks, vina_missing, strain_missing = Counter(), Counter(), Counter()
    pb_missing, strain_outliers, samples = [], [], []
    pb_vina = {key: dict(n=0, improved=0, attempted_in_paired_pockets=0) for key in ('score', 'minimize', 'dock')}
    exact_reference = 0
    for cid, c in cases.items():
        expected = canonical(load_mol(c['native_sdf']))
        missing_count = 0
        for draw in range(plan['settings']['draws']):
            path = root / 'cases' / cid / 'generated' / f'{draw:03d}' / 'evaluation.json'
            row = json.loads(path.read_text()); pb = row.get('pb', {}); vina = row.get('vina', {})
            failed_checks.update(pb.get('failed_checks', []))
            if pb.get('missing'):
                pb_missing.append(dict(case_id=cid, draw=draw, reason=pb.get('error')))
            if row.get('rdkit_valid') and vina.get('status') != 'ok':
                missing_count += 1; vina_missing[missing_category(vina)] += 1
            if row.get('rdkit_valid') and 'strain_kcal_mol' not in row.get('strain', {}):
                strain_missing[missing_category(row.get('strain', {}))] += 1
            value = row.get('strain', {}).get('strain_kcal_mol')
            if value is not None and value > 1000:
                strain_outliers.append(dict(case_id=cid, draw=draw, strain=value,
                                            pb_pass=pb.get('passes'), pb_failed_checks=pb.get('failed_checks')))
            if row.get('smiles') == expected:
                exact_reference += 1
            for metric in pb_vina:
                ref = native[cid].get('vina', {}).get(metric)
                if ref is not None:
                    pb_vina[metric]['attempted_in_paired_pockets'] += 1
                    if pb.get('passes') and metric in vina:
                        pb_vina[metric]['n'] += 1
                        pb_vina[metric]['improved'] += vina[metric] < ref
            samples.append(dict(case_id=cid, draw=draw, rdkit_valid=row['rdkit_valid'],
                                pb_pass=pb.get('passes', False), pb_missing=pb.get('missing', False),
                                vina_status=vina.get('status'), vina_missing_category=missing_category(vina) if vina.get('status') == 'missing' else None))
        case_summary = next(x for x in summary['per_case'] if x['case_id'] == cid)
        per_case.append(dict(case_id=cid, pocket_group=clusters[cid], atoms=c['atoms'],
                             rdkit_valid=case_summary['rdkit_valid'], pb_pass=case_summary['pb_pass'],
                             native_pb_pass=native[cid]['pb']['passes'], vina_missing=missing_count,
                             unique_fraction=case_summary['unique_smiles'] / case_summary['rdkit_valid'],
                             diversity=case_summary['intra_pocket_diversity']['mean']))
    result = dict(created_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
                  experiment=root.name, completed_evaluation_cases=100, scheduled=10000,
                  paired_vina=vina_pairs, paired_properties=property_pairs, paired_pb=pb_pairs,
                  bootstrap='20,000 resamples of 93 source pocket-directory groups; equal test-case weights, exploratory rather than protein-sequence-independent generalization',
                  generated_pb_failed_checks=dict(failed_checks), pb_missing=pb_missing,
                  generated_vina_missing=dict(vina_missing), generated_strain_missing=dict(strain_missing),
                  strain_over_1000=sorted(strain_outliers, key=lambda x: x['strain'], reverse=True),
                  strain_quantiles={str(q): float(generated.strain_kcal_mol.quantile(q)) for q in (.5, .9, .95, .99, 1.)},
                  exact_reference_smiles_outputs=exact_reference, pb_valid_vina_comparisons=pb_vina,
                  macro_unique_fraction=float(np.mean([x['unique_fraction'] for x in per_case])),
                  macro_diversity=float(np.mean([x['diversity'] for x in per_case])),
                  all_pockets_have_pb_valid_sample=all(x['pb_pass'] > 0 for x in per_case),
                  source_hashes={name: sha256_file(root / name) for name in ('study.json', 'summary.json', 'molecule_metrics.csv')},
                  analysis_code_sha256=sha256_file(__file__), limitations=summary['limitations'])
    assert sum(vina_missing.values()) == summary['generated']['rdkit_valid'] - summary['generated']['vina']['score']['n']
    assert sum(row['pb_pass'] for row in samples) == summary['generated']['pb_pass']
    assert len(pb_missing) == summary['generated']['pb_missing_on_valid']
    write_json(out / 'analysis.json', result)
    paired = pd.DataFrame(all_pairs)
    paired.to_csv(out / 'paired_vina.csv', index=False)
    pd.DataFrame(per_case).to_csv(out / 'per_pocket.csv', index=False)
    pd.DataFrame(samples).to_csv(out / 'measurement_audit.csv', index=False)
    plot_results(out, summary, result, paired, generated, reference)
    write_report(out, root, summary, result, per_case)
    print(json.dumps(dict(output=str(out), paired_vina=vina_pairs, paired_properties=property_pairs,
                          paired_pb=pb_pairs, generated_vina_missing=dict(vina_missing), exact_reference_smiles_outputs=exact_reference), ensure_ascii=False))


def write_report(out, root, s, r, per_case):
    def ci(record):
        a, b = record['mean_delta_cluster_bootstrap95']
        return f'[{a:.3f}, {b:.3f}]'
    rows = []
    for key, label in [('score', '원래 생성 pose'), ('minimize', 'Vina local 최소화'), ('dock', 'Vina 재도킹')]:
        v = r['paired_vina'][key]
        rows.append(f"| {label} | {v['reference_mean']:.3f} | {v['generated_mean']:.3f} | {v['mean_delta']:+.3f} | {ci(v)} | {v['lower']}/{v['n']} |")
    worst = sorted(per_case, key=lambda x: x['vina_missing'], reverse=True)[:5]
    text = f'''# CrossDocked2020 SBDD 완료 결과 분석

작성: {r['created_kst']} (KST). GPU 추가 추론·학습·재평가 없이 저장된 전체 결과를 분석했다.

## 결론

FLOWR.root SPINDR base는 이 조건에서 높은 기본 SBDD 생성 품질을 보였다. 전체 시도의 PB 통과율은 96.82%,
RDKit-valid 출력 중 통과율은 97.97%다. 원본 reference는 96/100 통과했다.
동일 pocket 비교에서 평균 Vina는 좋아지고 QED는 소폭 증가했지만, SA는 나빠졌다.
따라서 “물리적으로 타당한 새 ligand를 잘 생성한다”는 결과이며, 모든 성질을 동시에 개선하거나
주어진 ligand를 보존하며 국소 편집하는 SBMO 성공을 입증한 결과는 아니다.

## 완료 여부와 실험 범위

- test 100개 × 100 raw 시도, 총 10,000개 생성·평가 처리 기록; 원본 100개 평가 완료.
- GPU 생성 완료: 2026-10-19 13:25 KST. 마지막 분자 평가 저장: 2026-10-20 20:50:47 KST.
- 최종 자동 보고서 저장: 2026-10-20 20:52 KST. `status=complete`, case evaluator 실패 없음.
- 생성: SPINDR base, Euler 100-step, 최대 batch 32, GPU5. 원본 graph·개별 atom XYZ를 고정하지 않은 de novo 생성.
- 원본의 heavy-atom 수와 pocket/결합 부위 정의·중심 정보는 사용한다. 원자 수까지 자유로운 생성이 아니다.
- 후처리·선별·재시도로 좋은 분자만 남기지 않았다. 기존 생성 pose와 평가용 최소화/재도킹 복사본은 구분된다.
- reference는 CrossDocked 제공 pose로, 교차 docking된 pose를 포함하며 모든 것이 crystal native는 아니다.

## 기본 유효성과 PB

| 항목 | 수 |
|---|---:|
| 전체 시도 | 10,000 |
| RDKit-valid 분자 | 9,883 (98.83%) |
| build 실패 | 117 (1.17%) |
| 단일 연결 graph | 9,854 (전체의 98.54%) |
| PB 통과 | 9,682 |
| 측정된 PB 실패 | 190 |
| PB 측정 누락 | 11 (모두 600초 제한) |

PB 통과율은 **9,682/10,000=96.82%**, RDKit-valid 기준으로는 **9,682/9,883=97.97%**다.
측정 누락을 제외한 비율은 9,682/9,872=98.08%지만, 이것만 단독으로 보고하지 않는다.
원본은 96/100=96%이며 분모가 다르다. test case별로 원본 PB 여부와 생성 통과 비율을 짝지으면
전체 시도 기준 평균 차이 {r['paired_pb']['attempt']['mean_delta'] * 100:+.2f}%p,
source pocket-group bootstrap 95% 구간은
[{r['paired_pb']['attempt']['mean_delta_cluster_bootstrap95'][0] * 100:.2f}, {r['paired_pb']['attempt']['mean_delta_cluster_bootstrap95'][1] * 100:.2f}]%p다.
4개의 원본 실패만으로 전체 평균 차이가 크게 바뀔 수 있어 원본보다 확실히 우수하다고 단정하지 않는다.
모든 100개 pocket에서 적어도 1개의 PB 통과 분자를 얻었다.

PB는 분자가 연결됐는지, 결합 길이·각도·평면성·내부 에너지·분자 내부와 protein 사이의 충돌 등을 검사한다.
reference graph와의 RMSD 일치나 실제 binding affinity를 검증하는 지표가 아니다.
cofactor/water가 입력 pocket에 없으면 해당 하위 검사를 실제 환경 평가로 해석하지 않는다.
주요 실패 check(중복 가능): 내부 에너지 64, 이중결합 평면성 62, 내부 충돌 31, 비연결 graph 29.

## 같은 pocket의 affinity proxy 비교

Vina는 낮을수록 유리한 계산 점수(kcal/mol)이며 실험적 affinity가 아니다.
원본과 생성물 둘 다 측정 가능한 **97개 동일 test case**에서, 각 case의 생성물 평균에 동일 가중치를 줬다.
일부 pocket에 더 많은 유효 출력이 있다는 이유로 가중치가 커지지 않게 했다.

| 평가 | 원본 평균 | 생성 평균 | 생성−원본 | 평균 차이 95% bootstrap | 평균이 개선된 pocket |
|---|---:|---:|---:|---|---:|
{chr(10).join(rows)}

계산 가능한 분자 쌍 기준 개선율은 원래 pose {r['paired_vina']['score']['improved_ligands']}/{r['paired_vina']['score']['scored_ligands']}
({r['paired_vina']['score']['ligand_improved_fraction']:.2%}), local 최소화 {r['paired_vina']['minimize']['ligand_improved_fraction']:.2%},
재도킹 {r['paired_vina']['dock']['ligand_improved_fraction']:.2%}다. 최고 1개만 뽑은 점수가 아닌 전체 측정 출력 기준이다.
한 pocket의 모든 100개 후보가 좋아진다는 뜻은 아니며, 원래 pose 평균이 좋아진 pocket은 52/97에 그쳤다.
local 최소화·재도킹으로 변한 좌표의 PB를 다시 검사한 실험은 아니므로 그 점수를 원래 pose PB와 혼동하지 않는다.

## 화학 속성

다음은 원본 100개와 RDKit-valid 생성 출력 전체의 기술 통계다. case 균등 평균 및 bootstrap은 `analysis.json`에 별도로 저장했다.

| 지표 | 원본 | 생성 | 해석 |
|---|---:|---:|---|
| QED 평균 | {s['reference']['properties']['qed']['mean']:.3f} | {s['generated']['properties']['qed']['mean']:.3f} | drug-likeness가 소폭 좋아짐; 실제 약효/독성 검증은 아님 |
| SA 평균(1–10) | {s['reference']['properties']['sa_raw_1_to_10']['mean']:.3f} | {s['generated']['properties']['sa_raw_1_to_10']['mean']:.3f} | 낮을수록 쉬우므로 합성 용이성 추정은 나빠짐 |
| LogP 평균 | {s['reference']['properties']['logp']['mean']:.3f} | {s['generated']['properties']['logp']['mean']:.3f} | 친수성 쪽으로 이동; 목표 범위 없이 개선/악화 단정 불가 |
| Lipinski 조건 통과 | {s['reference']['properties']['lipinski_pass']['mean']:.2%} | {s['generated']['properties']['lipinski_pass']['mean']:.2%} | 간단한 drug-like 규칙이며 약효 보장은 아님 |
| Heavy atom 평균 | {s['reference']['properties']['heavy_atoms']['mean']:.2f} | {s['generated']['properties']['heavy_atoms']['mean']:.2f} | 원본 atom 수를 조건으로 줬으므로 독립적인 성능 지표로 삼지 않음 |

목표 QED/SA/LogP를 지정하거나 보상을 최적화한 생성이 아니다. 어떤 방향으로 샘플링 분포가 이동했는지 보는 결과다.
case별 QED 차이는 +0.028이지만 95% pocket-group bootstrap 구간이
[-0.004, +0.060]으로 0을 포함하므로 확실한 QED 개선으로 단정하지 않는다.
SA는 79/100개 case에서 생성 평균이 더 높았고, 평균 차이 +0.487의 95% 구간은 [+0.302, +0.669]다.

## 다양성과 원본 유사도

- 고유 canonical isomeric SMILES: 7,739/9,883, 전체 출력 내 고유 비율 78.31%.
- pocket별 고유 비율 평균: {r['macro_unique_fraction']:.2%}; pocket 내 fingerprint 거리 평균: {r['macro_diversity']:.3f}.
- 원본과의 Morgan Tanimoto 평균: 0.177 (1이면 같은 fingerprint, 작을수록 다름).
- 원본과 canonical SMILES가 정확히 같은 출력: {r['exact_reference_smiles_outputs']}/9,883.
- local CrossDocked train SMILES 기준 신규 출력: 9,822/9,883=99.38%, 고유 training reference 8,476개.

이는 원본의 조금 바뀐 유사체만 생성하는 조건이 아니라 다양한 새로운 graph를 만드는 조건임을 보여준다.
novelty는 local CrossDocked training SMILES 대비 값이며 FLOWR 전체 사전학습 데이터 대비 신규성은 알 수 없다.

## Strain과 좌표 분포

Strain은 “생성 pose를 ligand 단독 MMFF94s로 최소화했을 때 얼마나 에너지가 내려가는가”다.
낮으면 그 force field 관점에서 덜 긴장된 pose이지만 protein 내 affinity나 원래 pose 정확도와 같은 뜻은 아니다.
원본/생성 strain 중앙값은 **65.26/33.82 kcal/mol**, 평균은 **76.21/75.97**이다.
생성의 95백분위는 {r['strain_quantiles']['0.95']:.2f}, 99백분위는 {r['strain_quantiles']['0.99']:.2f},
최대는 {r['strain_quantiles']['1.0']:.2f}다. 1,000 초과 이상값이 {len(r['strain_over_1000'])}개 있어 평균을 크게 끌어올린다.
이상값을 제거하지 않았으며 해당 분자별 PB 결과는 `analysis.json`에 기록했다.
1,000 초과 15개는 모두 PB 검사에서 실패해, 드물게 발생하는 큰 물리적 실패를 PB가 걸러냈음을 확인했다.
생성 strain 측정 가능 9,856개, 파라미터 미지원 27개; 수렴 9,855개와 미수렴 1개를 구분했다.

reference 대비 pooled bond-length W1은 0.00935Å, bond-angle W1은 0.33755°다.
W1은 두 분포의 차이로 낮을수록 유사하지만 평균 분포가 비슷해도 개별 분자의 실패가 사라지는 것은 아니다.
이는 논문과 다른 unbinned/pooled 집계이며 논문의 weighted W1과 직접 비교하지 않는다.

## 누락과 오래 걸린 이유

- RDKit-valid 9,883개 중 Vina 3종 모두 측정된 출력은 9,633개다.
- 누락 250개: {json.dumps(r['generated_vina_missing'], ensure_ascii=False)}.
- 원본 Vina는 97/100개만 측정됐고 3개는 600초 제한이었다.
- 현재 평가 wrapper는 Vina 3종을 마지막에 일괄 저장하므로 **재도킹이 시간 초과되면 앞선 score/minimize도 함께 누락**된다.
  따라서 동일한 9,633개가 세 metric의 분모다. 누락은 실패 점수 0이나 개선으로 대체하지 않았다.
- 일부 pocket/분자에서 반복된 600초 제한이 종료 시각을 늦췄다. 누락이 복잡한 분자에 몰릴 수 있어
  Vina 평균은 관측 가능한 부분집합의 결과이며 10,000개 전체의 무조건 평균은 아니다.
- Vina 누락이 가장 많은 case: {', '.join(x['case_id'] + ' (' + str(x['vina_missing']) + ')' for x in worst)}.

## 논문 및 SBMO와의 관계

[논문 CrossDocked 표 2](https://arxiv.org/html/2510.02578v3#S5.SS2)는 RDKit-valid 부분집합에서
PB-valid 0.97을 보고한다. 이번 조건부 PB 0.9797과 숫자는 가깝지만 checkpoint가 SPINDR base이고,
protein/receptor 준비와 strain 및 W1 구현 등이 달라 논문 재현이나 논문보다 우수하다는 결론은 내리지 않는다.

이번 결과는 FLOWR.root의 de novo 생성 자체가 전체적으로 붕괴한 상태가 아님을 보여준다.
이전 SBMO의 고정 core·편집 graph 조건에서의 어려움과 이번 높은 SBDD PB는 함께 성립할 수 있다.
SBMO에는 원본 유사성/고정 부분 보존/편집 실행과 원하는 성질 변화까지 별도로 확인해야 한다.
따라서 이 실험만으로 추가 학습 필요 여부나 기존 조건부 실패 원인을 확정하지 않는다.

## 그림과 재현 자료

![유효성·동일 pocket Vina·QED·SA](overview.png)

![pocket별 Vina·strain 꼬리 분포·PB 실패 종류](diagnostics.png)

- `analysis.json`: 동일 pocket 비교, 20,000회 source pocket-group bootstrap, 누락과 이상값, 입력 hash.
- `paired_vina.csv`: test case별 원본·생성 평균과 차이.
- `per_pocket.csv`: pocket별 PB·누락·다양성.
- `measurement_audit.csv`: 10,000개 출력의 metric 처리 상태.
- 원본 실험: `{root}`. 본 분석은 원본 결과와 cache를 변경하지 않았다.
'''
    (out / 'REPORT.md').write_text(text)


if __name__ == '__main__':
    main()
