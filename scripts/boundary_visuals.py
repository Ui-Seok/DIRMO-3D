"""Static scientific figures and dependency-free HTML for paired saved poses."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

STAGES = ('native', 'fm', 'uff')
LABELS = ('Native reference', 'FM generated', 'FM + pocket UFF')
COLORS = {'core': '#168c8a', 'free': '#eb981b', 'boundary': '#bd3186', 'pocket': '#a9b4c0'}
ELEMENTS = {'N': '#3467dc', 'O': '#e54848', 'S': '#c4a300', 'P': '#b16ce3',
            'F': '#68ad40', 'Cl': '#3d9a3b', 'Br': '#ab4932', 'I': '#9049bd'}


def color(atom):
    return ELEMENTS.get(atom['element'], COLORS['core'] if atom['fixed'] else COLORS['free'])


def basis_for(c):
    # Put the damaged bond in the image plane. Keep a second, maximally spread
    # direction for this ligand. This is a common orthographic camera only:
    # neither source coordinates nor metric distances are transformed/aligned.
    xyz = np.array(c['coordinates']['fm'])
    b = max(c['boundary_bonds'], key=lambda b: abs(b['fm_minus_native_A']))
    i, j = b['indices_0based']
    x = xyz[j] - xyz[i]
    x /= np.linalg.norm(x)
    centered = xyz - xyz.mean(axis=0)
    perpendicular = centered - (centered @ x)[:, None] * x[None, :]
    _, _, vt = np.linalg.svd(perpendicular, full_matrices=False)
    y = vt[0] - np.dot(vt[0], x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def limits(c, basis, zoom):
    projected = {s: np.array(c['coordinates'][s]) @ basis for s in STAGES}
    if zoom:
        ids = sorted({i for b in c['boundary_bonds'] for i in b['indices_0based']})
        points = np.concatenate([p[ids, :2] for p in projected.values()])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        span = max(5.6, float(np.ptp(points, axis=0).max()) + 3.0)
    else:
        points = np.concatenate([p[:, :2] for p in projected.values()])
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        span = float(np.ptp(points, axis=0).max()) + 5.0
    return center, span


def draw_pose(ax, c, stage, zoom=False, title=True):
    basis = basis_for(c)
    xyz = np.array(c['coordinates'][stage]) @ basis
    pocket = np.array([a['xyz'] for a in c['pocket']['atoms']]) @ basis
    center, span = limits(c, basis, zoom)
    all_stages = np.concatenate([np.array(c['coordinates'][s]) @ basis for s in STAGES])
    near = np.linalg.norm(pocket[:, None, :] - all_stages[None, :, :], axis=2).min(axis=1) <= (4. if zoom else 5.)
    segments = [[pocket[i, :2], pocket[j, :2]] for i, j in c['pocket']['bonds'] if near[i] or near[j]]
    ax.add_collection(LineCollection(segments, colors=COLORS['pocket'], linewidths=.7,
                                     alpha=.4 if zoom else .3, zorder=0))
    ax.scatter(pocket[near, 0], pocket[near, 1], s=4, color=COLORS['pocket'], alpha=.35, zorder=0)
    if zoom and stage != 'native':
        native = np.array(c['coordinates']['native']) @ basis
        for i, j, _ in c['bonds']:
            if not c['fixed_mask'][i] or not c['fixed_mask'][j]:
                ax.plot(native[[i, j], 0], native[[i, j], 1], color='#8b96a3',
                        linewidth=1, alpha=.75, linestyle='--', zorder=1)
    for i, j, order in sorted(c['bonds'], key=lambda b: (xyz[b[0], 2] + xyz[b[1], 2]) / 2):
        middle = (xyz[i, :2] + xyz[j, :2]) / 2
        for k in (i, j):
            ax.plot([xyz[k, 0], middle[0]], [xyz[k, 1], middle[1]],
                    color=color(c['atoms_data'][k]), linewidth=3.1 if zoom else 2.3,
                    solid_capstyle='round', zorder=3)
    for b in c['boundary_bonds']:
        i, j = b['indices_0based']
        ax.plot(xyz[[i, j], 0], xyz[[i, j], 1], color=COLORS['boundary'],
                linewidth=4.2 if zoom else 3.6, zorder=4)
        middle = xyz[[i, j], :2].mean(axis=0)
        length = b['lengths_A'][stage]
        if zoom:
            vector = xyz[j, :2] - xyz[i, :2]
            offset = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector) * 25
            alignment = 'left' if offset[0] > 12 else ('right' if offset[0] < -12 else 'center')
            ax.annotate(f'{length:.3f} Å', middle, xytext=offset, textcoords='offset points',
                        color=COLORS['boundary'], ha=alignment, fontsize=12, weight='bold', zorder=10,
                        bbox=dict(boxstyle='round,pad=.2', fc='white', ec='none', alpha=.92))
    ids = np.argsort(xyz[:, 2])
    ax.scatter(xyz[ids, 0], xyz[ids, 1], s=46 if zoom else 30,
               c=[color(c['atoms_data'][i]) for i in ids], edgecolors='white', linewidths=.5, zorder=5)
    endpoints = sorted({i for b in c['boundary_bonds'] for i in b['indices_0based']})
    if zoom:
        for i in endpoints:
            a = c['atoms_data'][i]
            ax.annotate(f"{a['element']}{i + 1}", xyz[i, :2], xytext=(0, -18),
                        textcoords='offset points', ha='center', fontsize=10, weight='bold', zorder=10,
                        bbox=dict(boxstyle='round,pad=.1', fc='white', ec='none', alpha=.85))
    ax.set_xlim(center[0] - span / 2, center[0] + span / 2)
    ax.set_ylim(center[1] - span / 2, center[1] + span / 2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#d9e1e8')
    if title:
        pb = c['pb'][stage]['passes']
        counts = c['pb_check_counts'][stage]
        vina = c['vina_' + stage]
        label = LABELS[STAGES.index(stage)]
        ax.set_title(f'{label}\nPB {"PASS" if pb else "FAIL"} ({counts["passed"]}/{counts["total"]})'
                     f'  |  Vina {vina:.3f}', fontsize=11, color='#1c2e43', pad=10)
    # Scale bar, in the original angstrom coordinate system (orthographic).
    x0, y0 = center - span * .43
    bar = 1. if zoom else 2.
    ax.plot([x0, x0 + bar], [y0, y0], color='#526174', linewidth=2)
    ax.text(x0 + bar / 2, y0 + span * .025, f'{bar:.0f} Å', fontsize=8, ha='center', color='#526174')


def legend(fig, y=.025):
    handles = [Line2D([0], [0], color=COLORS['core'], lw=4, label='Fixed core carbon'),
               Line2D([0], [0], color=COLORS['free'], lw=4, label='Free-region carbon'),
               Line2D([0], [0], color=COLORS['boundary'], lw=4, label='Fixed/free boundary bond'),
               Line2D([0], [0], color=COLORS['pocket'], lw=2, label='Pocket context'),
               Line2D([0], [0], color='#3467dc', marker='o', lw=0, label='N'),
               Line2D([0], [0], color='#e54848', marker='o', lw=0, label='O'),
               Line2D([0], [0], color='#c4a300', marker='o', lw=0, label='S')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, y),
               ncol=7, frameon=False, fontsize=9)


def make_region_map(c, folder):
    mol = Chem.MolFromMolFile(str(folder / 'native.sdf'), removeHs=False)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    rdDepictor.Compute2DCoords(mol)
    options = Draw.MolDrawOptions()
    options.bondLineWidth = 2
    options.legendFontSize = 23
    options.legendFraction = .12
    endpoints = {i for b in c['boundary_bonds'] for i in b['indices_0based']}
    for i in endpoints:
        options.atomLabels[i] = f"{mol.GetAtomWithIdx(i).GetSymbol()}{i + 1}"
    rgb = matplotlib.colors.to_rgb
    highlights = {i: rgb(COLORS['core'] if fixed else COLORS['free'])
                  for i, fixed in enumerate(c['fixed_mask'])}
    bond_ids = [mol.GetBondBetweenAtoms(*b['indices_0based']).GetIdx() for b in c['boundary_bonds']]
    image = Draw.MolToImage(mol, size=(1000, 650), options=options,
                            highlightAtoms=list(range(mol.GetNumAtoms())), highlightAtomColors=highlights,
                            highlightBonds=bond_ids,
                            highlightBondColors={i: rgb(COLORS['boundary']) for i in bond_ids},
                            legend=f"{c['case_id']} | teal: fixed core | amber: free region | pink: boundary")
    image.save(folder / 'region_map.png')


def make_figures(cases, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    overview, axes = plt.subplots(len(cases), 3, figsize=(14, 4.15 * len(cases)), squeeze=False)
    for row, c in enumerate(cases):
        folder = output / 'cases' / f"{c['case_id']}_draw{c['draw']:02d}"
        for col, stage in enumerate(STAGES):
            draw_pose(axes[row, col], c, stage, zoom=False)
        axes[row, 0].set_ylabel(f"{c['case_id']} / draw {c['draw']}\n{c['selection_reason'].replace('_', ' ')}",
                                fontsize=10, labelpad=12)
        fig, ax = plt.subplots(2, 3, figsize=(16, 10.8), squeeze=False)
        for col, stage in enumerate(STAGES):
            draw_pose(ax[0, col], c, stage)
            draw_pose(ax[1, col], c, stage, zoom=True, title=False)
            details = [b['pb_bond_details'][stage]['pb_bond_violation'] for b in c['boundary_bonds']]
            ax[1, col].set_title('Boundary detail: ' + ('PB bond violation' if any(details) else 'PB bond within bounds'),
                                 fontsize=11, color='#ad2359' if any(details) else '#167764')
        ax[0, 0].set_ylabel('Pocket + ligand', fontsize=12, labelpad=15)
        ax[1, 0].set_ylabel('Boundary close-up', fontsize=12, labelpad=15)
        failure = ', '.join(c['pb']['fm']['failed_checks'])
        fig.suptitle(f"{c['case_id']} / original graph / draw {c['draw']}\n"
                     f"FM boundary failure and pocket-aware UFF repair | held drift {c['held_max_drift_A']['uff']:.4f} Å",
                     fontsize=15, weight='bold', y=.985)
        fig.text(.5, .075, textwrap.fill('FM failed PB checks: ' + failure, 125), ha='center', fontsize=9)
        fig.text(.5, .052, 'Same orthographic camera and scale per row; no structural alignment. '
                 'Dashed gray in close-ups = native free-region reference. Labels are 1-based SDF indices.',
                 ha='center', fontsize=8, color='#526174')
        legend(fig, .012)
        fig.subplots_adjust(left=.07, right=.98, top=.89, bottom=.15, wspace=.12, hspace=.20)
        for suffix in ('png', 'pdf'):
            fig.savefig(folder / f'comparison.{suffix}', dpi=200, facecolor='white')
        plt.close(fig)
        make_region_map(c, folder)
    overview.suptitle('Boundary failures: native → FM generation → pocket-aware UFF',
                       fontsize=17, weight='bold', y=.993)
    legend(overview, .009)
    overview.subplots_adjust(left=.16, right=.98, top=.95, bottom=.06, wspace=.13, hspace=.36)
    for suffix in ('png', 'pdf'):
        overview.savefig(output / f'overview.{suffix}', dpi=170, facecolor='white')
    plt.close(overview)
    fig, axes = plt.subplots(2, len(cases), figsize=(4.0 * len(cases), 8.3), squeeze=False)
    stage_colors = ['#899aaf', '#da637b', '#2c9a88']
    all_vina = [c['vina_' + s] for c in cases for s in STAGES]
    for col, c in enumerate(cases):
        ax = axes[0, col]
        values = [c['vina_' + s] for s in STAGES]
        bars = ax.bar(range(3), values, color=stage_colors, width=.65)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.3f}',
                    ha='center', va='bottom' if value > 0 else 'top', fontsize=10)
        ax.axhline(0, color='#526174', lw=.8)
        ax.set_ylim(min(all_vina) - 2, max(all_vina) + 2)
        ax.set_xticks(range(3), ['Native', 'FM', '+ UFF'])
        ax.set_title(f"{c['case_id']} / draw {c['draw']}\nPB: PASS → FAIL → PASS", fontsize=11)
        ax.set_ylabel('Vina score-only (kcal/mol); lower is favorable' if col == 0 else '')
        ax.grid(axis='y', alpha=.15)
        ax = axes[1, col]
        b = max(c['boundary_bonds'], key=lambda b: abs(b['fm_minus_native_A']))
        values = [b['lengths_A'][s] for s in STAGES]
        bars = ax.bar(range(3), values, color=stage_colors, width=.65)
        ax.bar_label(bars, fmt='%.3f', padding=3)
        native_diagnostic = b['pb_bond_details']['native']
        params = c['pb_bond_threshold']
        low = native_diagnostic['lower_bound_A'] * (1 - params)
        high = native_diagnostic['upper_bound_A'] * (1 + params)
        ax.axhspan(low, high, color='#3da58e', alpha=.12, label='PB dock bond tolerance')
        ax.set_ylim(0, 3.5)
        ax.set_xticks(range(3), ['Native', 'FM', '+ UFF'])
        i, j = b['labels_1based']
        ax.set_title(f"Boundary {b['elements'][0]}{i}–{b['elements'][1]}{j}", fontsize=11)
        ax.set_ylabel('3D boundary bond length (Å)' if col == 0 else '')
        ax.grid(axis='y', alpha=.15)
        if col == 0:
            ax.legend(loc='upper left', fontsize=8, frameon=False)
        for a in axes[:, col]:
            a.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Selected illustrations: physical repair and Vina change are separate outcomes',
                  fontsize=15, weight='bold')
    fig.text(.5, .018, 'Post-hoc examples with confirmed boundary PB violations; no population effect size can be inferred. '
             'PB check counts are not a calibrated quality score.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'metrics.{suffix}', dpi=190, facecolor='white')
    plt.close(fig)


def make_html(cases, output):
    template = Path(__file__).with_name('boundary_viewer.html').read_text()
    data = json.dumps(cases, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (output / 'index.html').write_text(template.replace('__CASES_JSON__', data)
                                     .replace('__SAMPLE_COUNT__', str(len(cases))))


def make_report(manifest, output):
    reasons = {'stretched_boundary': '길어진 경계 결합', 'compressed_boundary': '눌린 경계 결합',
               'largest_vina_recovery': '큰 Vina 회복',
               'pb_recovery_without_vina_recovery': 'PB 회복·Vina 악화 대조',
               'additional_vina_recovery': '추가 Vina 회복 사례', 'explicit_user_selection': '지정 사례'}
    lines = [
        '# 경계 손상과 pocket-aware UFF 보정 시각화', '',
        f"작성 시각: {manifest['created_at_kst']} (KST, Asia/Seoul).", '',
        '## 파일 보기', '',
        '- `index.html`: 인터넷·서버·추가 JS 라이브러리 없이 여는 회전/확대 가능한 3D 뷰어. '
        '각 사례의 원본·FM·UFF를 나란히 표시한다. 세 뷰의 카메라가 함께 움직인다.',
        '- `overview.png` / `.pdf`: 모든 사례의 pocket + ligand 비교.',
        '- `metrics.png` / `.pdf` / `.svg`: PB, Vina, 경계 결합 길이 비교.',
        '- `cases/…/comparison.png` / `.pdf`: 사례별 pocket 전체와 경계 확대.',
        '- `cases/…/region_map.png`: 고정/자유 영역의 2D 화학 구조.',
        '- `cases/…/{native,fm,uff}.sdf`, `pocket.pdb`, `condition.json`: 복사한 원본 구조와 mask.',
        '- `cases/…/view.pml`: 해당 폴더에서 `pymol view.pml`로 여는 PyMOL 장면.',
        '- `selected_cases.csv`, `boundary_bonds.csv`, `manifest.json`: 수치·출처·SHA256·선정 근거.', '',
        '## 어떤 실험인가', '',
        'PDBbind2019 refined-set의 기존 64-step 실험 `clamp_gpu1_20261021_v2/full` '
        '및 후처리 `refine_20261022_v1`의 **original graph** 조건을 사용했다. '
        'CrossDocked de novo 생성이나 최근 중단한 Router 데이터 수집과는 다른 실험이다.', '',
        '원본 ligand의 전체 화학 그래프를 유지하고 일부 core 원자의 좌표를 고정한 채 FLOWR.root로 '
        '자유 영역의 좌표를 생성한 뒤, 같은 생성물에 pocket-aware UFF hard 보정을 적용했다. '
        '고정 mask는 실제 `condition.json`에서 읽었다. 경계는 **고정 원자와 자유 원자를 직접 연결하는 결합**이다. '
        '원래 분석의 hop-0에는 경계와 인접한 결합도 들어가지만 이번 그림에서는 직접 경계 결합을 표시한다.', '',
        'UFF 조건은 ligand에 수소를 추가하고 ligand 주변 8 Å 이내 잔기로 만든 pocket을 함께 넣어 '
        '`ignoreInterfragInteractions=False`로 최적화한 것이다. pocket 전체와 ligand core를 고정하고 '
        '최대 500회 최소화했다. soft/tether arm은 이번 비교에 포함하지 않았다. '
        '이번 작업은 이미 저장된 전후 결과를 사용했고 새로운 FM·LLM 실행, UFF 최적화, docking은 하지 않았다.', '',
        '## 샘플 선정', '',
        '원본 PB 통과, FM PB 실패, pocket UFF 후 PB 통과, graph/입체화학 보존, '
        '고정 좌표 변위 ≤0.001 Å를 만족하는 사례 중에서 FM Vina가 원본보다 1 kcal/mol 이상 불리하고 '
        '직접 경계 결합의 원본 대비 길이 오차가 0.1 Å 이상이며 UFF 후 감소한 사례를 골랐다. '
        f"이 필터를 통과한 출력은 {manifest['filter_counts']['boundary_recovered']}개이며, "
        f"그중 실제 경계 결합의 PB 위반도 확인한 {manifest['filter_counts']['boundary_pb_violation']}개에서 골랐다. "
        '길어짐·눌림·큰 회복·PB 회복만 있는 사례를 서로 다른 pocket에서 결정론적으로 선택했다. '
        '**사후에 고른 설명용 사례이며 무작위 표본이나 전체 성능 추정치가 아니다.**', '',
        '| Pocket / draw | 선정 이유 | PB 통과 검사 수: 원본→FM→UFF | Vina: 원본→FM→UFF (kcal/mol) | 경계 길이: 원본→FM→UFF (Å) |',
        '|---|---|---|---|---|',
    ]
    for c in manifest['cases']:
        checks = ' → '.join(f"{c['pb_check_counts'][s]['passed']}/{c['pb_check_counts'][s]['total']}" for s in STAGES)
        vina = ' → '.join(f"{c['vina_' + s]:.3f}" for s in STAGES)
        b = max(c['boundary_bonds'], key=lambda b: abs(b['fm_minus_native_A']))
        lengths = ' → '.join(f"{b['lengths_A'][s]:.3f}" for s in STAGES)
        lines.append(f"| {c['case_id']} / {c['draw']} | {reasons[c['selection_reason']]} | {checks} | {vina} | {lengths} |")
    lines += [
        '', '## 그림 읽는 법', '',
        '- 청록색 탄소: 고정 core. 주황색 탄소: 자유 영역. 질소 파랑·산소 빨강·황 노랑 등은 원소 색이다. '
        '분홍색 선이 직접 고정/자유 경계 결합이고 회색 선은 pocket이다.',
        '- 원자 표시는 SDF 원자 순서의 **1-based 번호**다. 내부 0-based index와 atom-map ID도 JSON에 따로 저장했다.',
        '- 정적 그림은 동일한 정사영 카메라와 축척을 적용한다. 경계 확대의 회색 점선은 원본 자유 영역이다. '
        '3D 거리는 원본 좌표에서 계산했으므로 2D 투영의 눈대중 길이와 다를 수 있다.',
        '- 세 ligand를 서로 정렬하지 않았다. 원본 pocket 좌표계를 그대로 공유한다. '
        '고정 좌표의 변위와 실제 결합 길이를 다시 계산해 저장 기록과 대조했다.',
        '- HTML에서는 마우스 드래그로 회전하고 휠로 확대한다. 원본 겹치기·원자 번호·pocket 표시를 전환할 수 있다. '
        '방향성 수소결합·단백질 표면·2차 구조를 추론해서 그린 것은 아니며 pocket 결합 선은 거리 기반 표시용이다.', '',
        '## PB와 Vina 해석', '',
        'PB의 기본 결과는 전체 검사 **통과/실패**다. `17/20` 같은 표시는 통과한 검사 개수를 보여주는 '
        '보조 정보이며 연속 PB score, 확률, 또는 서로 동등한 20개 품질 축이 아니다. '
        '물·cofactor가 없는 protein-only 입력에서는 관련 검사가 정보 없이 통과할 수 있다.', '',
        '경계 결합의 PB 위반 여부는 설치된 동일 `dock` 설정의 빠른 distance-geometry 모듈로 직접 확인했다. '
        '실제 PB bound와 25% 허용치를 사용하며, 샘플 선별용 원본 대비 0.1 Å 오차와는 다른 기준이다. '
        '전체 PB 결과와 Vina는 기존 저장값을 그대로 사용했다. PB 실패 원인이 경계 하나뿐이라고 '
        '단정하지 않고 다른 실패 항목도 JSON/HTML에 함께 표시했다.', '',
        'Vina는 **score-only**이며 낮을수록 유리하다. re-docking 결과나 실험 결합력이 아니다. '
        'UFF가 물리적 구조를 수선해도 affinity가 반드시 개선되지는 않는다. '
        'PB는 회복했지만 Vina가 악화되는 대조 사례를 함께 포함했다. '
        'UFF와 PB는 일부 기하/에너지 대상을 공유하므로 PB 회복을 독립적인 약효 검증으로 해석하면 안 된다.', '',
        '고정 원자 좌표가 유지되는 범위에서도 자유 영역의 방향과 pocket 접촉은 달라질 수 있다. '
        'native보다 Vina가 여전히 불리한 경우는 원래 결합 양상을 완전히 회복했다고 볼 수 없다. '
        '이번 사례는 화학 그래프가 동일하므로 QED/SA 같은 그래프 물성 개선을 주장하지 않는다.', '',
        '## 재현', '',
        '필요 패키지는 기존 `dirmo` 환경의 numpy, scipy, RDKit, matplotlib, pandas 및 '
        'vendored PoseBusters다. HTML만 볼 때는 Python 환경이 필요 없다.', '',
        '```bash', 'conda activate dirmo', 'cd /home/uslee/code/DIRMO-3D',
        'CUDA_VISIBLE_DEVICES="" python scripts/visualize_boundary_refinement.py \\',
        '  --samples 4 \\', '  --output reports/boundary-visualization-reproduced', '```', '',
        '`--samples`는 3~5를 지원한다. 출력 디렉터리가 이미 있으면 덮어쓰지 않는다. '
        '특정 사례를 고정하려면 적격한 서로 다른 pocket을 `--case 4yuz:6` 형식으로 3~5개 지정한다.', '',
        '![수치 비교](metrics.png)', '', '![Pocket와 ligand 비교](overview.png)', '',
    ]
    for c in manifest['cases']:
        relative = f"cases/{c['case_id']}_draw{c['draw']:02d}"
        lines += [f"### {c['case_id']} / draw {c['draw']}", '',
                  f"FM PB 실패 항목: {', '.join(c['pb']['fm']['failed_checks'])}.", '',
                  f"![{c['case_id']} 경계 비교]({relative}/comparison.png)", '']
    (output / 'README.md').write_text('\n'.join(lines))
