# 편집 graph 조건 FLOWR.root 실험 분석 — 2026-10-18

**100-pocket 실험은 완주하지 못했다. 43개 pocket의 생성이 완료됐고, 44번째 5edl에서 입력 표현 오류로 중단됐다. 평가가 완결된 30개 pocket만 아래 품질 비교에 사용한다.**

분석 시각: 2026-10-18T16:08:34.964381+09:00 (KST). 원본: `/home/uslee/code/DIRMO-3D/runs/fm_edited_graph_gpu3_100_20261016_v1`.

## 실행 상태와 중단 원인

- 준비 100/100, 생성 완료 43/100, 완결 평가 30/100.
- 생성 완료 43개: 2,752 예정 슬롯 중 SDF 2,574개, 초기화 실패 178개. SDF 중 엄격한 graph/stereo 보존 2,302개, 계약 실패 272개.
- 44번째 5edl: 64슬롯이 `CapabilityError: Tensorization changed the graph or atom order`로 실패. 8개 graph/조건 batch가 연속 실패하자 안전 중단했다. 후속 56개 pocket의 3,584슬롯은 실행되지 않았다. 이 중단은 OOM 기록이 아니라 graph round-trip 거부다.
- 저장된 5edl 입력의 RDKit 표현에는 carbon radical 전자 1개(`[CH]`)가 있다. CPU에서 FLOWR `GeometricMol.from_rdkit → to_rdkit`만 실행해도 `[CH] → C`로 변해 radical 1→0, canonical identity 불일치가 재현됐다. 기존 tensor 표현에 radical/수소 지정 보존 채널이 없다는 설명과 일치한다. 이는 이번 입력 표현의 진단이며 실제 실험 ligand가 radical이라는 주장은 아니다. 실제 로그의 검사 전체를 모델과 함께 재실행한 것은 아니다.
- 무작위 시드와 기존 cohort는 유지했다. 이 입력을 자동 중성화·교체하거나 이후 결과를 채워 넣지 않았다. 새 GPU 실행은 하지 않았다.
- CPU 평가 30개는 2026-10-16 22:23 KST까지 끝났다. 원래 summary는 controller 종료 시각 22:13 KST의 24개/일부 행만 반영했다. 이번에 원본 exit snapshot을 보관하고 summary/metrics를 30개 완결 결과로 갱신했다.
- 생성은 끝났지만 평가가 시작되지 않은 13개: 5zwh, 4yuz, 3vru, 2qfo, 1bm6, 5fnc, 1k3t, 4c70, 1apw, 4bdd, 4wzv, 3vrw, 1sgu.

## 실험이 검증한 것

동일 pocket에서 원본 graph와 자유 말단의 같은 C를 N/O/F로 바꾼 세 graph를 미리 확정했다. graph당 graph-only ETKDG 시작 pose 8개를 만들고, 보호 core에 배치했다. 원본 자유 영역 좌표는 사용하지 않았다. 같은 시작 SDF를 전체 좌표 자유(FM whole)와 core XYZ 고정(FM core)에 넣었다. **둘 다 편집 후 전체 graph를 고정**한다. FM이 자유 영역의 graph를 직접 설계한 실험은 아니다. Pocket은 원본 frame의 고정 6 Å protein-only 영역이며, whole도 초기 core 위치 정보를 받으므로 blind docking이 아니다. 원본 initial ligand에서 어느 편집이 더 좋은지 LLM이 고른 것도 아니다.

PB는 dock 20-check 설정이다. pocket이 protein-only이므로 water/cofactor 검사는 실제 보조분자가 없는 입력에 대한 결과다. 종합 성공은 PB 전체 통과 + 요청 graph의 입체화학 보존이며, core 조건은 보호 heavy XYZ 최대 이동 ≤0.001 Å도 요구한다. 초기화 실패는 성공률 분모에 남겼다. 새 편집 분자의 실험 pose가 없으므로 native 전체 RMSD를 정확도로 쓰지 않았다.

## 물리 타당성과 종합 성공

| 방법 | PB 통과 | 종합 성공 | SDF 생성/준비 |
| --- | --- | --- | --- |
| Native | 27/30 (90.0%) | 27/30 (90.0%) | 30/30 (100.0%) |
| Starting pose | 397/960 (41.4%) | 397/960 (41.4%) | 900/960 (93.8%) |
| UFF whole | 252/960 (26.2%) | 244/960 (25.4%) | 900/960 (93.8%) |
| UFF core | 503/960 (52.4%) | 489/960 (50.9%) | 900/960 (93.8%) |
| FM whole | 716/960 (74.6%) | 652/960 (67.9%) | 900/960 (93.8%) |
| FM core | 600/960 (62.5%) | 563/960 (58.6%) | 900/960 (93.8%) |

Native를 제외한 각 방법은 같은 30 pockets × 4 graphs × 8 draws = 960슬롯이다. 시작 pose 초기화는 900/960 성공했다. PB 측정 timeout은 없으며 PB 누락 60개/방법은 초기화 SDF 부재다. Native는 30개 기준으로 별도 해석한다. Native PB 실패는 내부 에너지 2개, 내부 입체 충돌 1개이며 단백질 접촉 실패는 없었다.

![Generation quality](/home/uslee/code/DIRMO-3D/reports/fm-edited-graph-20261018-v1/generation_quality.png)

| Graph | Starting pose | UFF whole | UFF core | FM whole | FM core |
| --- | --- | --- | --- | --- | --- |
| original | 100/240 (41.7%) | 61/240 (25.4%) | 122/240 (50.8%) | 168/240 (70.0%) | 146/240 (60.8%) |
| C_to_N | 99/240 (41.2%) | 59/240 (24.6%) | 118/240 (49.2%) | 159/240 (66.2%) | 135/240 (56.2%) |
| C_to_O | 98/240 (40.8%) | 60/240 (25.0%) | 120/240 (50.0%) | 156/240 (65.0%) | 135/240 (56.2%) |
| C_to_F | 100/240 (41.7%) | 64/240 (26.7%) | 129/240 (53.8%) | 169/240 (70.4%) | 147/240 (61.2%) |

8개 후보 중 종합 성공이 한 개 이상 있는 pocket 비율:

| Graph | FM whole | FM core |
| --- | --- | --- |
| original | 27/30 (90.0%) | 25/30 (83.3%) |
| C_to_N | 28/30 (93.3%) | 27/30 (90.0%) |
| C_to_O | 26/30 (86.7%) | 27/30 (90.0%) |
| C_to_F | 28/30 (93.3%) | 28/30 (93.3%) |

모든 비교는 동일 30개 pocket에서 한다. 8개 중 성공률은 1회 생성 성공률과 구분한다. 여러 후보를 생성하면 성공 후보를 찾을 가능성이 커진다. 위 결과는 최종 100개 성능이나 사전학습과 분리된 test benchmark 성능이 아니다. 중단과 CPU 평가 완료 순서에 의한 부분 cohort이므로 과도한 일반화를 피한다.

## 조건을 실제로 지켰는가

전체 저장 SDF 2574개를 독립 재검사했다. 모든 SDF는 원자 순서별 원소·전하·동위원소·radical·수소 수·결합 연결/차수를 보존했다. 즉 272개 계약 실패는 비입체 graph가 바뀐 것이 아니라 좌표에서 재추론한 입체화학이 요청과 달랐기 때문이다. 분자 identity 보존에는 이 입체화학도 필요하므로 성공으로 세지 않았다.

- Core: 1287/1287 출력의 보호 좌표 보존, 저장 SDF 최대 이동 0.000000 Å.
- Whole: 전체 좌표가 자유라 원본 core 이동은 허용한다. 원본 core 최대 이동 4.483 Å.
- 엄격한 stereo 보존: whole 1116/1287 (86.7%), core 1186/1287 (92.2%).
- 저장 condition 2574개에서 graph/좌표 mask와 입력 hash 확인. 전체/부분 조건 사이 시작 SDF는 1287/1287쌍 동일했다.

core 고정 후 물리 성공률이 낮아진 이유는 보호 XYZ 무시보다, 고정 영역과 자유 영역의 경계에서 결합 기하를 동시에 만족시키기 어려운 현상에 가깝다. 평가된 900 SDF/방법에서 core는 결합 길이 208개·각도 179개, whole은 각각 63개·76개가 실패했다. 반면 단백질 최소거리 실패는 시작 pose 450개 → whole 30개/core 36개로 줄었다. 단일 출력이 여러 검사를 실패할 수 있으므로 검사 실패 수는 합산하지 않는다. 경계가 직접 원인인지는 추가 공간별 진단이 필요하다.

UFF whole은 ligand-only energy를 줄여도 단백질을 고려하지 않아 protein 최소거리 실패가 638/900개로 늘었다. 이 결과는 단백질 문맥 없이 ligand만 최적화하는 대조군의 한계를 보여준다.

## Vina: 같은 시작 pose와 비교

Vina receptor 준비 성공 19/30, 실패 11/30이다. 잔기 template 불일치·결손 등 로그는 receptor_availability.csv의 파일 경로를 따른다. 실패 사례를 임의 residue 삭제로 통과시키지 않았다. 각 방법의 점수는 567개 입력/출력 쌍에서 비교 가능했다. Vina는 score-only 대리 점수이며 실험 affinity가 아니다. Δ=출력−시작 pose, 음수가 낮은 점수다.

| 방법 | subset | pairs/pockets | 전체 pair Δ 중앙값 | 낮아진 pairs | pocket 중앙값 [95% CI] |
| --- | --- | --- | --- | --- | --- |
| UFF whole | all_scored | 567/19 | +2.506 | 69/567 (12.2%) | +2.553 [+0.938, +6.035] |
| UFF whole | both_graph_preserved | 514/19 | +2.302 | 66/514 (12.8%) | +2.553 [+0.938, +6.035] |
| UFF whole | both_strict_success | 126/13 | +0.779 | 20/126 (15.9%) | +0.546 [+0.312, +1.378] |
| UFF core | all_scored | 567/19 | -0.059 | 324/567 (57.1%) | -0.014 [-1.147, +0.056] |
| UFF core | both_graph_preserved | 549/19 | -0.055 | 311/549 (56.6%) | -0.014 [-0.137, +0.057] |
| UFF core | both_strict_success | 222/14 | -0.016 | 121/222 (54.5%) | -0.003 [-0.085, +0.043] |
| FM whole | all_scored | 567/19 | -1.725 | 526/567 (92.8%) | -2.585 [-3.613, -0.813] |
| FM whole | both_graph_preserved | 498/18 | -1.934 | 458/498 (92.0%) | -2.125 [-4.659, -0.882] |
| FM whole | both_strict_success | 203/15 | -0.767 | 175/203 (86.2%) | -0.790 [-1.240, -0.448] |
| FM core | all_scored | 567/19 | -1.326 | 498/567 (87.8%) | -2.236 [-4.211, -0.421] |
| FM core | both_graph_preserved | 510/19 | -1.264 | 450/510 (88.2%) | -1.258 [-4.211, -0.410] |
| FM core | both_strict_success | 199/15 | -0.154 | 159/199 (79.9%) | -0.421 [-1.250, -0.037] |

all_scored는 PB 또는 stereo 실패도 포함한다. both_graph_preserved는 양쪽 graph/stereo가 맞는 쌍, both_strict_success는 양쪽 모두 물리/조건 검사까지 통과한 더 작은 조건부 집합이다. 이 필터 결과는 별도의 전체 성능이 아니다. 한 pocket 안의 8개 pose를 독립 표본으로 가정하지 않도록 pocket 중앙값과 pocket 재표집 5,000회 CI를 함께 기록했다. 현재 부분 cohort의 탐색적 불확실성 범위다.

## 화학 편집이 원본 graph보다 나았는가

아래는 동일 pocket·FM 조건에서 graph별 종합 성공 pose들의 Vina 평균을 만든 후, 편집 graph 평균−원본 graph 평균을 비교한 것이다. 두 graph에 유효 후보가 있는 pocket만 사용한다. graph 간 초기 seed가 다르고 성공 subset 크기도 다르므로 단일 원자 편집의 인과 효과나 공정한 고정 성공률 비교로 해석하지 않는다.

| 조건 | 편집 | pockets | Δ Vina 중앙값 | 낮아진 pocket | 95% CI |
| --- | --- | --- | --- | --- | --- |
| FM core | C_to_F | 16 | +0.015 | 5/16 | [-0.064, +0.078] |
| FM core | C_to_N | 16 | +0.013 | 7/16 | [-0.118, +0.304] |
| FM core | C_to_O | 15 | +0.102 | 5/15 | [-0.156, +0.210] |
| FM whole | C_to_F | 17 | +0.059 | 7/17 | [-0.084, +0.219] |
| FM whole | C_to_N | 17 | +0.129 | 6/17 | [-0.119, +0.309] |
| FM whole | C_to_O | 15 | +0.226 | 5/15 | [-0.089, +0.318] |

![Paired Vina](/home/uslee/code/DIRMO-3D/reports/fm-edited-graph-20261018-v1/paired_vina.png)

원본 실험 ligand와의 score 비교는 generated_original_vs_native_vina.csv에 있다. 원본과 같은 graph의 성공 pose만 사용하며, 비교 가능한 native도 PB 통과한 사례로 제한했다.

- FM core 원본 graph vs 실험 native: 15 pockets, pocket별 평균 Δ의 중앙값 +0.173 kcal/mol, 낮아진 pocket 4/15.
- FM whole 원본 graph vs 실험 native: 16 pockets, pocket별 평균 Δ의 중앙값 -0.045 kcal/mol, 낮아진 pocket 10/16.

## 2D 화학 특성

| 편집 | 평균 Δqed | 평균 Δsa | 평균 Δlogp | 평균 Δhbd | 평균 Δhba | 평균 Δtpsa |
| --- | --- | --- | --- | --- | --- | --- |
| C_to_N | -0.072 | +0.193 | -1.087 | +1.000 | +0.900 | +26.020 |
| C_to_O | -0.073 | +0.166 | -0.990 | +1.000 | +0.900 | +20.230 |
| C_to_F | -0.030 | +0.242 | -0.106 | +0.000 | +0.000 | -0.000 |

평가 완료 30개 pocket에서 graph 편집 직후의 평균 변화다. QED·SA·RDKit LogP 변화는 C→N/O/F 편집으로 결정되며 FM 재생성이 만들어낸 변화가 아니다. 동일 graph/stereo가 보존되면 이들 descriptor는 재생성 전후 동일하다. LogP를 낮추는 방향과 QED/SA를 개선하는 방향은 항상 같지 않다. 목표 범위를 미리 주거나 후보를 점수로 골라 만든 편집이 아니어서 평균 편집 이득을 보장할 수 없다. pKa·용해도·실험 affinity는 이번에 측정하지 않았다. 전체 선정 100개 graph의 특성 변화도 chemical_change_summary.json에 별도로 기록했다.

![Chemical changes](/home/uslee/code/DIRMO-3D/reports/fm-edited-graph-20261018-v1/chemical_changes.png)

## 해석과 다음 단계

1. 이 부분 결과는 **명시적 편집 graph를 유지하면서 pocket 안 좌표를 개선할 수 있음**을 지지한다. 시작 pose 대비 FM의 protein clash와 Vina score가 개선됐다.
2. core 좌표 hard hold는 실제 작동했다. 그러나 stereo와 경계 기하 품질은 별개이며, 전체 자유 pose보다 종합 성공률은 낮았다. hard condition 준수만으로 좋은 생성이 보장되지 않는다.
3. 어떤 화학 편집이 목표를 개선하는지는 아직 증명하지 못했다. N/O/F 모두 임의의 고정 규칙이며, LLM 목표 설정이나 후보 랭킹 효과를 평가한 것이 아니다. 편집 후 물성·Vina의 상충을 함께 봐야 한다.
4. 먼저 입력 적격성에 radical/수소·graph round-trip 사전 검사를 추가하고, 개별 pocket의 입력 실패가 cohort 전체를 중단시키지 않도록 복구 경로를 설계해야 한다. 5edl을 원본대로 보존/명시적 unsupported 처리할지 결정하고, 같은 선정 100개와 seed를 유지해 미완료 부분을 재개하는 것이 다음 우선순위다. 이번 분석에서는 재생성이나 누락 평가를 새로 돌리지 않았다.
5. 100개 완료 후 전체 비교와 pocket별 paired 분석을 다시 해야 한다. core의 실패 원자가 고정/자유 경계에 몰리는지도 점검하고, 결과가 유지되면 그 뒤 목표 기반 graph 후보 선택 실험으로 진행한다.

## 파일

- analysis.json: 완료/실패 상태와 pooled 통계, paired binary 차이 및 pocket bootstrap CI.
- evaluated_rows.csv, group_metrics.csv: 완결 30개 기준 전체 행과 조건별 통계.
- independent_graph_coordinate_audit.csv, condition_audit.csv: 전체 저장 SDF와 condition 독립 검증.
- paired_vina_*_vs_prior.csv, paired_vina_summary.json: 동일 시작 pose와 paired score 비교.
- edited_vs_original_successful_pose_vina.csv, edited_vina_summary.json: 같은 pocket에서 graph 편집 비교.
- chemical_properties_all_100.csv, chemical_changes.csv: graph별 2D descriptor와 변화.
- pb_failure_breakdown.csv, receptor_availability.csv: 물리 검사 실패 및 receptor 누락.
- controller_exit_snapshot: 중단 당시의 옛 summary/REPORT/metrics. 원본 결과와 오류는 실험 run에 유지.
