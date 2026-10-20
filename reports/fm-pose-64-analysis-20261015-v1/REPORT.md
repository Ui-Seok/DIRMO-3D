# FLOWR 64-step 부분 pose 복구 결과 분석

**중간 분석: 평가가 종료된 1,430/2,000개 복합체를 고정하여 비교했다.**

현재 완료 사례에서는 FLOWR base가 v2 및 ligand-only UFF보다 물리 검사 통과와 종합 복구 성공에서 우세하다. 이는 화학 구조 편집이나 결합 친화도 개선에 대한 결과가 아니다.

분석 시작: 2026-10-15T13:06:54.648164+09:00; 스캔 완료: 2026-10-15T13:07:16.909988+09:00.

## 실험과 분모

전체 원소·전하·결합 그래프를 조건으로 고정하고, 결정적으로 선택한 2–8개 말단 원자의 좌표만 바꿨다. 원본에 torsion 오차 small 20–40도 / large 50–70도를 주고 복구했다. 각 수준의 16개는 서로 다른 흐트러뜨린 입력이며, 같은 입력의 독립적인 16회 생성이 아니다. 두 모델은 정확히 같은 입력을 받았다. 64는 ODE integration step 수다.

입체화학은 별도 고정 tensor가 없으므로 생성 XYZ에서 계산해 원본 stereo와 일치하는지 검사한다. LLM 호출·추가 학습·Vina/친화도 목표는 사용하지 않았다. base는 FLOWR checkpoint 이름이며 initial ligand가 아니다.

표의 각 방법·오차 수준 분모는 22,880개다. 실패·누락을 분모에서 빼지 않았다. 진행 중 사례의 일부 결과는 배제해 같은 완료 복합체·같은 입력으로 비교한다. 완료 여부에 따른 선택 편향은 남는다.

## 전체 2,000개 생성 처리 점검

| 모델 | 예정 | SDF 저장 | 계약 통과 | 계약 불일치 | 전처리 실패 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 64,000 | 62,656 | 61,653 | 1,003 | 1,344 |
| v2 | 64,000 | 62,656 | 62,380 | 276 | 1,344 |

두 모델의 공통 전처리 실패는 42개 복합체다. 37개: CapabilityError: Tensorization changed the graph or atom order / 5개: KeyError: np.str_('SE'). `128,000 처리 완료`는 `128,000개 정상 분자 생성`을 뜻하지 않는다. 실패 기록은 지우거나 쉬운 사례로 교체하지 않았다.

전체 생성 배치의 보호 좌표 tensor 최대 drift: base 0.0 Å, v2 0.0 Å. Stereo를 제외한 graph 변경 여부 및 저장 SDF의 drift 검사는 summary.json에 별도 기록했다.

## 원본·입력·복구 후 물리 검사

원본 ligand: pocket 87.90%, 전체 protein 84.83% 통과. 원본부터 물리 검사를 통과하지 못한 사례도 포함했다.

| 오차 | 방법 | Pocket 통과 | 전체 protein 통과 | 종합 성공 |
| --- | --- | ---: | ---: | ---: |
| small | input | 59.20% | 54.10% | — |
| small | UFF | 61.98% | 57.28% | 44.45% |
| small | base | 81.58% | 74.31% | 65.36% |
| small | v2 | 64.19% | 58.44% | 41.47% |
| large | input | 36.17% | 31.00% | — |
| large | UFF | 51.77% | 46.66% | 39.86% |
| large | base | 68.89% | 59.23% | 57.28% |
| large | v2 | 41.22% | 35.21% | 26.56% |

종합 성공 = 원본 stereo까지 graph 보존 + 보호 좌표 drift ≤0.001 Å + pocket PoseBusters 통과 + 입력보다 자유 영역 RMSD 감소. 입력 자체는 자기 자신보다 개선될 수 없어 종합 성공 비교에서 기준선으로만 쓴다. 이것은 논문에서 가져온 단일 utility 점수가 아니라 이 pose 복구 실험의 명시적인 조건 결합이다.

전체 protein 검사는 물·보조인자까지 포함하여 고정 pocket 검사보다 엄격하다. UFF는 ligand-only이며 보호 좌표는 고정하지만 protein 에너지 항은 사용하지 않는다.

## RMSD: 모든 방법이 graph/보호 좌표를 보존한 공통 입력에서 비교

| 오차 | 공통 입력 수 | input | UFF | base | v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 21,910 | 0.885 Å | 0.760 Å | 0.562 Å | 0.861 Å |
| large | 21,663 | 1.710 Å | 1.273 Å | 1.123 Å | 1.678 Å |

RMSD는 protein frame을 유지한 자유 영역 값이다. 전체 ligand를 정답에 정렬하지 않는다. 보호 원자의 대응을 유지하는 graph symmetry를 고려한다. 이 표는 공통으로 비교 가능한 입력에 조건부인 값이므로, graph 실패까지 포함한 전체 복구율은 종합 성공률로 함께 판단해야 한다.

## 입력 상태별 복구와 훼손

| 오차 | 방법 | 물리 검사 실패 입력 → 통과 | 통과 입력 → 실패/누락 |
| --- | --- | ---: | ---: |
| small | UFF | 37.23% | 20.97% |
| small | base | 66.65% | 8.13% |
| small | v2 | 22.81% | 7.29% |
| large | UFF | 35.05% | 18.70% |
| large | base | 56.78% | 9.73% |
| large | v2 | 13.19% | 9.32% |

## 같은 복합체에서의 차이와 불확실성

각 복합체의 16개 입력을 먼저 평균하고 exact protein sequence ID로 묶어 paired bootstrap을 수행했다. 32개 입력을 독립적인 단백질처럼 세지 않았다. 아래 구간은 현재 완료 cohort에 조건부인 탐색적 95% bootstrap 구간이다. Scaffold grouping 결과도 JSON에 저장했다.

| 오차 | 비교 | 지표 | 차이(%p) | Protein cluster bootstrap 95% |
| --- | --- | --- | ---: | --- |
| small | base-v2 | pocket_valid | +17.39 | [+15.26, +19.61] |
| small | base-UFF | pocket_valid | +19.60 | [+17.19, +21.92] |
| small | base-v2 | primary_success | +23.89 | [+21.38, +26.35] |
| small | base-UFF | primary_success | +20.91 | [+18.31, +23.63] |
| large | base-v2 | pocket_valid | +27.67 | [+25.04, +30.15] |
| large | base-UFF | pocket_valid | +17.12 | [+14.91, +19.37] |
| large | base-v2 | primary_success | +30.72 | [+27.92, +33.48] |
| large | base-UFF | primary_success | +17.42 | [+15.15, +19.74] |

## 잘 안 되는 부분

Physical validity 통과와 원본 pose 복구는 다르다. 유효한 다른 conformation을 만들면 물리 검사에는 통과해도 RMSD 복구에 실패할 수 있다. 반대로 RMSD가 줄어도 잘못된 stereo 또는 원자 겹침이 있으면 종합 성공이 아니다.

v2는 graph/stereo 계약 위반은 적지만 이번 결과의 종합 복구 성공률은 base보다 낮다. Base도 모든 입력을 개선하지 않으므로 통과 입력을 나쁘게 만드는 비율과 UFF 대비 차이를 함께 확인해야 한다.

Base의 물리 검사 통과율 향상은 주로 protein과의 접촉 충돌 감소와 함께 나타난다. 그러나 자유 영역의 결합 길이·각도·내부 에너지 오류가 늘어난 경우도 있다. Graph의 결합 종류를 고정하는 것과 3D 결합 길이를 올바르게 유지하는 것은 서로 다른 조건이다. 검사별 수치는 FAILURE_ANALYSIS.md에서 확인한다.

## 해석의 한계

미완료 570개 복합체는 이 비교에 포함하지 않았다. GPU는 큰 pocket부터 처리하고 CPU도 완료 순서에 영향을 받으므로 완료 cohort를 무작위 표본으로 해석하면 안 된다. 구조·단백질·scaffold 분포는 cohort JSON에 저장했다.

동일 protein/scaffold의 서로 다른 복합체가 포함돼 있다. Exact sequence grouping은 서열 유사도 clustering과 다르며, FM 사전학습 데이터 중복도 미확인이다. 독립적인 unseen-target 일반화나 ligand affinity 개선을 주장할 수 없다.

현재 조건의 작은 말단 영역 pose 복구에서는 base를 우선 후보로 둘 근거가 있다. 128·256-step에서도 같은 완료 복합체를 맞춰 물리 검사·stereo 보존·종합 성공·NFE를 비교한 뒤 step 수의 효과를 판단해야 한다. 이 결과만으로 FM 재학습이 필요하거나 불필요하다고 단정할 수 없다.

## 산출물

- `summary.json`: 집계, 분모, 공통 RMSD, paired bootstrap, 생성 실패 감사 결과.
- `cohort.json`: 완료/대기 사례와 읽은 평가 파일 SHA-256.
- `metrics.csv.gz`: 현재 cohort의 입력별 요약 측정값.
- `case_rates.csv`: 복합체별·오차별·방법별 성공 비율.
- `failure_checks.json`: 검사별 실패 수(한 분자가 여러 검사에 실패할 수 있음).
- `figures/overview.{png,pdf}` 및 `figures/rmsd_ecdf.{png,pdf}`: 고정된 snapshot의 그림.

분석 코드: `scripts/analyze_pose_repair.py`. 이 분석은 저장된 JSON/SDF만 읽으며 실행 중인 생성·평가 프로세스나 조건을 변경하지 않는다.

## 저장된 출력의 stereo 검사

- base: stereo만 불일치한 저장 출력 1,003/62,656개 (1.60%).
- v2: stereo만 불일치한 저장 출력 276/62,656개 (0.44%).

원소·연결·결합 종류와 보호 좌표 조건을 유지해도 자유 영역의 3D stereo는 자동 보장되지 않는다. 잘못된 stereo 출력은 종합 성공으로 세지 않았다.

## 개별 사례와 최종 분석

설명용 성공·실패 사례는 `EXAMPLES.md`에 정리했다. 나머지 CPU 평가가 모두 끝나면 `reports/fm-pose-64-analysis-20261015-final/REPORT.md`를 자동 생성한다. 현재 보고서는 1,430개 snapshot을 유지한다.
