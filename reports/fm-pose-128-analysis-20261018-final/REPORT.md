# FLOWR 64-step 부분 pose 복구 결과 분석

**최종 분석: 평가가 종료된 2,000/2,000개 복합체를 고정하여 비교했다.**

현재 완료 사례에서는 FLOWR base가 v2 및 ligand-only UFF보다 물리 검사 통과와 종합 복구 성공에서 우세하다. 이는 화학 구조 편집이나 결합 친화도 개선에 대한 결과가 아니다.

분석 시작: 2026-10-18T16:34:45.851213+09:00; 스캔 완료: 2026-10-18T16:35:40.227267+09:00.

## 실험과 분모

전체 원소·전하·결합 그래프를 조건으로 고정하고, 결정적으로 선택한 2–8개 말단 원자의 좌표만 바꿨다. 원본에 torsion 오차 small 20–40도 / large 50–70도를 주고 복구했다. 각 수준의 16개는 서로 다른 흐트러뜨린 입력이며, 같은 입력의 독립적인 16회 생성이 아니다. 두 모델은 정확히 같은 입력을 받았다. 64는 ODE integration step 수다.

입체화학은 별도 고정 tensor가 없으므로 생성 XYZ에서 계산해 원본 stereo와 일치하는지 검사한다. LLM 호출·추가 학습·Vina/친화도 목표는 사용하지 않았다. base는 FLOWR checkpoint 이름이며 initial ligand가 아니다.

표의 각 방법·오차 수준 분모는 32,000개다. 실패·누락을 분모에서 빼지 않았다. 진행 중 사례의 일부 결과는 배제해 같은 완료 복합체·같은 입력으로 비교한다. 완료 여부에 따른 선택 편향은 남는다.

## 전체 2,000개 생성 처리 점검

| 모델 | 예정 | SDF 저장 | 계약 통과 | 계약 불일치 | 전처리 실패 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 64,000 | 62,656 | 61,462 | 1,194 | 1,344 |
| v2 | 64,000 | 62,656 | 61,749 | 907 | 1,344 |

두 모델의 공통 전처리 실패는 42개 복합체다. 37개: CapabilityError: Tensorization changed the graph or atom order / 5개: KeyError: np.str_('SE'). `128,000 처리 완료`는 `128,000개 정상 분자 생성`을 뜻하지 않는다. 실패 기록은 지우거나 쉬운 사례로 교체하지 않았다.

전체 생성 배치의 보호 좌표 tensor 최대 drift: base 0.0 Å, v2 0.0 Å. Stereo를 제외한 graph 변경 여부 및 저장 SDF의 drift 검사는 summary.json에 별도 기록했다.

## 원본·입력·복구 후 물리 검사

원본 ligand: pocket 88.40%, 전체 protein 85.15% 통과. 원본부터 물리 검사를 통과하지 못한 사례도 포함했다.

| 오차 | 방법 | Pocket 통과 | 전체 protein 통과 | 종합 성공 |
| --- | --- | ---: | ---: | ---: |
| small | input | 61.09% | 55.43% | — |
| small | UFF | 63.10% | 58.11% | 44.55% |
| small | base | 76.26% | 68.69% | 55.01% |
| small | v2 | 62.67% | 56.30% | 35.22% |
| large | input | 38.91% | 32.88% | — |
| large | UFF | 54.32% | 48.79% | 40.19% |
| large | base | 66.47% | 56.87% | 53.21% |
| large | v2 | 42.92% | 36.12% | 25.85% |

종합 성공 = 원본 stereo까지 graph 보존 + 보호 좌표 drift ≤0.001 Å + pocket PoseBusters 통과 + 입력보다 자유 영역 RMSD 감소. 입력 자체는 자기 자신보다 개선될 수 없어 종합 성공 비교에서 기준선으로만 쓴다. 이것은 논문에서 가져온 단일 utility 점수가 아니라 이 pose 복구 실험의 명시적인 조건 결합이다.

전체 protein 검사는 물·보조인자까지 포함하여 고정 pocket 검사보다 엄격하다. UFF는 ligand-only이며 보호 좌표는 고정하지만 protein 에너지 항은 사용하지 않는다.

## RMSD: 모든 방법이 graph/보호 좌표를 보존한 공통 입력에서 비교

| 오차 | 공통 입력 수 | input | UFF | base | v2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 30,483 | 0.827 Å | 0.748 Å | 0.630 Å | 0.866 Å |
| large | 30,034 | 1.584 Å | 1.240 Å | 1.065 Å | 1.593 Å |

RMSD는 protein frame을 유지한 자유 영역 값이다. 전체 ligand를 정답에 정렬하지 않는다. 보호 원자의 대응을 유지하는 graph symmetry를 고려한다. 이 표는 공통으로 비교 가능한 입력에 조건부인 값이므로, graph 실패까지 포함한 전체 복구율은 종합 성공률로 함께 판단해야 한다.

## 입력 상태별 복구와 훼손

| 오차 | 방법 | 물리 검사 실패 입력 → 통과 | 통과 입력 → 실패/누락 |
| --- | --- | ---: | ---: |
| small | UFF | 37.67% | 20.70% |
| small | base | 62.17% | 14.77% |
| small | v2 | 28.55% | 15.60% |
| large | UFF | 36.57% | 17.82% |
| large | base | 56.13% | 17.29% |
| large | v2 | 17.48% | 17.15% |

## 같은 복합체에서의 차이와 불확실성

각 복합체의 16개 입력을 먼저 평균하고 exact protein sequence ID로 묶어 paired bootstrap을 수행했다. 32개 입력을 독립적인 단백질처럼 세지 않았다. 아래 구간은 현재 완료 cohort에 조건부인 탐색적 95% bootstrap 구간이다. Scaffold grouping 결과도 JSON에 저장했다.

| 오차 | 비교 | 지표 | 차이(%p) | Protein cluster bootstrap 95% |
| --- | --- | --- | ---: | --- |
| small | base-v2 | pocket_valid | +13.59 | [+11.53, +15.59] |
| small | base-UFF | pocket_valid | +13.16 | [+10.81, +15.44] |
| small | base-v2 | primary_success | +19.79 | [+17.42, +22.09] |
| small | base-UFF | primary_success | +10.46 | [+7.93, +12.84] |
| large | base-v2 | pocket_valid | +23.56 | [+21.32, +25.90] |
| large | base-UFF | pocket_valid | +12.16 | [+10.00, +14.33] |
| large | base-v2 | primary_success | +27.35 | [+24.97, +29.80] |
| large | base-UFF | primary_success | +13.01 | [+10.87, +15.14] |

## 잘 안 되는 부분

Physical validity 통과와 원본 pose 복구는 다르다. 유효한 다른 conformation을 만들면 물리 검사에는 통과해도 RMSD 복구에 실패할 수 있다. 반대로 RMSD가 줄어도 잘못된 stereo 또는 원자 겹침이 있으면 종합 성공이 아니다.

v2는 graph/stereo 계약 위반은 적지만 이번 결과의 종합 복구 성공률은 base보다 낮다. Base도 모든 입력을 개선하지 않으므로 통과 입력을 나쁘게 만드는 비율과 UFF 대비 차이를 함께 확인해야 한다.

Base의 물리 검사 통과율 향상은 주로 protein과의 접촉 충돌 감소와 함께 나타난다. 그러나 자유 영역의 결합 길이·각도·내부 에너지 오류가 늘어난 경우도 있다. Graph의 결합 종류를 고정하는 것과 3D 결합 길이를 올바르게 유지하는 것은 서로 다른 조건이다. 검사별 수치는 FAILURE_ANALYSIS.md에서 확인한다.

## 해석의 한계

예정한 2,000개 복합체의 평가가 종료됐다. 평가 누락 및 전처리 실패는 summary.json에서 별도로 확인해야 한다. 구조·단백질·scaffold 분포는 cohort JSON에 저장했다.

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

- base: stereo만 불일치한 저장 출력 1,194/62,656개 (1.91%).
- v2: stereo만 불일치한 저장 출력 907/62,656개 (1.45%).

원소·연결·결합 종류와 보호 좌표 조건을 유지해도 자유 영역의 3D stereo는 자동 보장되지 않는다. 잘못된 stereo 출력은 종합 성공으로 세지 않았다.
