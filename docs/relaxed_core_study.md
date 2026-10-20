# 고정 경계 한 결합 완화 비교

2026-10-18 GPU4에서 `runs/fm_relaxed_core_gpu4_100_20261018_v1`로 실행한다. GPU3의 기존 실험은 그대로 계속한다.

기존 100개 pocket, 원본/N/O/F graph 4개, graph당 시작 pose 8개를 그대로 사용한다. 전체 graph는 고정한 채 자유 원자 집합의 한 결합 이웃까지 좌표 자유 영역을 확장한다. 이 cohort에서는 모두 기존 고정 경계 원자 1개만 추가로 풀리고, 고정 원자는 최소 3개 남는다. PB/성능을 보고 mask를 선택하지 않는다. FLOWR.root base, 64-step, 최대 homogeneous batch8이며 OOM 시 축소한다. 추가 FM 예정 결과는 3,200개다.

시작 SDF 경로·내용 hash·seed를 기존 실험과 대조한다. 새로운 ETKDG, 좌표 재배치, graph 편집을 수행하지 않는다. 입력 초기화 실패와 radical unsupported 입력 5edl도 100개 cohort/3,200 예정 분모에 남긴다. 기존 core 고정과 전체 좌표 자유 결과는 재생성하지 않고 재사용한다. 같은 완화 mask를 쓰는 ligand-only UFF200을 CPU 대조군으로 추가한다.

종합 성공은 기존과 같은 PB dock20 전체 통과, target graph/입체화학 보존, 이번에 고정하도록 요청한 나머지 core 좌표의 최대 이동 ≤0.001 Å를 요구한다. 풀어준 원자의 이동은 고정 위반으로 세지 않는다. 원래 core 전체 이동과 풀어준 원자의 이동을 별도로 저장해 core 위치를 얼마나 양보했는지 확인한다. 시작 pose에는 원래 core 위치 정보가 있으므로 blind docking 비교가 아니다.

CPU PB 50-conformer reference/measurement cache는 기존 run과 잠금 및 입력 identity key로 공유한다. Vina receptor는 기존 완결 결과가 있으면 복사하고, 없으면 동일한 준비 코드/설정으로 별도 생성한다. 동일 수용체 hash의 쌍에서만 Vina paired 변화를 해석한다. 새로 편집한 분자의 실험 pose가 없으므로 native 전체 RMSD를 정확도로 사용하지 않는다.

`scripts/benchmark_relaxed_core.py`는 기존 controller/생성/평가 코드를 별도 process에서 재사용하며 이 run의 단일 `core` arm을 **완화된 core**로 정의한다. GPU3 process의 arm이나 mask는 바꾸지 않는다. CPU worker와 GPU worker는 모두 같은 wrapper로 진입한다. `study.json`에 전체 mask·code hash를 고정하며, `condition.json`의 실제 tensor mask와 최종 고정 좌표를 검사한다.

`REPORT.md`와 `summary.json`은 진행 상황과 두 실험 모두 평가가 끝난 동일 pocket의 original_core / relaxed_core / whole 비교를 갱신한다. 실행 중 표는 부분 결과다. GPU4가 먼저 끝나면 모델 process는 종료하고 GPU3의 대조군 평가 완료를 기다려 최종 paired CSV와 평가 파일 hash를 저장한다. 새로운 학습이나 LLM 호출은 없다.
