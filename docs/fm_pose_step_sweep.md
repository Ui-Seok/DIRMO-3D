# GPU 3·4 pose 복구 / 배치 32 / 2,000 복합체 실험

실행과 최신 상태는 `runs/fm_pose_gpu4_2000_batch32_v1/progress.json`, 전체 진행 및 step 비교는 같은 폴더의 `REPORT.md`에 기록한다. 기존 결과와 입력 파일 경로를 유지하기 위해 폴더 이름은 그대로 둔다. 상위 제어기는 백그라운드에서 실행되며 GPU 3과 4를 사용한다. 각 step의 base와 v2는 동일한 GPU에서 병행한다.

## 입력 및 생성 조건

- 서로 다른 protein–ligand 복합체 2,000개를 PDBbind2019 refined-set과 v2019-other-PL에서 선정한다. 동일 단백질 또는 scaffold의 다른 복합체도 허용하고 ID와 데이터 출처를 저장한다. 따라서 2,000개의 독립적인 단백질에 대한 일반화 검증은 아니다.
- 원본 XYZ를 보존한다. 원본의 미지정 입체화학은 그 XYZ에서 정의하고, 명시된 입체화학이 XYZ와 충돌하는 사례는 제외한다. 구조 오류 및 제외 사유는 `steps_64/eligibility-exclusions.json`에 저장한다. 물리적 타당성은 선정 조건으로 사용하지 않는다.
- metric으로 영역을 고르지 않는다. 회전 가능한 말단의 2–8개 heavy atom 영역을 결정적으로 선택한다. 전체 원소·전하·결합 그래프를 고정하고, 자유 영역 바깥의 좌표만 고정한다.
- 자유 영역은 small(20–40도) / large(50–70도) torsion으로 흐트러뜨린다. 수준마다 seed 16개를 사용해 복합체당 32개 입력을 저장한다. 두 모델과 64·128·256 step이 정확히 같은 입력 파일을 사용한다.
- 모델에는 흐트러뜨린 입력만 제공한다. 정답 자유 영역 좌표는 평가에만 사용한다. LLM 호출과 추가 학습은 없다.

## 배치 및 실행 순서

배치는 최대 32이며 한 복합체의 서로 다른 입력으로 구성한다. 혼합 크기 batch에서는 단독 생성과 좌표 차이를 관측하여 대규모 결과에 사용하지 않는다. 큰 pocket은 예상 메모리에 따라 배치를 낮추고 OOM이면 해당 배치를 더 줄여 재시도한다. 실제 배치 크기, GPU UUID, 최대 메모리와 forward 횟수는 `phases/{base,v2}/batches.jsonl`에 저장한다.

기존 50개의 배치 32 확인은 완료됐다. 2026-10-14에 GPU 3을 추가해 진행 중인 64-step은 GPU 4에서 유지하고 128-step을 GPU 3에서 시작했다. 256-step은 먼저 GPU 생성이 끝나는 장치에서 실행한다. CPU 평가는 병행하며 모든 예정 작업과 CPU 평가가 끝나야 전체 완료다.

실행 중인 제어기는 PID와 실제 실행 명령을 확인해 인계받는다. 다른 GPU를 추가하려고 같은 step의 모델을 중복 실행하지 않는다. 상위 제어기의 기본값은 GPU 4 단독이며, 두 장치 설정은 `--gpus 4 3`이다. 기존 GPU 4 모델 프로세스는 GPU 3 추가 과정에서 재시작하지 않았다. 인계 정보는 `scheduler-handoff.json`에 저장한다.

2,000개 × 32개 입력 × 2개 모델 × 3개 step = **384,000개 생성 결과**다. 모델 forward가 step마다 한 번 및 최종 한 번이면 예정 sample NFE는 **57,728,000**이다. OOM 재시도에서 발생한 추가 연산은 별도로 기록한다. 데이터 크기에 따라 실행 시간은 크게 달라지며 전체는 수일 규모일 수 있다.

## 물리 평가 및 결과 해석

원본, 흐트러뜨린 입력, FM의 raw 출력에 동일한 PoseBusters dock 검사를 적용한다. 고정 pocket과 원본 전체 protein(물 및 보조인자 포함)을 각각 검사한다. 전체 protein을 한 번 읽어 캐시하지만 원자나 검사 항목은 줄이지 않는다. step 사이에서 동일한 원본·입력·UFF 대조군의 측정값은 재사용한다.

큰 전체 protein의 volume 검사에서는 upstream과 같은 원자를 같은 순서로 삭제하되 RDKit batch edit로 인덱스 갱신을 한 번만 수행한다. 판정에 남는 원자·좌표·임계값·shape 계산은 유지한다. upstream 파일은 수정하지 않으며 작은 분자와 큰 protein에서 기존 결과와의 일치를 검증한다.

추가로 graph/stereo 보존, 보호 좌표의 최대 drift, protein frame에서의 자유 영역 RMSD를 측정한다. RMSD는 보호 원자의 대응을 유지하는 대칭을 고려하며 정답에 ligand를 정렬하지 않는다. 종합 성공은 graph 보존, 보호 drift ≤0.001 Å, pocket 물리 검사 통과, 입력보다 RMSD 개선을 모두 요구한다.

단순 input 유지 및 ligand-only UFF를 대조군으로 둔다. UFF는 보호 좌표를 고정하지만 protein 에너지 항을 사용하지 않는다. 원본부터 물리 검사에 실패하는 사례도 유지한다. 생성 실패·측정 누락은 통과로 집계하지 않는다.

## 코드 및 로그

- `scripts/run_fm_pose_step_sweep.py`: GPU별 작업 배치, 진행 중 제어기 인계, 추가 입력 생성, 단계 간 입력 공유, 진행 보고서와 step 비교.
- `scripts/benchmark_fm_pose_repair.py`: 데이터 선정, GPU 모델별 실행, CPU 물리 평가와 대조군, 단계별 집계.
- `src/dirmo/generation/pose_repair.py`: graph 조건과 좌표 조건을 분리한 FLOWR pose 생성.
- `src/dirmo/evaluation/metrics.py`: PoseBusters 검사 결과 정리와 이미 읽은 protein 객체 입력 지원.
- `reports/fm-pose-gpu3-gpu4-2000-sweep-v1.log`: GPU 3·4 상위 제어기 로그. 이전 GPU 4 단독 실행 로그도 보존한다.
- `runs/fm_pose_gpu4_2000_batch32_v1/{batch32_50_case_confirmation,steps_64,steps_128,steps_256}`: 단계별 결과와 로그.

각 단계의 `controller.json`과 `summary.json`에 상태를 기록한다. `complete_with_missing_evaluation`은 생성 완료와 별개로 일부 물리 평가가 누락된 상태이며, 전체 측정 성공을 뜻하지 않는다. `evaluations/*.log`, `phases/*/generation.log`, 상위 `error.json`에서 실제 오류를 확인한다.
