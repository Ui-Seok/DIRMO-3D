# FLOWR 생성물의 PB 진단 기반 부분 재생성

실행 폴더: `/home/uslee/code/DIRMO-3D/runs/fm_pb_guided_gpu4_20261015_v1`.

GPU4의 기존 256-step 생성 worker 두 개만 SIGSTOP/SIGCONT로 전환한다. 공용 scheduler와 GPU3의 128-step은 유지한다. 새 실험이 CPU PB 입력을 기다릴 때는 256-step을 실행하고, 복구 입력이 준비되면 기존 worker의 PID·명령을 검증하고 메모리 여유를 확인한 뒤 새 실험에 우선권을 준다. SIGSTOP은 CUDA 메모리를 반환하지 않으므로 남은 공간에서 새 생성 배치를 자동 조절한다. 새 GPU 단계가 종료돼도 CPU PB 평가는 계속할 수 있다.

## 과제와 고정 조건

기존 PDBbind2019 적격 복합체 2,000개 pool에서 seed 20261015로 섞고 exact protein sequence가 중복되지 않게 100개를 선택했다. 결과를 본 뒤 쉬운 pocket으로 교체하지 않는다. 이는 공식 SPINDR test set이나 FM 사전학습과 분리된 benchmark라는 뜻이 아니다. 사전학습 중복은 미확인이다.

FLOWR base checkpoint로 pocket마다 16개를 64 integration steps/Euler로 생성한다. 초기 생성에서는 ligand의 원소·전하·결합·좌표를 고정하지 않는다. Reference ligand는 pocket 위치와 원자 수를 정하는 데 사용한다. 원본 graph를 고정한 pose 생성과 구분한다. 최종 graph overwrite, valence repair, relaxation, oversampling/filtering은 하지 않는다. 모델이 반환하지 못한 슬롯도 전체 1,600개 분모에 남긴다.

초기 raw SDF에 고정 6 Å pocket의 PoseBusters dock 설정을 적용한다. 선택된 20개 이진 검사 전체가 True여야 통과한다. 이번 주 지표에는 전체 protein/water/cofactor 재검사가 포함되지 않는다. 원본 전체 protein 경로는 후속 검사에 사용할 수 있도록 보관한다.

## PB 실패 위치와 영역

PoseBusters가 실제 사용한 module 함수의 반환값을 관찰하여 원자쌍·결합·각도·평면성 위반을 추출한다. 임계값과 판정 결과를 변경하지 않는다. Protein 접촉 실패는 ligand 원자로, 결합·각도 실패는 관련 원자들로, 평면성 실패는 해당 원자 그룹으로 연결한다. 내부 에너지 등 전역 실패만 있는 경우 fragment 점수를 임의로 만들지 않고 지역 복구 대상에서 제외한다. 제외된 실패도 초기 생성물 전체 분모에는 유지한다.

후보 영역은 non-ring single bond 절단으로 얻은 연결 fragment와 ring을 온전히 포함하는 이웃 영역이다. 최소 2개, 최대 8–12개 heavy atom 범위에서 원본 전체보다 작은 영역을 사용한다. 가장 많은 실패 event를 포함하고, 위반 심각도 합이 크며, 작은 영역을 우선한다. 이는 인과적으로 증명된 문제 fragment나 독립적인 fragment PB 점수가 아니라 이 실험의 명시적 위치 선택 규칙이다. 하나의 event가 여러 원자를 포함할 때 그중 일부만 자유 영역에 들어갈 수 있으며 실제로 해결되는지는 생성 뒤 다시 검사한다.

무작위 대조군은 같은 원자 수와 같은 경계 결합 수의 후보 영역에서 고른다. 같은 영역을 뽑거나 대안이 없는 경우를 숨기지 않는다. `random_equals_targeted`, `distinct_random_alternatives`를 저장하므로 위치 선택의 이득을 해석할 때 이 사례들을 구분해야 한다.

## 비교 방법

초기 PB에 실패하고 연결 graph로 읽을 수 있으며 실패 위치를 지역화할 수 있는 동일 입력을 비교한다.

| 방법 | 고정 정보 | 후보 수 |
| --- | --- | ---: |
| targeted | 원본 생성 graph/stereo + 선택 영역 밖 좌표 | 8 |
| random | 원본 생성 graph/stereo + 무작위 영역 밖 좌표 | 8 |
| whole | 원본 생성 graph/stereo, 좌표 전체 자유 | 8 |
| UFF_targeted | 원본 생성 graph + 진단 영역 밖 heavy atom 좌표 | 1 |

각 FM 후보는 자유 좌표에 sigma=0.3 Å Gaussian noise를 넣어 실제로 다른 prior를 만든다. 동일 후보 seed에서 전체 원자의 noise vector를 먼저 만들어 방법 사이의 noise를 대응시킨다. 초기 noise가 생성물의 stereo를 바꾸면 최대 64회 다시 뽑고, 그래도 초기화하지 못한 슬롯은 실패로 남긴다. 따라서 rejection 횟수가 다르면 방법 사이의 실제 noise vector가 달라질 수 있으며 해당 횟수를 기록한다. 같은 입력에서 seed만 바꾼 동일 FM 출력을 독립 후보로 세지 않는다.

UFF는 처음 생성된 입력에 적용한다. Ligand-only UFF, 보호 좌표 고정, 최대 200회 최소화이며 protein 에너지 항은 없다. FM과 계산 비용이 같은 대조군으로 취급하지 않는다.

각 FM 후보에 사용한 noisy prior도 동일 PB 검사로 측정해 `noise_only_control`에 저장한다. 단순히 noise를 주는 것과 FM의 추가 효과를 구분할 수 있다. 최종 후보 CSV에 없는 세부 항목은 입력별 evaluation JSON에서 확인한다.

이번 단계는 pose-only 복구다. 원자·결합 종류를 바꾸는 fragment 재설계는 포함하지 않는다. LLM 호출 및 모델 학습도 없다.

## 성공과 집계

성공은 원래 생성물의 graph/stereo 보존, 보호 원자 최대 변위 ≤0.001 Å, pocket PB 전체 통과를 모두 요구한다. 새로운 분자에는 대응하는 실험 정답 pose가 없으므로 native RMSD 개선을 요구하지 않는다. 생성 전 좌표와의 RMSD는 수정량일 뿐 pose 정확도가 아니다.

주요 집계는 초기 전체 1,600개 생성 시도에 대한 PB 통과율, 지역화 가능한 PB 실패의 수와 비율, 같은 실패 입력에서 8개 후보 내 적어도 하나의 성공 비율, 방법별 전체 초기 입력당 최종 유효 출력 비율이다. 생성·초기화 실패와 지역화 불가능 사례도 별도로 보고한다. 같은 pocket의 16개 생성물을 독립 target처럼 세지 않고 pocket 단위 bootstrap을 사용한다.

PB를 선택과 평가에 사용하는 목적 최적화 실험이다. 높은 PB 통과율이 결합 친화도 향상이나 독립적인 물리 검증을 뜻하지는 않는다. UFF와 추가 생성의 실제 비용, FM NFE, 후보별 통과와 실패도 함께 보관한다.

## 실행 파일과 상태

- `scripts/generate_pb_initial.py`: 100 pocket 고정, raw de novo 생성, OOM 배치 조절과 저장 지점 재개.
- `src/dirmo/generation/de_novo.py`: ligand graph를 조건으로 주지 않는 upstream 생성 경로.
- `src/dirmo/evaluation/pb_guided.py`: 실제 PB 근거 추출, 영역 선택, seed별 noise 입력, UFF 및 보존 검사.
- `scripts/run_pb_guided_repair.py`: CPU 평가 8개 병행, GPU 복구, 집계, 그림, 기존 256-step 자동 재개.
- `tests/test_pb_guided.py`: 실제 PB 결합 오류 위치, ring 보존, 무작위 영역 크기·경계 대응, stereo와 보호 좌표, whole-pose 대조군 검사.

실행 폴더의 `summary.json`과 `REPORT.md`가 진행 상황이다. `initial.log`, `repair.log`, `controller.log`, `evaluations/logs/*.log`에서 오류를 확인한다. 초기 생성물은 `cases/*/initial`, 실패 근거·mask·prior는 `cases/*/sources`, 복구 결과는 `repairs`에 있다. 완료 후 `statistics.json`, 세 종류의 metrics CSV, `results.png/pdf`를 생성한다.

PB의 50-conformer 기준 에너지는 설정을 유지하고 동일 InChI·conformer 수·단일 CPU thread·RDKit 버전·PB module hash에 대해서만 디스크 cache를 재사용한다. Module 함수, 검사 항목, energy threshold를 줄이지 않는다.

## 2026-10-15 제어기 복구

초기 구현은 CPU의 한 복합체 평가가 3,600초를 넘으면 전체 제어기가 종료되는 문제가 있었다. 그 뒤 GPU 복구 worker는 미완료 입력을 기다리고 기존 256-step도 정지 상태여서 GPU4가 유휴 상태로 남았다. 기존 생성 결과와 완료된 PB 결과를 유지한 채 같은 실행 폴더에서 재개했다.

개별 PB 측정을 별도 CPU 프로세스로 격리하고 기본 600초 제한을 적용한다. 시간 초과는 `missing=true`, `measurement_timeout=true`로 남기며 물리 통과 또는 측정된 물리 실패로 바꾸지 않는다. Conformer 수와 PB 임계값은 동일하다. 동일 SDF·pocket·평가 코드의 측정 결과를 cache하며, 더 긴 예산으로 기존 timeout을 재시도할 수 있다.

복합체별 CPU 작업은 저장 지점에서 한 번 재시도한다. 두 번 실패해도 해당 사례의 누락을 명시하고 다른 사례와 GPU 전환을 계속한다. 평가가 미확정된 상태로 작업이 종료되면 최종 상태는 `complete_with_missing_evaluation`이다. GPU4 전환 기록은 `gpu_switches.jsonl`, 복구 기록은 `recovery_20261015/`에 저장한다.

관련 코드: `src/dirmo/evaluation/pb_worker.py`, `src/dirmo/gpu_priority.py`. `tests/test_pb_recovery.py`에서 timeout을 미측정으로 유지하는지, 완료 결과를 보존하는지, PID가 재사용되면 신호를 보내지 않는지 검증한다.
