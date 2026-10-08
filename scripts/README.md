# 목적별 Bash 실행 스크립트

프로젝트 루트에서 `bash scripts/<목적>.sh`로 실행합니다. 내부에서는 기존 `python -m dirmo` CLI를 호출합니다. `scripts/settings.sh`의 Python 절대 경로를 사용하므로 기본 설정에서는 Conda를 활성화하거나 사용자가 `export`할 필요가 없습니다. Bash 스크립트이므로 `sh` 대신 `bash`를 사용합니다.

## 먼저 확인할 설정

[settings.sh](settings.sh)에서 공통 설정을 바꿉니다. GPU, 모델 경로, LLM/FM worker Python은 선택한 [local.json](../configs/local.json)에 그대로 둡니다. 다른 Config를 쓰려면 `DIRMO_CONFIG`를 변경합니다. 현재 사용자 설정의 GPU 3·4를 이 스크립트 추가 과정에서 변경하지 않았습니다.

| 설정 | 기본값·의미 |
| --- | --- |
| `DIRMO_CPU_PYTHON` | `/home/uslee/miniconda3/envs/dirmo/bin/python` |
| `DIRMO_CONFIG` | `configs/local.json` |
| `DIRMO_MANIFEST` | `examples/gpu_case.json`이 있으면 사용하며, 없으면 `examples/local_case.json` |
| `DIRMO_RUN_DIR` | `runs/pipeline_01`; 단계별 결과와 로그의 공통 위치 |
| `DIRMO_DATASET` | `<RUN_DIR>/branches.jsonl`; 실제 수집한 원본 데이터 |
| `DIRMO_TRAIN_DATASET` | `<RUN_DIR>/training.jsonl`; 학습 조건으로 고른 데이터 |
| `DIRMO_ROUTER_MODEL` | `<RUN_DIR>/router.json` |
| `DIRMO_KERNEL_ID` | 기본 빈 값; 같은 step에 kernel이 여럿이면 선택할 ID |
| `DIRMO_EPOCHS` | `100` |
| `DIRMO_START_TIMEOUT` | `600`초 |

CPU Python은 설치된 `dirmo` 환경이어야 하며 FLOWR worker에는 Python 3.12가 필요합니다. 라이브러리 설치는 [설치 문서](../docs/installation.md)를 참고합니다. 기본 GPU 예제는 사용자가 별도로 만든 파일이므로 새 checkout에는 없을 수 있습니다. 작은 기본 예제의 예산이 부족하면 입력의 `budgets`를 조정합니다.

원한다면 명령 앞에서 해당 실행에만 설정을 바꿀 수 있습니다.

```bash
DIRMO_CONFIG=configs/gpu.json bash scripts/doctor.sh
DIRMO_MANIFEST=examples/gpu_case.json bash scripts/collect.sh
```

이미 shell에 설정된 `DIRMO_*` 변수는 defaults보다 우선합니다. worker 관리·수집·최적화에는 같은 Config와 같은 환경 변수 값을 사용해야 합니다. 서버 실행 후 GPU/모델/설정을 바꾸려면 기존 worker를 종료하고 재시작합니다.

## 목적별 파일

| 파일 | 실행 목적 |
| --- | --- |
| [doctor.sh](doctor.sh) | Python·모델 경로·GPU·FLOWR capability 조사 |
| [start_workers.sh](start_workers.sh) | LLM과 FLOWR 서버를 함께 시작 |
| [start_llm.sh](start_llm.sh) | LLM 서버만 시작 |
| [stop_workers.sh](stop_workers.sh) | 관리 대상 서버 종료; `--only llm`/`--only fm` 가능 |
| [health.sh](health.sh) | 등록된 서버의 실제 health 요청 |
| [check_llm.sh](check_llm.sh) | 기존 LLM 서버의 응답 확인; `--health-only`는 생성 없음 |
| [evaluate_initial.sh](evaluate_initial.sh) | 모델 없이 초기 ligand CPU 평가 |
| [collect.sh](collect.sh) | LLM·FLOWR·평가를 실행해 branch 데이터 수집 |
| [train_router.sh](train_router.sh) | 데이터 조건 확인·선별 후 FM GPU에서 router 학습 |
| [optimize_rule.sh](optimize_rule.sh) | rule 정책 최적화 |
| [optimize_learned.sh](optimize_learned.sh) | 학습된 router 정책 최적화 |
| [compare.sh](compare.sh) | 기본으로 rule과 learned 정책 비교 |
| [run_pipeline.sh](run_pipeline.sh) | 서버 시작 → 수집 → 데이터 준비·학습 → 서버 재시작 → learned 최적화 |
| [settings.sh](settings.sh) | 모든 실행 스크립트가 공유하는 사용자 설정 |
| [_common.sh](_common.sh) | 프로젝트 경로·CPU mask·CLI 호출·로그 저장을 공유하는 내부 함수 |
| [prepare_router_data.py](prepare_router_data.py) | 설정된 generation step과 하나의 kernel에 해당하는 실제 측정 결과를 JSONL로 선별 |

이전 [dirmo](dirmo) wrapper는 그대로 유지합니다. 새 실행 스크립트는 `settings.sh`의 Python 경로를 사용합니다. `settings.sh`와 `_common.sh`는 다른 스크립트에서 source하는 파일이며 개별 작업을 실행하는 entry point는 아닙니다.

대부분의 명령은 뒤에 기존 CLI 옵션을 추가할 수 있습니다.

```bash
bash scripts/start_workers.sh --only fm
bash scripts/collect.sh --manifest examples/gpu_case.json
bash scripts/collect.sh --output runs/extra_seed/branches --dataset runs/pipeline_01/branches.jsonl --seed-offset 1
bash scripts/train_router.sh --epochs 200
bash scripts/train_router.sh --dataset runs/other/branches.jsonl --kernel-id 실제_KERNEL_ID
bash scripts/optimize_rule.sh --resume
bash scripts/compare.sh --policies rule edit pose sequence llm
bash scripts/collect.sh --help
bash scripts/train_router.sh --help
```

`train_router.sh --dataset`은 **원본 수집 데이터**를 지정합니다. wrapper가 이를 선별해 `DIRMO_TRAIN_DATASET`에 저장하고 CLI에 전달합니다. `--config`는 개별 명령 뒤에 추가하지 말고 `settings.sh` 또는 `DIRMO_CONFIG`로 지정합니다. 모든 상대 경로는 프로젝트 루트 기준이며 입력 manifest 안의 상대 경로는 기존 CLI처럼 manifest 위치 기준입니다.

`collect`는 기본적으로 행동마다 한 번씩 생성합니다. `seeds_per_graph`는 최대 시도 수이며 자동 반복 횟수가 아닙니다. 같은 kernel의 측정 관측이 2개 미만이면 `--seed-offset 1` 등으로 seed를 바꾸고 새 output 폴더에서 같은 dataset에 추가 수집합니다. 64-step과 128-step 관측은 서로 다른 kernel입니다. 기존 dataset에 있는 graph/kernel/seed를 반복 요청하면 해당 branch를 실행하지 않습니다. LLM 후보 거부 이유는 case 폴더의 `llm.jsonl`, FM 생성 오류는 `actions.jsonl`, 평가 실패는 `observations.jsonl`, 미실행 사유는 `branch_coverage.json`에서 확인합니다.

## 단계별 실행

```bash
cd /home/uslee/code/DIRMO-3D

bash scripts/doctor.sh
# 이미 관리 대상 worker가 실행 중이면 종료한 뒤 두 서버를 함께 시작합니다.
bash scripts/stop_workers.sh
bash scripts/start_workers.sh
bash scripts/health.sh

bash scripts/collect.sh
bash scripts/train_router.sh

# 학습은 FM worker를 종료합니다. LLM도 종료한 뒤 두 서버를 함께 재시작합니다.
bash scripts/stop_workers.sh
bash scripts/start_workers.sh
bash scripts/optimize_learned.sh

# 필요하면 rule/learned 비교를 추가 실행합니다.
bash scripts/compare.sh
bash scripts/stop_workers.sh
```

개별 실행 스크립트는 다른 단계나 서버 시작을 자동으로 실행하지 않습니다. `train_router.sh`는 실제 학습에 들어가면 기존 CLI에 따라 FM worker를 종료합니다. 기본 `compare.sh`에는 학습된 router artifact가 필요합니다.

## 전체 파이프라인 한 번에 실행

```bash
cd /home/uslee/code/DIRMO-3D
bash scripts/run_pipeline.sh runs/gpu_02
```

출력 경로를 생략하면 `settings.sh`의 `DIRMO_RUN_DIR`을 사용합니다. 이름은 원하는 실험명으로 바꿉니다. 기존 출력 경로나 dataset/model 파일이 있으면 **worker를 종료하기 전에** 중단하므로 새 경로를 사용하세요. 전체 실행은 기존 관리 대상 worker를 먼저 종료하고, 학습 후 두 worker를 재시작하며, 정상 완료나 중간 실패 시 관리 대상 worker를 종료합니다. 수집·학습·최적화가 실제 GPU를 사용하는 명령입니다.

단계가 실패하면 다음 단계로 넘어가지 않습니다. 데이터 선별 실패는 모델 학습 전에 중단합니다. 성공하면 다음 파일이 생성됩니다.

```text
runs/gpu_02/
├── branches/
│   └── <case_id>/                 # 입력·관측·행동·FLOWR 산출물
├── branches.jsonl                 # 원본 수집 데이터
├── training.jsonl                 # 선택한 조건의 측정 데이터
├── router.json                    # 학습된 MLP
├── router.json.report.json        # 학습·검증 보고서
├── learned/
│   └── <case_id>/
│       ├── result.json
│       └── final.sdf
└── logs/
    ├── collect.log
    ├── prepare_training.log
    ├── train.log
    └── learned.log
```

모델 서버의 로그는 Config의 `runtime_dir` 아래 `llm.log`와 `fm.log`에 저장됩니다. 개별 단계 로그는 같은 이름으로 다시 실행하면 덮어쓰므로 실험별 출력 경로를 구분합니다.

## 학습 데이터 조건

학습 wrapper는 `fm.parameters.integration_steps`(생략 시 64)와 일치하고 `utility_change`가 실제로 측정된 행만 사용합니다. 원본 수집 JSONL은 변경하지 않습니다. 같은 step에서 여러 pocket/kernel이 있으면 ID별 개수를 표시하고 중단하므로 `DIRMO_KERNEL_ID` 또는 `--kernel-id`로 선택해야 합니다. 선택한 kernel의 측정 결과가 최소 2개 있어야 합니다.

이 조건을 만족해도 예제 한 건의 소량 데이터는 실행 흐름을 확인하는 용도입니다. 현재 kernel ID는 pocket 해시를 포함하므로 여러 target에 적용할 공용 router 학습에는 별도 코드 변경이 필요합니다. 조건을 만족하는 개선 후보가 없으면 최적화는 `no_improvement`와 원본 SDF를 반환할 수 있습니다.

스크립트 추가 과정에서는 Bash/Python 구문, 문서 연결과 모든 실행 스크립트의 `--help` 동작을 확인했습니다. 모델 로드·서버 요청·GPU 학습은 실제 실행하지 않았습니다.
