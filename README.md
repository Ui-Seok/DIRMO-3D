# DIRMO-3D

**DIRMO: Diagnosis-Guided Intervention Routing for 3D Structure-Based Molecular Optimization**

DIRMO-3D는 단백질 pocket 안의 분자를 평가하고, 측정 결과를 근거로 화학 구조 편집과 3D pose 재생성 중 다음 행동을 선택하는 연구용 코드입니다. 동결된 LLM이 편집 후보를 제안하고, router가 행동을 선택하며, 동결된 FLOWR.root가 3D 좌표를 생성합니다. 생성된 결과는 실제 평가 도구로 다시 측정합니다.

현재 LLM 서버는 **Hugging Face Transformers를 사용하는 자체 HTTP worker**입니다. vLLM 또는 SGLang 연결은 구현되어 있지 않습니다. CPU 실행 관리, LLM 서버, FLOWR.root 서버는 별도 프로세스이고, 학습된 router의 추론은 NumPy로 CPU에서 수행합니다. 기본 파이프라인의 학습 대상은 router입니다. 별도 실험으로 추가한 [SMDD-Bench instruction tuning](docs/smdd_instruction_tuning.md)은 Qwen3.5의 LoRA adapter를 학습하고 검증합니다. 해당 문서에 prompt와 학습 코드가 있고, [전체 학습·평가 안내](docs/smdd_full_training.md)에 LR 비교, 5-epoch 추가 학습, 전체 split 평가와 로그 위치를 정리했습니다. [학습 없는 편집·도구 피드백 실험](docs/smdd_feedback.md)은 base LLM의 짧은 편집 가설을 RDKit·ADMET으로 검사하고 실패 조건을 반영해 수정하며, 같은 제안 예산의 비교와 생존 후보의 Boltz 검증을 지원합니다.

2026-10-19에 추가한 [CrossDocked2020 SBDD benchmark](docs/crossdocked_sbdd_benchmark.md)는 FLOWR.root base의 test 100개 × 100회 생성을 평가합니다. 기존 ligand를 개선하는 SBMO 실험과 구분하며, 설정·성능 지표·진행 상태·결과 위치는 해당 문서를 참고하세요.

이 문서는 2026-10-08에 로컬 디렉터리의 소스, 설정, 문서, 저장된 보고서와 생성 파일을 읽어 정리했습니다. README 작성 과정에서는 테스트, 서버 요청, 모델 로드 또는 추론을 실행하지 않았습니다. `reports/`의 과거 결과와 `worker-runtime/`의 사용자 실행 기록을 현재 환경의 검증 결과로 취급하지 않습니다.

- [폴더 구조](#폴더-구조)
- [폴더별 역할](#폴더별-역할)
- [코드 파일별 역할](#코드-파일별-역할)
- [설정·예제·문서 파일](#설정예제문서-파일)
- [보고서와 실행 산출물](#보고서와-실행-산출물)
- [자동 생성 파일과 Git 관리 파일](#자동-생성-파일과-git-관리-파일)
- [설치와 실행](#설치와-실행)
- [목적별 Bash 스크립트](#목적별-bash-스크립트)
- [구현 범위와 해석상 제한](#구현-범위와-해석상-제한)

## 전체 실행 흐름

```text
입력 Case: protein/pocket + initial ligand + 목표 + 보호 조건 + 예산
  → Run: 초기 atom map, 고정 pocket, 초기 평가, 상태와 예산 준비
  → Diagnosis: 실제 평가값을 근거 Evidence로 구성
  → Planner / LLM worker: 근거를 인용하는 화학 편집 후보 제안
  → available_actions: 실행 가능성·보호 조건·예산으로 행동을 제한
  → Router: 편집 / pose 재생성 / 생성 step 변경 / 종료 선택
  → Executor / FLOWR worker: 필요한 편집을 적용하고 3D pose 생성
  → Evaluator: graph, geometry, 물성, PoseBusters, Vina, 보호 조건 평가
  → Run: 관측·비용·상태 저장 후 반복하거나 최종 SDF 반환
```

`collect`는 이 실행 경로에서 실제로 시도한 행동과 결과를 학습 데이터로 모읍니다. `train-router`는 행동별 utility 변화와 실행 비용을 예측하는 MLP를 학습합니다. 모든 정책은 공통 executor와 동일한 실행 가능 행동 제한을 사용합니다.

## 폴더 구조

아래는 문서 작성 시 실제로 존재하는 폴더와 파일입니다. 모든 Python 캐시 폴더를 표시하되, 그 안의 반복적인 파일명은 `*.pyc`로 묶었습니다. `.git/` 내부의 역할은 아래 관리 파일 설명에서 정리합니다. `runs/`는 해당 출력 경로로 실행할 때 생성되고, LLM 캐시는 각 case 실행 폴더의 `cache/`에 생성됩니다. 현재 트리에는 두 종류의 폴더가 없습니다.

```text
DIRMO-3D/
├── .git/
│   └── ... (Git 내부 구조는 아래 관리 파일 설명 참고)
├── configs/
│   ├── gpu.json
│   ├── llm-alternatives.json
│   └── local.json
├── docs/
│   ├── schemas/
│   │   ├── action.schema.json
│   │   ├── case.schema.json
│   │   └── proposal.schema.json
│   ├── backbone_audit.md
│   ├── implementation_request.md
│   ├── installation.md
│   └── review.md
├── examples/
│   ├── gpu_case.json
│   └── local_case.json
├── reports/
│   ├── archive/
│   │   ├── cpu-validation/
│   │   │   └── crossdock_14gs_20gs/
│   │   │       ├── inputs/
│   │   │       │   ├── fixed_pocket.pdb
│   │   │       │   ├── initial.sdf
│   │   │       │   └── preparation.json
│   │   │       ├── checkpoint.json
│   │   │       ├── costs.json
│   │   │       ├── initial_evaluation.json
│   │   │       ├── manifest.json
│   │   │       ├── observations.jsonl
│   │   │       └── resolved_config.json
│   │   ├── cpu-validation-final/
│   │   │   └── crossdock_14gs_20gs/
│   │   │       ├── inputs/
│   │   │       │   ├── fixed_pocket.pdb
│   │   │       │   ├── initial.sdf
│   │   │       │   └── preparation.json
│   │   │       ├── checkpoint.json
│   │   │       ├── costs.json
│   │   │       ├── initial_evaluation.json
│   │   │       ├── manifest.json
│   │   │       ├── observations.jsonl
│   │   │       └── resolved_config.json
│   │   └── cpu-validation-v2/
│   │       └── crossdock_14gs_20gs/
│   │           ├── inputs/
│   │           │   ├── fixed_pocket.pdb
│   │           │   ├── initial.sdf
│   │           │   └── preparation.json
│   │           ├── checkpoint.json
│   │           ├── costs.json
│   │           ├── initial_evaluation.json
│   │           ├── manifest.json
│   │           ├── observations.jsonl
│   │           └── resolved_config.json
│   ├── cpu-validation-final/
│   │   └── crossdock_14gs_20gs/
│   │       ├── inputs/
│   │       │   ├── fixed_pocket.pdb
│   │       │   ├── initial.sdf
│   │       │   └── preparation.json
│   │       ├── checkpoint.json
│   │       ├── costs.json
│   │       ├── events.jsonl
│   │       ├── initial_evaluation.json
│   │       ├── manifest.json
│   │       ├── observations.jsonl
│   │       └── resolved_config.json
│   ├── README.md
│   ├── cpu-initial.stderr.log
│   ├── cpu-initial.stdout.json
│   ├── doctor.json
│   ├── planner-prompt.json
│   ├── tokenizer-audit.json
│   ├── unit-tests.json
│   └── unit-tests.log
├── scripts/
│   ├── __pycache__/
│   │   └── *.pyc
│   ├── README.md
│   ├── _common.sh
│   ├── check_llm.py
│   ├── check_llm.sh
│   ├── collect.sh
│   ├── compare.sh
│   ├── dirmo
│   ├── doctor.sh
│   ├── evaluate_initial.sh
│   ├── health.sh
│   ├── optimize_learned.sh
│   ├── optimize_rule.sh
│   ├── prepare_router_data.py
│   ├── run_pipeline.sh
│   ├── settings.sh
│   ├── start_llm.sh
│   ├── start_workers.sh
│   ├── stop_workers.sh
│   ├── train_router.sh
│   └── verify.py
├── src/
│   ├── dirmo/
│   │   ├── __pycache__/
│   │   │   └── *.pyc
│   │   ├── chemistry/
│   │   │   ├── __pycache__/
│   │   │   │   └── *.pyc
│   │   │   ├── __init__.py
│   │   │   ├── graphs.py
│   │   │   └── properties.py
│   │   ├── data/
│   │   │   ├── __pycache__/
│   │   │   │   └── *.pyc
│   │   │   ├── __init__.py
│   │   │   └── collect.py
│   │   ├── diagnosis/
│   │   │   ├── __pycache__/
│   │   │   │   └── *.pyc
│   │   │   └── __init__.py
│   │   ├── evaluation/
│   │   │   ├── __pycache__/
│   │   │   │   └── *.pyc
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py
│   │   │   └── pocket.py
│   │   ├── generation/
│   │   │   ├── __pycache__/
│   │   │   │   └── *.pyc
│   │   │   ├── __init__.py
│   │   │   ├── flowr.py
│   │   │   └── lift.py
│   │   ├── llm/
│   │   │   ├── __pycache__/
│   │   │   │   └── *.pyc
│   │   │   ├── __init__.py
│   │   │   ├── local.py
│   │   │   └── proposals.py
│   │   ├── pipeline/
│   │   │   ├── __pycache__/
│   │   │   │   └── *.pyc
│   │   │   ├── __init__.py
│   │   │   ├── budget.py
│   │   │   └── run.py
│   │   ├── routing/
│   │   │   ├── __pycache__/
│   │   │   │   └── *.pyc
│   │   │   ├── __init__.py
│   │   │   ├── features.py
│   │   │   ├── learned.py
│   │   │   └── policies.py
│   │   ├── training/
│   │   │   ├── __pycache__/
│   │   │   │   └── *.pyc
│   │   │   ├── __init__.py
│   │   │   ├── router.py
│   │   │   └── worker.py
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── doctor.py
│   │   ├── gpu.py
│   │   ├── io.py
│   │   ├── schemas.py
│   │   ├── services.py
│   │   └── worker.py
│   └── dirmo_3d.egg-info/
│       ├── PKG-INFO
│       ├── SOURCES.txt
│       ├── dependency_links.txt
│       ├── entry_points.txt
│       ├── requires.txt
│       └── top_level.txt
├── tests/
│   ├── __pycache__/
│   │   └── *.pyc
│   ├── test_core.py
│   └── test_executor.py
├── worker-runtime/
│   ├── config.json
│   ├── llm.log
│   └── workers.json
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements-llm.txt
└── requirements.txt

```

## 폴더별 역할

| 폴더 | 역할 |
| --- | --- |
| 프로젝트 루트 | 패키지 설치 정보, 두 환경의 requirements, README와 Git 설정을 보관합니다. |
| `configs/` | worker 환경, 모델 경로, GPU, 생성 조건, 평가 도구 설정과 대체 LLM 프로필을 보관합니다. |
| `docs/` | 원문 구현 요구사항, backbone 조사, 설치 안내와 검토 기록을 보관합니다. |
| `docs/schemas/` | 입력 Case, 행동 Action, LLM Proposal의 JSON Schema를 문서용으로 보관합니다. |
| `examples/` | 로컬 데이터셋을 참조하는 실행 입력 예제를 제공합니다. |
| `src/` | setuptools의 src layout에 따른 Python 패키지 소스 위치입니다. |
| `src/dirmo/` | CLI, 공통 데이터 모델, 환경 설정, GPU 식별, 파일 입출력, worker 관리 코드를 보관합니다. |
| `src/dirmo/chemistry/` | RDKit 분자 읽기·쓰기, graph 검증, atom map, 화학 편집과 분자 물성 계산을 담당합니다. |
| `src/dirmo/diagnosis/` | 관측값을 LLM이 인용할 근거로 만들고, graph의 pose 회복 시도에 대한 불확실성을 계산합니다. |
| `src/dirmo/evaluation/` | 고정 pocket 준비와 실제 분자·pose·보호 조건 평가를 담당합니다. |
| `src/dirmo/generation/` | 실제 FLOWR.root 생성 adapter와 비활성 LiFT 비교 인터페이스를 보관합니다. |
| `src/dirmo/llm/` | 로컬 모델 로드·생성, system/user prompt, Proposal 검증, 캐시와 재시도를 담당합니다. |
| `src/dirmo/pipeline/` | 실험 상태, 행동 실행, 예산 집계, 반복 최적화, 재개와 최종 후보 선택을 담당합니다. |
| `src/dirmo/routing/` | 실행 가능 행동 생성, rule·고정·LLM 정책과 학습된 router의 CPU 추론을 담당합니다. |
| `src/dirmo/data/` | 동일 상태에서 실제 branch를 실행해 router 학습 데이터를 수집합니다. |
| `src/dirmo/training/` | router MLP 학습, 데이터 그룹 분할, 검증 통계와 GPU 학습 프로세스를 담당합니다. |
| `scripts/` | 목적별 Bash 실행 스크립트, 공통 실행 설정, 학습 데이터 선별, 기존 LLM 서버 확인, CPU 단위 테스트 실행 도구를 보관합니다. |
| `tests/` | 화학 검증, 예산, 상태, routing, 학습과 mock executor의 단위 테스트를 보관합니다. |
| `reports/` | 과거 CPU 평가·단위 테스트·환경 조사·prompt 검사 결과를 보관합니다. |
| `reports/cpu-validation-final/` | 저장된 CPU 초기 평가의 최종 기록입니다. 하위 case 폴더 하나가 한 입력 사례를 나타냅니다. |
| `reports/archive/` | 이전 평가 버전의 결과를 남겨 수정 전후 기록을 확인할 수 있게 합니다. |
| `reports/archive/cpu-validation/` | 초기 CPU 평가 기록입니다. PoseBusters 컬럼 처리 수정 전 결과를 포함합니다. |
| `reports/archive/cpu-validation-v2/` | 수정 후 중간 CPU 평가 기록입니다. |
| `reports/archive/cpu-validation-final/` | 이전에 보관한 final 평가 기록입니다. 현재 `reports/cpu-validation-final/`과 별도 디렉터리입니다. |
| 각 평가 폴더의 `crossdock_14gs_20gs/` | 해당 case의 설정, 입력 식별 정보, 관측, 비용과 상태를 보관합니다. |
| 각 case 폴더의 `inputs/` | 한 번 준비한 pocket, atom map이 부여된 초기 ligand와 준비 정보를 보관합니다. |
| `worker-runtime/` | 서버 시작 시 확정한 설정, 관리 대상 worker 정보와 로그를 보관합니다. |
| `src/dirmo_3d.egg-info/` | editable 설치 등에서 setuptools가 생성한 패키지 메타데이터입니다. |
| 모든 `__pycache__/` | 해당 폴더의 Python 파일을 import/실행할 때 만들어지는 bytecode 캐시입니다. |
| `.git/` | 버전 관리 데이터입니다. 분자 최적화 코드 실행에는 사용되지 않습니다. |

## 코드 파일별 역할

Python 코드 40개와 Bash 파일 16개를 아래에 모두 설명합니다. 같은 파일명이라도 위치에 따라 역할이 다릅니다. 특히 `src/dirmo/worker.py`는 모델 서버이고, `src/dirmo/training/worker.py`는 router 학습 프로세스입니다.

### 공통 진입점과 실행 관리: `src/dirmo/`

| 파일 | 주요 클래스·함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/__init__.py) | `__version__` | 패키지 설명과 버전 `0.1.0`을 정의합니다. 패키지 import 자체로 모델이나 CUDA를 초기화하지 않습니다. |
| [__main__.py](src/dirmo/__main__.py) | `main` 호출 | `python -m dirmo`를 CLI로 연결하고 종료 코드를 반환합니다. |
| [cli.py](src/dirmo/cli.py) | `load_cases`, `policy`, `main` | 하위 명령과 정책을 선택합니다. JSON 객체·배열 또는 JSONL manifest를 읽고 입력 상대 경로를 manifest 위치 기준으로 해석합니다. CPU orchestrator의 CUDA mask를 비우며 case와 비교 정책을 순차 실행합니다. |
| [config.py](src/dirmo/config.py) | `WorkerConfig`, `LLMConfig`, `FMConfig`, `EvaluationConfig`, `Config`, `load_config` | JSON 설정을 검증하고 `DIRMO_*` 환경 변수로 일부 필드를 덮어씁니다. LLM/FM의 GPU selector·port 분리와 context/output cap 관계를 확인합니다. |
| [schemas.py](src/dirmo/schemas.py) | `Case`, `Objective`, `Protection`, `Proposal`, `Action`, `Observation`, `State`, `GenerationRequest`, `GenerationResult` 등 | 모듈 간 데이터 계약을 Pydantic으로 정의합니다. 목표 방향, 예산, 편집·보호 조건, 상태, 요청·결과를 검증하고 알 수 없는 필드나 비정상 수치를 거부합니다. |
| [gpu.py](src/dirmo/gpu.py) | `inventory`, `resolve_device`, `worker_environment`, `assert_worker_mask` | `nvidia-smi`로 physical GPU를 조사하고 index/UUID를 해석합니다. 현재 장치 검증은 A6000 및 총 메모리 47,000 MiB 이상을 요구합니다. 각 worker에 하나의 GPU UUID만 노출하고 내부 장치는 `cuda:0`으로 고정합니다. |
| [io.py](src/dirmo/io.py) | `digest`, `sha256_file`, `write_json`, `append_jsonl`, `read_jsonl`, 환경·revision 수집 함수 | 입력·설정·모델 식별용 해시, atomic JSON 저장, JSONL 누적 기록과 실행 환경 정보를 제공합니다. |
| [doctor.py](src/dirmo/doctor.py) | `interpreter_probe`, `doctor` | Python 환경, 모델 경로·revision, FLOWR 소스·checkpoint, GPU와 capability blocker를 조사합니다. 별도 Python probe는 CUDA를 숨기며 모델을 로드하지 않습니다. `ready` 판정만으로 실제 추론 성공이 검증되지는 않습니다. |
| [services.py](src/dirmo/services.py) | `WorkerClient`, `clients`, `owned_process`, `start_workers`, `stop_workers` | Bearer 인증 HTTP client와 worker 생명주기를 관리합니다. VRAM·GPU 분리를 확인하고 프로세스를 시작하며 config hash, instance와 GPU identity를 검사합니다. 종료 전 관리 대상 프로세스인지 확인합니다. |
| [worker.py](src/dirmo/worker.py) | `serve`, `main`, HTTP handler | 역할에 따라 `LocalLLM` 또는 `FlowrBackend`를 한 번 로드해 유지하는 localhost HTTP 서버입니다. 모델 요청은 lock으로 직렬화하며 health, 추론, 종료 route를 제공합니다. 실제 GPU UUID를 검사하고 요청 크기와 오류 응답을 관리합니다. |

### 분자 graph와 물성: `src/dirmo/chemistry/`

| 파일 | 주요 클래스·함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/chemistry/__init__.py) | 패키지 선언 | 화학 graph 처리 패키지의 책임을 설명합니다. 별도 실행 로직은 없습니다. |
| [graphs.py](src/dirmo/chemistry/graphs.py) | `ChemicalError`, `read_sdf`, `write_sdf`, `mapped`, `graph_id`, `graph_check`, `protected_graph`, `execute_edit`, `input_conformer` | 유한한 3D 좌표를 가진 단일 연결 분자를 읽고, atom map과 graph identity를 관리합니다. 원소·전하·결합·동위원소·stereo를 검증하고 보호 원자·결합·SMARTS를 확인합니다. 원자 치환, 말단 원자 추가·제거, 결합 변경 또는 mapped product SMILES 편집을 적용합니다. ETKDG conformer는 편집 후 입력 준비용으로만 사용합니다. |
| [properties.py](src/dirmo/chemistry/properties.py) | `graph_properties`, `feature_factory`, `pharmacophore` | QED, RDKit contrib SA, MW, logP, HBD/HBA, TPSA, 회전 가능 결합, Ro5 위반 수를 계산합니다. RDKit feature의 종류·원자 map·좌표를 제공하며 graph 물성과 feature factory를 캐시합니다. |

### 진단 근거: `src/dirmo/diagnosis/`

| 파일 | 주요 함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/diagnosis/__init__.py) | `assemble_evidence`, `wilson` | 이 initializer에는 실제 구현이 있습니다. 평가값 또는 도구 실패 상태를 식별 가능한 `Evidence`로 만들고, pose 회복 성공/시도 횟수에 Wilson 구간을 계산합니다. 시도가 적거나 없는 경우 불확실성을 명시합니다. |

### 실제 평가: `src/dirmo/evaluation/`

| 파일 | 주요 클래스·함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/evaluation/__init__.py) | 패키지 선언 | 실제 평가 패키지의 책임을 설명합니다. |
| [pocket.py](src/dirmo/evaluation/pocket.py) | `structure`, `protein_atoms`, `prepare_fixed_pocket` | PDB/mmCIF 단백질 원자를 읽습니다. 전체 protein이 주어지면 초기 ligand와의 거리로 pocket을 준비하고, 이미 pocket이 주어지면 해당 protein residue를 유지합니다. pocket을 초기 상태에서 한 번 저장하며 후보마다 다시 자르거나 좌표를 정렬하지 않습니다. |
| [metrics.py](src/dirmo/evaluation/metrics.py) | `geometry`, `posebusters`, `vina_score`, `interactions`, `preservation`, `measure`, `Evaluator` | 결합 길이·충돌, PoseBusters dock 검사, CPU Vina score-only, 선택적 ProLIF 상호작용과 보호 조건을 평가합니다. 도구별 결과·실패·시간·버전을 기록하고 고정 목표의 utility, 허용 손실, feasibility, target 충족 여부를 계산합니다. |

### FLOWR 생성과 비교 인터페이스: `src/dirmo/generation/`

| 파일 | 주요 클래스·함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/generation/__init__.py) | 패키지 선언 | 3D 생성 adapter 패키지의 책임을 설명합니다. |
| [flowr.py](src/dirmo/generation/flowr.py) | `CapabilityError`, `audit_paths`, `FlowrBackend` | 고정 revision의 FLOWR.root 소스와 실제 checkpoint를 확인하고 Python 3.12의 단일 GPU에 동결 모델을 로드합니다. fixed-graph/random-prior/Euler 생성, upstream tensor round-trip 검증, seed와 integration step 설정, 실제 forward NFE 집계, raw SDF·kernel 저장과 생성 후 graph 검증을 담당합니다. 실패·OOM·graph mismatch를 명시하며 자동 repair나 대체 생성은 하지 않습니다. |
| [lift.py](src/dirmo/generation/lift.py) | `LiFTComparison` | **비활성 placeholder**입니다. capability에서 지원하지 않는 기능을 표시하고 fixed-graph 또는 local-fragment 요청에 오류를 반환합니다. 실제 LiFT semantic 생성·비교는 구현되어 있지 않습니다. |

### LLM 로드와 화학 제안: `src/dirmo/llm/`

| 파일 | 주요 클래스·함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/llm/__init__.py) | 패키지 선언 | 로컬 LLM과 제안 처리 패키지의 책임을 설명합니다. |
| [local.py](src/dirmo/llm/local.py) | `LocalLLM` | Qwen3.5는 `Qwen3_5ForConditionalGeneration`, ChemDFM-v2는 `AutoModelForCausalLM`으로 로드합니다. 로컬 revision·weights, dtype, thinking template와 context cap을 확인하고 `generate`를 실행합니다. 응답에는 raw text, token 수, latency와 모델 identity를 포함합니다. OOM은 실패로 반환합니다. |
| [proposals.py](src/dirmo/llm/proposals.py) | `messages`, `parse_proposal`, `Planner` | 실제 측정 근거, mapped SMILES, 목표·보호·예산, Proposal schema로 system/user messages를 구성합니다. Proposal 형식·parent graph·근거 ID·편집 유효성을 검증하고 중복 후보를 제거합니다. 모델·생성 설정·prompt를 포함한 캐시 키, token 예산 예약과 제한된 parsing 재시도를 관리합니다. |

최적화용 system prompt는 `proposals.py`의 `messages`에서 정의합니다. `check_llm.py --system`은 수동 응답 확인 요청의 system prompt만 변경합니다.

### 상태·예산·실행: `src/dirmo/pipeline/`

| 파일 | 주요 클래스·함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/pipeline/__init__.py) | 패키지 선언 | 실행 pipeline 패키지의 책임을 설명합니다. |
| [budget.py](src/dirmo/pipeline/budget.py) | `BudgetExhausted`, `Budget` | FM draw, FM NFE, LLM 호출·token, 평가 호출과 wall time 예산을 분리 관리합니다. 실행 전 비용 상한을 예약하고 알려진 미사용 비용은 환급하며, 실패 비용과 예산 이벤트를 기록합니다. |
| [run.py](src/dirmo/pipeline/run.py) | `input_identity`, `Run`, `Executor`, `optimize` | 입력·코드·환경 identity, 초기 평가, 작업 분자·best/history, checkpoint와 재개를 관리합니다. 모든 정책의 행동을 같은 executor로 실행하고, 편집 후 FLOWR 생성·raw graph 재검증·CPU 평가·비용 집계를 연결합니다. 실패한 편집 graph도 회복 시도 대상으로 유지합니다. 종료 시 목표와 최소 개선을 만족하는 history 후보를 고르고, 없으면 원본과 `no_improvement`를 반환합니다. |

### 행동 선택: `src/dirmo/routing/`

| 파일 | 주요 클래스·함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/routing/__init__.py) | 패키지 선언 | 행동 제한과 router 패키지의 책임을 설명합니다. |
| [features.py](src/dirmo/routing/features.py) | `FEATURE_NAMES`, `features` | 행동 실행 전의 평가값·누락 여부, 회복 시도·불확실성, 목표·보호 조건, 예산, 행동 종류·steps·편집 물성 변화를 고정 순서의 수치 feature로 만듭니다. 실행 후 결과 label을 입력 feature에 넣지 않습니다. |
| [policies.py](src/dirmo/routing/policies.py) | `available_actions`, `RuleRouter`, `FixedActionRouter`, `FixedSequenceRouter`, `LLMRouterBaseline` | capability, 보호 조건, graph 시도 횟수와 남은 예산으로 모든 정책의 행동을 먼저 제한합니다. rule 정책은 측정 실패 시 회복 또는 종료를 선택하며, edit/pose 고정 정책, 교대 sequence 정책, LLM의 preferred action 정책을 제공합니다. |
| [learned.py](src/dirmo/routing/learned.py) | `LearnedRouter` | JSON artifact의 정규화 값과 MLP 가중치를 읽어 **CPU NumPy**로 추론합니다. 학습에 포함된 kernel·행동·조건 범위를 확인하고 예측 utility 개선량에서 비용 항을 차감해 행동을 선택합니다. 출력은 성공 확률이 아닙니다. |

### 학습 데이터 수집: `src/dirmo/data/`

| 파일 | 주요 클래스·함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/data/__init__.py) | 패키지 선언 | 실제 행동 결과 수집 패키지의 책임을 설명합니다. |
| [collect.py](src/dirmo/data/collect.py) | `collect` | 같은 state snapshot을 복사해 공통 seed 조건에서 가능한 branch를 실제 실행합니다. `--seed-offset`으로 추가 수집의 seed를 바꾸고 기존 dataset의 graph/kernel/seed 중복 요청을 차단합니다. 실행 전 feature, 실행 후 평가, utility 변화·비용·kernel identity를 JSONL로 기록합니다. branch snapshot과 coverage도 저장하며 미실행·측정 실패 branch에 보상을 만들어 넣지 않습니다. 전체 수집에는 공통 유한 예산을 적용합니다. |

### Router 학습: `src/dirmo/training/`

| 파일 | 주요 클래스·함수 | 역할 |
| --- | --- | --- |
| [__init__.py](src/dirmo/training/__init__.py) | 패키지 선언 | router 학습 패키지의 책임을 설명합니다. |
| [router.py](src/dirmo/training/router.py) | `group_components`, `grouped_folds`, `cluster_bootstrap`, `fit_model`, `train_router` | 실제 측정 label이 있는 branch 데이터로 utility 변화와 wall time을 예측하는 `입력 → 32 → 16 → 2` MLP를 학습합니다. target 또는 series 및 동일 snapshot이 분할을 넘지 않도록 묶고, 학습 fold의 통계로만 정규화합니다. grouped CV, cluster bootstrap과 충분하지 않은 그룹 수의 제한을 보고하고 JSON 모델·report를 저장합니다. |
| [worker.py](src/dirmo/training/worker.py) | `main` | FM GPU에서 router를 학습하는 별도 프로세스입니다. Torch import 전 단일 UUID mask를 검증하고 `cuda:0`에서 학습하며 실제 GPU·PID·peak memory를 보고합니다. 배포 시 router 추론은 CPU입니다. |

### 실행 보조 스크립트: `scripts/`

| 파일 | 주요 함수·형식 | 역할 |
| --- | --- | --- |
| [dirmo](scripts/dirmo) | Bash wrapper | 프로젝트의 `src`를 `PYTHONPATH`에 넣고 CPU CUDA mask를 비운 뒤 `python -m dirmo`를 실행합니다. `DIRMO_CPU_PYTHON`이 없으면 기존 `flowr` 환경의 Python을 사용합니다. 활성 Conda 환경을 자동 선택하는 wrapper는 아닙니다. |
| [check_llm.py](scripts/check_llm.py) | `positive_timeout`, `main` | Python 표준 라이브러리만으로 기존 worker manifest의 인증 정보를 읽어 `/health`와 선택적 `/propose`를 호출합니다. 기본 요청은 `{"ok": true}`이며 raw 응답·token·시간·메모리와 JSON 형식 일치 여부를 출력합니다. 모델을 새로 로드하지 않습니다. |
| [verify.py](scripts/verify.py) | 스크립트 진입부 | CUDA를 비운 하위 프로세스에서 `unittest discover`를 실행하고 `reports/unit-tests.log`와 `reports/unit-tests.json`에 결과를 저장합니다. 실제 LLM/FLOWR 모델 테스트는 포함하지 않습니다. |
| [prepare_router_data.py](scripts/prepare_router_data.py) | `prepare`, `main` | Config의 생성 step과 하나의 kernel에 해당하는 실제 측정 결과를 선별합니다. 최소 2개 관측, 유한한 utility와 원본/출력 경로 분리를 확인하고 atomic JSONL로 저장합니다. 여러 kernel이 있으면 임의 선택하지 않고 ID 선택을 요구합니다. |

목적별 Bash 파일의 책임은 다음과 같습니다. 실행 방법과 옵션은 [scripts/README.md](scripts/README.md)에 정리했습니다.

| 파일 | 역할 |
| --- | --- |
| [settings.sh](scripts/settings.sh) | CPU Python, Config, 입력, 공통 출력 경로, 학습 epochs·kernel과 시작 timeout을 설정합니다. |
| [_common.sh](scripts/_common.sh) | 프로젝트 루트 이동, CPU CUDA mask, 소스 경로, CLI 호출과 단계 로그를 공유합니다. |
| [doctor.sh](scripts/doctor.sh) | 환경·모델·GPU 조사 명령을 실행합니다. |
| [start_workers.sh](scripts/start_workers.sh) | LLM/FM worker를 시작합니다. |
| [start_llm.sh](scripts/start_llm.sh) | LLM worker만 시작합니다. |
| [stop_workers.sh](scripts/stop_workers.sh) | 관리 대상 worker를 종료합니다. |
| [health.sh](scripts/health.sh) | 실제 worker health를 요청합니다. |
| [check_llm.sh](scripts/check_llm.sh) | Config의 runtime 위치를 읽어 기존 LLM 확인 스크립트를 호출합니다. |
| [evaluate_initial.sh](scripts/evaluate_initial.sh) | 초기 CPU 평가를 실행하고 로그를 저장합니다. |
| [collect.sh](scripts/collect.sh) | 실제 branch 데이터 수집과 로그 저장을 실행합니다. |
| [train_router.sh](scripts/train_router.sh) | 학습 데이터를 선별한 뒤 FM GPU에서 router를 학습합니다. 데이터 준비가 실패하면 학습 CLI를 호출하지 않습니다. |
| [optimize_rule.sh](scripts/optimize_rule.sh) | rule 정책으로 최적화합니다. |
| [optimize_learned.sh](scripts/optimize_learned.sh) | JSON 모델 artifact의 learned 정책으로 최적화합니다. |
| [compare.sh](scripts/compare.sh) | 기본으로 rule/learned 정책을 비교합니다. 다른 정책은 CLI 옵션으로 지정합니다. |
| [run_pipeline.sh](scripts/run_pipeline.sh) | 수집·GPU 학습·learned 최적화를 연결하고 학습 후 worker를 재시작합니다. 기존 출력은 거부하고 종료 시 worker를 정리합니다. |

### 단위 테스트: `tests/`

| 파일 | 주요 테스트 | 역할 |
| --- | --- | --- |
| [test_core.py](tests/test_core.py) | `ChemistryTests`, `ProposalTests`, `BudgetAndGPUTests`, `EvaluationTests`, `StateAndRouterTests`, `TrainingTests` | graph/stereo/전하·편집·보호 조건, Proposal 근거 검증, 예산과 GPU mask, 평가 누락, 상태·재개·routing, 학습 분할과 artifact를 검증합니다. 분자 fixture와 임시 파일을 사용합니다. |
| [test_executor.py](tests/test_executor.py) | `FakeFM`, `FakeLLM`, `ExecutorTests` | fake worker를 사용해 생성 실패 후 편집 graph 회복, 실행 비용, 제한된 Proposal parsing 재시도를 검증합니다. PoseBusters의 변경된 컬럼명 처리도 확인합니다. 실제 GPU 모델 성능 검증은 아닙니다. |

## 설정·예제·문서 파일

### 루트 파일과 설정

| 파일 | 역할과 주의할 점 |
| --- | --- |
| [README.md](README.md) | 프로젝트 구조, 코드 책임과 실행 방법을 설명하는 현재 문서입니다. |
| [pyproject.toml](pyproject.toml) | 패키지명 `dirmo-3d`, 버전, Python 요구사항, src layout, `dirmo` console entry point, 기본·선택 의존성, pytest/ruff 설정을 정의합니다. 실제 패키지 설정의 기준입니다. |
| [requirements-llm.txt](requirements-llm.txt) | `dirmo-llm` 환경용입니다. Torch `2.6.0+cu124`, Transformers `5.15.0`, Accelerate, 공통 런타임과 editable DIRMO 설치를 지정합니다. 기본 SDPA 추론에 flash-attn은 필요하지 않습니다. |
| [requirements.txt](requirements.txt) | Python 3.12의 `dirmo` 환경용입니다. CPU 실행·평가, router 학습, FLOWR inference를 함께 설치합니다. Torch `2.11.0+cu126`, TensorDict `0.12.4`, Biotite `1.7.1`과 평가 라이브러리, `../flowr_root` 및 DIRMO의 editable 설치를 지정합니다. |
| [.gitignore](.gitignore) | Python 캐시·설치 메타데이터, runs, worker-runtime과 모델 weight 등을 버전 관리에서 제외합니다. 로컬에 생성된 파일이 새 checkout에는 없을 수 있습니다. |
| [configs/local.json](configs/local.json) | LLM/FM의 Python·GPU·port·VRAM 최소 여유, 모델·소스·revision, 생성 조건, 평가 활성화, runtime 위치와 seed의 설정입니다. 현재 사용자 설정은 `dirmo-llm`/`dirmo` Python과 physical GPU 3/4를 지정합니다. |
| [configs/gpu.json](configs/gpu.json) | 사용자가 만든 GPU 실행용 Config 복사본입니다. Bash 스크립트에서 사용하려면 `DIRMO_CONFIG`로 선택합니다. 로컬 파일이며 새 checkout에는 없을 수 있습니다. |
| [configs/llm-alternatives.json](configs/llm-alternatives.json) | Qwen3.5-4B와 ChemDFM-v2.0-14B의 대체 프로필입니다. 원하는 프로필의 필드를 `local.json`의 `llm`에 병합합니다. 이 파일 자체는 CLI용 Config 형식이 아니며 자동으로 선택되지 않습니다. |
| [examples/local_case.json](examples/local_case.json) | 로컬 CrossDocked의 protein pocket, ligand SDF, receptor PDBQT를 참조하는 입력입니다. 고정 목표·보호 조건·예산을 포함합니다. 예제 입력 파일과 데이터셋은 프로젝트 내부에 복사되어 있지 않습니다. |
| [examples/gpu_case.json](examples/gpu_case.json) | 사용자가 GPU 실행용으로 만든 예제입니다. FM NFE 예산은 4,096, wall-time 상한은 3,600초이며 draw 상한은 4입니다. 현재 Bash 스크립트가 이 파일을 발견하면 기본 manifest로 사용합니다. |
| [scripts/README.md](scripts/README.md) | 목적별 `.sh` 파일, 공유 설정, 단계별 실행과 전체 파이프라인 사용법을 설명합니다. |

기본 모델·소스 경로는 다음과 같습니다.

| 항목 | 기본값 |
| --- | --- |
| LLM | `/LocalData1/uslee/checkpoints/Qwen3.5-9B` |
| FLOWR.root 소스 | `/home/uslee/code/flowr_root` |
| FLOWR.root revision | `739a33c2aab74d074dd24605b9a2e34dc809ae9c` |
| FLOWR.root checkpoint | `/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_base_spindr.ckpt` |
| 데이터셋 위치 | `/LocalData1/uslee/datasets/` 아래 예제에 지정된 파일 |
| LLM / FM port | `8765` / `8766` |
| LLM / FM physical GPU selector | 현재 사용자 설정 `"3"` / `"4"`; 실행 장치에 맞게 변경 |
| FLOWR 생성 조건 | `random` prior, `euler` solver, `64` integration steps |
| 평가 | PoseBusters·Vina 활성, ProLIF interactions 비활성 |

`fm.checkpoint_sha256`이 `null`이면 모델 로드 시 실제 파일의 SHA256을 계산합니다. 모델을 바꿀 때는 경로뿐 아니라 revision, backend, dtype와 생성 설정도 해당 프로필에 맞춰 변경합니다. `DIRMO_LLM_MODEL`만 바꾸면 기존 revision은 그대로 남습니다.

### 문서와 schema

| 파일 | 역할 |
| --- | --- |
| [docs/implementation_request.md](docs/implementation_request.md) | 최초 구현 요청 원문과 연구·구현 요구사항입니다. 요구사항의 배경을 확인할 때 읽습니다. 요청된 모든 기능이 현재 구현되어 있다는 뜻은 아닙니다. |
| [docs/backbone_audit.md](docs/backbone_audit.md) | FLOWR.root와 LLM의 소스·weight·환경·conditioning 조사를 기록합니다. 초기 시점의 checkpoint 또는 환경 부재 내용이 포함되어 있으므로 현재 파일 존재 여부는 설정과 로컬 환경으로 확인해야 합니다. |
| [docs/installation.md](docs/installation.md) | 두 Conda 환경의 설치, Torch/CUDA·드라이버 조합, Python 경로 설정, worker 시작과 HTTP 응답 확인을 설명합니다. |
| [docs/review.md](docs/review.md) | 초기 구현 검토와 검증 범위, 알려진 제한을 정리한 기록입니다. |
| [docs/schemas/case.schema.json](docs/schemas/case.schema.json) | 입력 Case의 경로, 목표, 보호 조건과 예산 구조를 설명합니다. |
| [docs/schemas/action.schema.json](docs/schemas/action.schema.json) | 행동 종류, 편집 후보, generation parameters, seed와 근거 ID 구조를 설명합니다. |
| [docs/schemas/proposal.schema.json](docs/schemas/proposal.schema.json) | LLM이 반환할 편집 후보·가설·선호 행동의 구조를 설명합니다. |

실제 실행의 schema 검증은 `src/dirmo/schemas.py`의 Pydantic 모델이 담당합니다. `docs/schemas/`의 파일을 실행 중 직접 로드하는 구조는 아닙니다.

입력 manifest는 단일 JSON 객체, JSON 배열 또는 한 줄당 Case인 JSONL을 받습니다. `protein_path` 또는 `pocket_path`가 필요하며 `initial_ligand_path`는 동일한 단백질 좌표계의 3D SDF여야 합니다. SDF에서는 첫 번째 record를 사용합니다. Vina에는 준비된 receptor PDBQT가 필요하고, interactions를 활성화하려면 explicit-H protein 등 해당 평가의 입력 준비가 필요합니다.

## 보고서와 실행 산출물

### 이미 저장된 `reports/`

| 파일·폴더 | 역할 |
| --- | --- |
| [reports/README.md](reports/README.md) | 저장된 CPU 검증의 범위와 결과를 설명합니다. |
| [reports/unit-tests.log](reports/unit-tests.log) | 과거 CPU 단위 테스트 28개의 상세 로그입니다. |
| [reports/unit-tests.json](reports/unit-tests.json) | 당시 테스트 실행 시간·Python·종료 코드·검증 범위를 기계가 읽을 수 있게 저장합니다. |
| [reports/doctor.json](reports/doctor.json) | 초기 환경 조사 결과입니다. 당시 FM 환경/checkpoint 미설정과 GPU 접근 제한을 포함하며 현재 상태를 나타내는 파일이 아닙니다. |
| [reports/tokenizer-audit.json](reports/tokenizer-audit.json) | tokenizer의 non-thinking template과 prompt/context 크기를 CPU에서 확인한 기록입니다. 실제 LLM weight 로드·생성 검증과 구분합니다. |
| [reports/planner-prompt.json](reports/planner-prompt.json) | 측정 근거를 포함하는 system/user messages 예시입니다. |
| [reports/cpu-initial.stdout.json](reports/cpu-initial.stdout.json) | 초기 평가 CLI의 JSON 출력 기록입니다. |
| [reports/cpu-initial.stderr.log](reports/cpu-initial.stderr.log) | 해당 평가의 RDKit 등 경고 기록입니다. |
| [reports/cpu-validation-final/](reports/cpu-validation-final/) | 저장된 최종 CPU 초기 평가입니다. geometry·물성·PoseBusters·Vina 결과를 포함합니다. |
| [reports/archive/](reports/archive/) | `cpu-validation`, `cpu-validation-v2`, `cpu-validation-final`의 과거 평가를 각각 보관합니다. |

각 버전의 `crossdock_14gs_20gs/`에서 같은 파일명이 반복되는 이유는 실행 identity와 상태를 버전별로 보존하기 때문입니다.

| case 폴더 안의 파일 | 역할 |
| --- | --- |
| `manifest.json` | Case, 입력 파일 해시, 코드·환경·모델 관련 identity를 기록합니다. 원래 사용자가 입력한 manifest 파일과 별도로 생성됩니다. |
| `resolved_config.json` | 해당 실행에 실제로 적용된 설정을 저장합니다. |
| `checkpoint.json` | State, state hash, 산출물 checksum, 예산·wall time 및 pending action 등 재개 정보를 저장합니다. |
| `costs.json` | 평가·LLM·FM 비용 집계와 예산 이벤트를 저장합니다. |
| `initial_evaluation.json` | 초기 ligand의 평가 결과를 저장합니다. |
| `observations.jsonl` | 실제 평가 관측을 한 줄씩 누적합니다. |
| `events.jsonl` | 재개 등 실행 이벤트를 기록합니다. 현재 최종 CPU 기록에는 존재하며 모든 과거 버전에 있는 파일은 아닙니다. |
| `inputs/fixed_pocket.pdb` | 초기 상태에서 준비한 고정 pocket입니다. |
| `inputs/initial.sdf` | 초기 3D ligand와 atom map을 보관합니다. |
| `inputs/preparation.json` | 입력 준비 과정, pocket residue와 고정 좌표·box 등의 정보를 기록합니다. |

저장된 최종 CPU 평가에서는 QED가 약 `0.4317`, SA가 약 `2.6174`, Vina score가 `-5.942`이고, `feasible=true`, `target_met=false`입니다. 평가 호출은 1회이고 LLM/FM 비용은 0입니다. 이 기록은 최적화 정책의 개선 효과나 실제 FLOWR 생성 성공을 보여주는 실험이 아닙니다. 과거 테스트의 통과 기록도 이번 README 작성 중 다시 실행한 결과는 아닙니다.

### `worker-runtime/`

| 파일 | 역할 |
| --- | --- |
| [worker-runtime/config.json](worker-runtime/config.json) | worker 시작 시 확정한 설정입니다. 환경 변수 적용 후 값이므로 `configs/local.json`과 다를 수 있습니다. |
| `worker-runtime/workers.json` | 관리 대상 worker의 PID, port, instance, GPU, 로그 경로와 인증 token을 저장합니다. 파일 권한을 제한하며 token은 공유하지 않습니다. |
| [worker-runtime/llm.log](worker-runtime/llm.log) | LLM 모델 로드와 HTTP 요청 로그입니다. 현재 파일에는 사용자 실행의 health/propose 요청 기록이 있습니다. |
| `worker-runtime/fm.log` | FM worker를 시작하면 생성되는 로그입니다. 현재 폴더 트리에는 없습니다. |

현재 폴더에 worker 정보가 남아 있다는 사실만으로 프로세스가 살아 있다는 것을 알 수는 없습니다. `health`는 실제 서버에 요청하며, `workers.json`을 수동으로 지우거나 다른 실행의 파일로 교체하면 관리·인증 정보가 어긋날 수 있습니다.

### 앞으로 실행하면 생기는 파일

아래 파일은 코드가 생성하는 산출물이며 현재 폴더 트리에 모두 존재하는 것은 아닙니다. CLI의 `--output`과 `--dataset` 위치에 따라 저장 경로가 달라집니다. 별도 경로를 명시하지 않은 실행 기록 파일은 각 case 폴더를 기준으로 설명합니다.

| 경로·파일 | 역할 |
| --- | --- |
| `runs/<실험>/<case_id>/` | `--output` 아래 case별 실행 폴더입니다. 입력·설정·checkpoint·평가 기록을 저장합니다. |
| `llm_model.json`, `fm_health.json` | 실제 worker의 모델/kernel identity, 환경과 GPU 정보를 저장합니다. |
| `llm.jsonl` | 제안 요청, raw 응답, JSON/schema 검증 오류(`parsing_error`), 편집 검증 오류(`chemical_error`, `rejected`), token 비용과 cache 사용 기록입니다. HTTP 응답이 200이어도 편집 후보는 검증에서 거부될 수 있습니다. |
| `actions.jsonl`, `hypothesis_outcomes.jsonl` | 선택·실행한 행동과 측정 후 가설 관련 관측 기록입니다. |
| `preparations/` | 화학 편집으로 만든 graph의 입력 준비용 SDF를 저장합니다. |
| `generation/<요청 식별자>/` | 개별 FLOWR 요청의 산출물을 보관합니다. |
| `generation/…/request.json` | 요청 graph, pocket, seed와 생성 parameters를 저장합니다. |
| `generation/…/kernel.json`, `pocket_frame.json` | 실제 checkpoint·소스·조건 identity와 고정 pocket 좌표계 정보를 저장합니다. |
| `generation/…/raw.sdf`, `raw_mapped.sdf`, `graph_check.json` | raw 생성 pose, atom correspondence를 부여한 pose와 graph 검증 결과입니다. |
| `result.json`, `final.sdf` | 최종 선택 결과와 3D 분자입니다. 조건을 만족하는 개선 후보가 없으면 원본을 반환합니다. |
| `cache/llm/<해시>.json` | 모델 identity, 생성 설정과 prompt가 같은 유효 Proposal의 캐시입니다. |
| `branch_snapshot.json`, `branch_coverage.json` | 수집 시작 상태와 실제 실행 또는 미실행 branch의 범위를 기록합니다. |
| `collection_settings.json` | 추가 수집에 적용한 `seed_offset`, 기본 seed와 dataset 경로를 기록합니다. |
| `--dataset`으로 지정한 JSONL | 행동 실행 전 feature와 실제 실행 후 label·비용을 담은 router 학습 데이터입니다. |
| `router.json`, `router.json.report.json` | 예시 학습 출력명입니다. MLP 가중치·정규화·지원 조건과 grouped CV/통계 보고서를 저장합니다. report 파일명은 지정한 모델 출력 경로에 `.report.json`을 덧붙입니다. |
| `comparison.json` | `compare`가 같은 case에서 정책별 실행 결과를 모아 저장하는 파일입니다. |

기록된 모델·소스·입력 해시와 seed가 같은지를 확인해야 결과를 비교할 수 있습니다. 같은 seed를 사용하더라도 단일 case 실험만으로 정책 간 일반적인 우열을 판단하지 않습니다.

## 자동 생성 파일과 Git 관리 파일

### Python 캐시와 설치 메타데이터

`scripts/__pycache__/`, `tests/__pycache__/`, `src/dirmo/__pycache__/`와 모든 하위 패키지의 `__pycache__/`는 Python이 생성한 bytecode 저장소입니다. `*.cpython-311.pyc`, `*.cpython-312.pyc` 등은 환경에 따라 생기며 직접 수정할 소스가 아닙니다.

| `src/dirmo_3d.egg-info/` 파일 | 역할 |
| --- | --- |
| `PKG-INFO` | 설치 시점의 패키지명·버전·설명·의존성 메타데이터입니다. |
| `SOURCES.txt` | 배포에 포함되는 소스 파일 목록입니다. |
| `dependency_links.txt` | setuptools의 이전 방식 dependency link 파일입니다. 현재 내용은 비어 있습니다. |
| `entry_points.txt` | `dirmo` console command를 `dirmo.cli:main`으로 연결합니다. |
| `requires.txt` | 설치 시점의 기본·extra 의존성 목록입니다. |
| `top_level.txt` | 최상위 import 패키지 `dirmo`를 기록합니다. |

이 파일들은 설치 도구가 생성하므로 직접 수정하지 않습니다. 의존성 변경은 `pyproject.toml`과 requirements에서 확인합니다.

### `.git/` 하위 폴더

Git의 hash별 object 파일은 같은 저장 역할을 반복하므로 아래 구조에서는 묶었습니다.

```text
.git/
├── branches/
├── hooks/
│   └── *.sample
├── info/
│   └── exclude
├── logs/
│   ├── HEAD
│   └── refs/
│       ├── heads/
│       │   └── main
│       └── remotes/
│           └── origin/
│               └── main
├── objects/
│   ├── <해시 앞 두 자리>/
│   │   └── <나머지 해시: 압축된 Git object>
│   ├── info/
│   └── pack/
├── refs/
│   ├── heads/
│   │   └── main
│   ├── remotes/
│   │   └── origin/
│   │       └── main
│   └── tags/
├── COMMIT_EDITMSG
├── HEAD
├── config
├── description
└── index
```

| 경로 | 역할 |
| --- | --- |
| `.git/branches/` | Git의 이전 방식 branch 관리용 위치입니다. 현재 비어 있습니다. |
| `.git/hooks/` | Git 이벤트에 연결할 수 있는 sample 스크립트입니다. 현재 `.sample` 파일들은 활성 hook이 아닙니다. |
| `.git/info/` | 저장소 로컬 보조 정보를 보관합니다. `exclude`는 공유하지 않는 ignore 규칙입니다. |
| `.git/logs/` 및 `logs/refs/` | HEAD와 ref 변경 이력인 reflog를 보관합니다. |
| `.git/logs/refs/heads/` | 로컬 branch `main`의 변경 이력입니다. |
| `.git/logs/refs/remotes/` 및 `remotes/origin/` | remote tracking branch `origin/main`의 변경 이력입니다. |
| `.git/objects/` 및 hash 하위 폴더 | commit, tree, blob 등 압축된 Git object를 hash별로 보관합니다. 분자 모델 weight와 무관합니다. |
| `.git/objects/info/`, `.git/objects/pack/` | object 보조 정보와 packed object 저장 위치입니다. |
| `.git/refs/` | branch/tag가 가리키는 commit을 저장합니다. |
| `.git/refs/heads/` | 로컬 branch `main`의 ref입니다. |
| `.git/refs/remotes/` 및 `remotes/origin/` | remote tracking branch `origin/main`의 ref입니다. |
| `.git/refs/tags/` | tag ref 저장 위치입니다. |
| `HEAD`, `config`, `index`, `description`, `COMMIT_EDITMSG` | 각각 현재 branch, 저장소 설정, staging index, 저장소 설명, 최근 commit message를 저장합니다. |

`hooks/`에 있는 코드 파일들의 역할은 다음과 같습니다. 프로젝트 실행 스크립트와 달리 Git이 제공한 비활성 예제입니다.

| 파일 | 역할 |
| --- | --- |
| `applypatch-msg.sample` | 이메일 patch에서 만든 commit message를 검사하는 예제입니다. |
| `commit-msg.sample` | commit message를 검사하는 예제입니다. |
| `fsmonitor-watchman.sample` | Watchman을 이용해 변경 파일을 찾는 Perl 예제입니다. |
| `post-update.sample` | ref 갱신 후 저장소의 서버 보조 정보를 갱신하는 예제입니다. |
| `pre-applypatch.sample` | patch 적용 후 commit 전에 사전 검사를 실행하는 예제입니다. |
| `pre-commit.sample` | staged 변경의 whitespace 등 기본 검사를 수행하는 예제입니다. |
| `pre-merge-commit.sample` | merge commit 전에 pre-commit 검사를 연결하는 예제입니다. |
| `pre-push.sample` | push 직전 commit을 검사하며 WIP commit 제한을 보여주는 예제입니다. |
| `pre-rebase.sample` | branch 이력을 다시 작성해도 되는지 검사하는 예제입니다. |
| `pre-receive.sample` | 서버에서 push option을 처리하는 예제입니다. |
| `prepare-commit-msg.sample` | commit message를 준비·수정하는 예제입니다. |
| `push-to-checkout.sample` | push를 받는 checked-out branch의 working tree 갱신 예제입니다. |
| `update.sample` | ref별 push와 tag 갱신 정책을 검사하는 예제입니다. |

## 설치와 실행

### 두 Conda 환경

설치 상세 절차와 드라이버 관련 설명은 [docs/installation.md](docs/installation.md)를 참고하세요. 모델 serving과 FLOWR의 Torch·Python 요구사항을 분리하기 위해 두 환경을 사용합니다.

| 환경 | Python | 설치 파일 | 사용 범위 |
| --- | --- | --- | --- |
| `dirmo-llm` | 3.11 | `requirements-llm.txt` | Qwen/ChemDFM의 resident LLM server |
| `dirmo` | 3.12 | `requirements.txt` | CPU CLI·평가, router 학습과 FLOWR inference |

기존 `dirmo`가 Python 3.11이면 FLOWR.root를 포함한 통합 requirements를 설치하기 전에 Python 3.12 환경으로 준비해야 합니다. 프로젝트 자체의 최소 Python 3.11 요구사항과 FLOWR adapter의 Python 3.12 요구사항은 다릅니다.

환경을 준비한 뒤 프로젝트 루트에서 각각 설치합니다.

```bash
cd /home/uslee/code/DIRMO-3D

conda activate dirmo-llm
python -m pip install -r requirements-llm.txt

conda activate dirmo
python -m pip install -r requirements.txt
```

통합 requirements는 외부 `../flowr_root` 소스가 존재해야 합니다. 이 환경은 FLOWR.root에 포함된 PoseBusters fork를 사용하므로 PyPI `posebusters` 또는 이를 추가하는 `.[evaluation]` extra를 별도로 설치하지 않습니다. requirements는 전체 upstream `uv.lock`을 재현한 lockfile이 아닙니다.

두 requirements의 Torch wheel은 저장된 서버 드라이버 조사에 맞춰 CUDA 12.4/12.6 계열로 구분되어 있습니다. FLOWR 환경의 CUDA 12.6 조합은 CUDA 12.x minor-version compatibility를 고려한 설정이며, 실제 모델 실행 호환성은 이번 문서 작성에서 확인하지 않았습니다.

### 설정 방법과 환경 변수

`configs/local.json`의 `llm.python`과 `fm.python`을 각각 다음 경로로 수정하면 매번 `export`할 필요가 없습니다.

```json
{
  "llm": {
    "python": "/home/uslee/miniconda3/envs/dirmo-llm/bin/python"
  },
  "fm": {
    "python": "/home/uslee/miniconda3/envs/dirmo/bin/python"
  }
}
```

위 JSON은 기존 설정에 넣을 **필드 예시**입니다. 이 조각만으로 전체 `local.json`을 교체하지 않습니다. Conda 설치 위치가 다르면 실제 Python 경로를 사용하세요. GPU 역시 서로 다른 실제 장치의 index 또는 UUID로 지정합니다.

지원하는 환경 변수는 다음과 같습니다. 비어 있지 않은 값이 있으면 JSON 값보다 우선합니다.

| 환경 변수 | 덮어쓰는 대상 |
| --- | --- |
| `DIRMO_LLM_PYTHON` | `llm.python` |
| `DIRMO_FM_PYTHON` | `fm.python` |
| `DIRMO_LLM_GPU` | `llm.gpu` |
| `DIRMO_FM_GPU` | `fm.gpu` |
| `DIRMO_LLM_MODEL` | `llm.model_path` |
| `DIRMO_FM_CHECKPOINT` | `fm.checkpoint_path` |
| `DIRMO_FLOWR_REPO` | `fm.repo_path` |
| `DIRMO_CPU_PYTHON` | `scripts/dirmo`가 실행할 CPU Python; Config 필드는 아닙니다. |

`export` 없이 명령 앞에서 설정할 수도 있습니다.

```bash
conda activate dirmo
DIRMO_LLM_PYTHON=/home/uslee/miniconda3/envs/dirmo-llm/bin/python \
DIRMO_FM_PYTHON=/home/uslee/miniconda3/envs/dirmo/bin/python \
python -m dirmo --config configs/local.json doctor
```

같은 설정을 사용하는 worker 관리 명령에는 동일한 환경 변수도 적용해야 합니다. `clients`는 적용된 전체 config hash와 저장된 worker 설정이 일치하는지 확인합니다.

CLI는 CPU orchestrator에서 `CUDA_VISIBLE_DEVICES`를 비우고, worker를 시작할 때 해당 UUID 하나를 새 mask로 지정합니다. 따라서 `CUDA_VISIBLE_DEVICES=0 python -m dirmo ...`만으로 worker GPU를 선택하지 말고 `llm.gpu`/`fm.gpu` 또는 `DIRMO_LLM_GPU`/`DIRMO_FM_GPU`를 사용합니다.

### CLI 명령

이하 예시는 두 Python 경로와 GPU가 `configs/local.json`에 설정되어 있다고 가정합니다. `--config`는 전역 옵션이므로 하위 명령 **앞**에 둡니다.

| 명령 | 역할 |
| --- | --- |
| `doctor` | 모델 경로·환경·GPU·FLOWR capability blocker를 조사합니다. |
| `start-workers` | 설정된 모델 서버를 시작합니다. `--only llm` 또는 `--only fm`으로 단독 시작할 수 있습니다. |
| `stop-workers` | 관리 대상 서버를 종료합니다. `--only`로 역할을 제한할 수 있습니다. |
| `health` | 저장된 관리 정보로 실제 worker health를 요청합니다. |
| `evaluate-initial` | 모델 없이 초기 pocket 준비와 CPU 평가를 저장합니다. |
| `optimize` | rule, learned, llm, sequence, edit 또는 pose 정책으로 반복 최적화합니다. |
| `collect` | 실제 branch 결과를 router 학습 JSONL로 수집합니다. |
| `train-router` | MLP와 검증 report를 저장합니다. 기본은 FM GPU 학습이며 `--device cpu`도 가능합니다. |
| `compare` | 동일 case·목표·seed 규약·예산 상한에서 정책별 실행을 별도 저장합니다. |

```bash
conda activate dirmo
cd /home/uslee/code/DIRMO-3D

python -m dirmo --config configs/local.json doctor
python -m dirmo --config configs/local.json evaluate-initial \
  --manifest examples/local_case.json --output runs/initial

python -m dirmo --config configs/local.json start-workers
python -m dirmo --config configs/local.json health
python -m dirmo --config configs/local.json optimize \
  --manifest examples/local_case.json --output runs/rule --router rule
python -m dirmo --config configs/local.json stop-workers
```

현재 `optimize`, `collect`, `compare`의 CLI는 LLM과 FM worker가 모두 등록되어 있어야 진행합니다. 관리 중인 worker가 하나라도 살아 있으면 `start-workers`는 추가 시작을 거부합니다. LLM만 시작한 상태에서 FM을 추가하려면 기존 worker를 종료한 뒤 두 worker를 함께 시작합니다.

```bash
python -m dirmo --config configs/local.json stop-workers
python -m dirmo --config configs/local.json start-workers
```

### LLM 서버만 시작하고 응답 확인하기

```bash
conda activate dirmo
cd /home/uslee/code/DIRMO-3D
python -m dirmo --config configs/local.json start-workers --only llm

# 기존 서버 정보 확인: 생성 요청 없음
python scripts/check_llm.py --health-only

# 실제 생성 요청: 기본 JSON 응답 확인
python scripts/check_llm.py

# 수동 prompt와 system message 지정
python scripts/check_llm.py \
  --system "You are a chemistry assistant." \
  --prompt "Briefly explain what molecular QED measures."
```

위 명령은 사용자가 실행할 예시입니다. `check_llm.py`는 표준 라이브러리만 사용하므로 `dirmo` 환경에서도 실행할 수 있습니다. custom runtime 위치를 사용했다면 `--runtime-dir`을 함께 지정합니다.

LLM 주소의 기본값은 `http://127.0.0.1:8765`이며 브라우저용 홈페이지는 없습니다. 모든 route에 manifest의 Bearer 인증이 필요하므로 주소만 열어서는 동작을 확인할 수 없습니다.

| HTTP route | 역할 |
| --- | --- |
| `GET /health` | 모델·GPU·PID·instance 등 상태를 반환합니다. |
| `POST /propose` | LLM worker에서 messages를 받아 텍스트를 생성합니다. |
| `POST /generate` | FM worker에서 typed generation request를 처리합니다. |
| `POST /shutdown` | 해당 worker의 종료를 요청합니다. |

이 API는 현재 코드 내부 RPC이며 OpenAI-compatible `/v1/chat/completions` 형식이 아닙니다. 모델 요청의 동시 실행 수는 worker당 1이고 tensor parallelism을 사용하지 않습니다.

수동 확인에서 `generation_ok`는 비어 있지 않은 텍스트 생성 여부이고, `json_valid`·`json_matches_expected`는 JSON 형식 준수 여부입니다. 스크립트 종료 코드 0만으로 원하는 JSON까지 일치했다고 판단하지 않습니다. 생성 요청은 GPU를 실제 사용하지만 최적화 run의 예산 ledger에는 포함되지 않습니다. client timeout이 지나도 서버에서 이미 시작한 생성이 자동 취소되지는 않습니다.

### 데이터 수집·학습·정책 비교

```bash
python -m dirmo --config configs/local.json collect \
  --manifest examples/local_case.json --output runs/branches \
  --dataset runs/branches.jsonl

python -m dirmo --config configs/local.json train-router \
  --dataset runs/branches.jsonl --output runs/router.json --device fm

# train-router가 FM worker를 종료하므로 LLM도 종료한 뒤 함께 재시작합니다.
python -m dirmo --config configs/local.json stop-workers
python -m dirmo --config configs/local.json start-workers

python -m dirmo --config configs/local.json optimize \
  --manifest examples/local_case.json --output runs/learned \
  --router learned --router-model runs/router.json

python -m dirmo --config configs/local.json compare \
  --manifest examples/local_case.json --output runs/compare \
  --policies rule edit pose sequence llm

python -m dirmo --config configs/local.json optimize \
  --manifest examples/local_case.json --output runs/rule --router rule --resume
```

학습에는 같은 kernel에서 실제 utility가 측정된 관측이 최소 2개 필요합니다. 기본 `collect`는 행동마다 한 seed로 한 번씩 실행하며 `seeds_per_graph`는 최대 시도 수입니다. 64-step과 128-step 결과는 서로 다른 kernel이라 별도로 학습합니다. LLM 편집 후보가 모두 거부되면 각 step의 관측은 하나씩만 생길 수 있습니다. 추가 수집은 새 output 폴더와 다른 seed를 사용해 같은 dataset에 누적합니다. 기본 seed가 42이면 아래 명령은 seed 43을 사용하며 기존 worker를 재사용합니다.

```bash
python -m dirmo --config configs/local.json collect \
  --manifest examples/local_case.json --output runs/branches_seed43 \
  --dataset runs/branches.jsonl --seed-offset 1
```

같은 graph/kernel/seed가 이미 dataset에 있으면 추가 요청을 차단하고 `branch_coverage.json`에 이유를 남깁니다. 생성 오류는 `actions.jsonl`의 `generation_result.reason`, 평가 실패는 `observations.jsonl`의 `failures` 및 `evaluations`, 편집 후보 거부는 `llm.jsonl`에서 확인합니다. 원격 서버 로그는 `worker-runtime/fm.log`, `worker-runtime/llm.log`에 있습니다.

`train-router`는 `--device cpu`를 선택해도 현재 코드에서 FM worker를 먼저 종료합니다. 여러 kernel의 데이터가 있으면 `--kernel-id`로 하나를 선택합니다. 단일 target 또는 부족한 그룹만으로 학습했다면 held-out 일반화 평가가 충분하지 않을 수 있습니다. pose-resample 관측만으로 학습한 router는 화학 편집 행동을 지원하지 않습니다.

재개는 입력·코드·설정·환경 및 저장된 artifact의 hash가 맞는 action 경계에서 수행합니다. 원격 요청의 결과를 모르는 `pending_action`이 남아 있으면 자동으로 다시 실행하지 않고 중단합니다.

### 개발용 CPU 검증 명령

```bash
conda activate dirmo
cd /home/uslee/code/DIRMO-3D
python scripts/verify.py
```

이 명령은 기존 `reports/unit-tests.log`와 `reports/unit-tests.json`을 갱신합니다. 실제 모델 검증이나 처리량 benchmark는 수행하지 않습니다. README 작성 시에는 실행하지 않았습니다.

## 목적별 Bash 스크립트

명령마다 `python -m dirmo`와 여러 경로를 입력하는 대신 목적별 스크립트를 사용할 수 있습니다. 공통 설정은 [scripts/settings.sh](scripts/settings.sh), 전체 사용법은 [scripts/README.md](scripts/README.md)에 있습니다. 현재 설정된 Conda Python 절대 경로를 사용하므로 별도 `conda activate`나 사용자 `export`가 필요하지 않습니다. Bash를 사용하세요.

```bash
cd /home/uslee/code/DIRMO-3D

# 전체 실행: 새 실험 폴더를 지정합니다.
bash scripts/run_pipeline.sh runs/gpu_02

# 또는 목적별로 실행합니다.
bash scripts/doctor.sh
bash scripts/start_workers.sh
bash scripts/health.sh
bash scripts/collect.sh
bash scripts/train_router.sh
bash scripts/stop_workers.sh
bash scripts/start_workers.sh
bash scripts/optimize_learned.sh
bash scripts/stop_workers.sh
```

전체 실행은 기존 관리 대상 worker를 먼저 종료하고, 수집 → 데이터 선별·GPU 학습 → worker 재시작 → learned 최적화를 수행합니다. 정상 종료 또는 중간 실패 시 worker도 종료합니다. 개별 스크립트는 해당 목적만 실행하며, `train_router.sh`의 실제 학습은 FM worker를 종료합니다. 이미 worker가 실행 중이면 두 서버를 시작하기 전에 `stop_workers.sh`를 실행합니다.

수집 결과에서 설정된 integration step과 같은 kernel의 측정값을 골라 최소 2개 관측을 확인합니다. 여러 kernel이 있으면 `DIRMO_KERNEL_ID`나 `train_router.sh --kernel-id`로 하나를 지정합니다. 현재 kernel은 pocket 해시를 포함하므로 학습된 router는 같은 pocket·생성 조건에 적용합니다. 여러 target에 적용할 공용 router에는 별도 코드 변경이 필요합니다.

각 단계 로그는 `<RUN_DIR>/logs/`에 저장됩니다. 기존 실행 폴더를 덮어쓰지 않도록 전체 파이프라인은 새 출력 경로를 요구합니다. 이번 추가에서는 모델 로드·서버 요청·GPU 실행을 하지 않았습니다.

## 구현 범위와 해석상 제한

- **고정 좌표계:** pocket과 Vina box는 초기 ligand에서 한 번 정합니다. 보호 feature·원자 변위는 protein 좌표계에서 직접 측정하며 ligand를 다시 정렬하지 않습니다. 보호 원자 번호는 complete atom map을 유지하거나 초기 heavy-atom 순서로 부여합니다.
- **화학 편집:** 보호 원자·결합·SMARTS와 유효 graph를 확인한 편집만 실행합니다. 예제의 보호 조건은 비어 있으므로 연구 목적에 필요한 조건을 사용자가 입력해야 합니다.
- **생성 범위:** FLOWR adapter는 whole-graph fixed-graph 생성, random prior와 Euler를 노출합니다. 조건 변경은 integration step 수입니다. 원소·전하·결합·동위원소·stereo가 upstream tensor round-trip 또는 생성 후 검사에서 보존되지 않으면 거부합니다.
- **입력 conformer와 생성 결과:** RDKit ETKDG는 편집 graph의 입력을 준비하는 도구입니다. 이를 최종 FLOWR pose로 취급하지 않습니다. raw 생성 결과와 map을 부여한 결과를 따로 기록합니다.
- **미구현 기능:** local-fragment generation, LiFT semantic comparison, scalar guidance, conformer/harmonic prior, midpoint solver, `EXTRA_VALIDATION` 실행, relaxation/redocking은 지원하지 않습니다. 자동 repair, SMILES uniqueness filtering, 내부 oversampling과 OOM 자동 재시도도 없습니다.
- **평가 해석:** SA는 낮을수록 좋고 Vina는 score-only입니다. PoseBusters에 reference ligand identity/RMSD를 적용하지 않습니다. raw metric과 목표 가중 utility를 분리하며 필수 도구의 missing/failed 결과는 기본적으로 feasible 후보에서 제외합니다. 평가를 끄면 incomplete 상태를 기록합니다.
- **입력 준비:** protein protonation이나 receptor preparation을 자동으로 해결하지 않습니다. ProLIF에는 준비된 explicit-H protein이 필요합니다. 외부에서 준비한 입력의 provenance는 별도로 기록해야 합니다.
- **예산:** `evaluation_calls`는 한 pose에 대한 평가 bundle 수입니다. FM draw와 실제 model forward NFE, LLM 호출과 입출력 token, 평가 bundle, wall time을 구분합니다. 실패·parsing 재시도도 비용에 반영하며 단일 CPU 평가 도구가 wall-time 한계를 초과할 수 있는 제한이 있습니다.
- **학습·통계:** 근거와 회복 구간은 실제 관측의 요약입니다. LLM 가설은 검증 전 추정으로 취급하고, 미실행 branch의 결과를 생성하지 않습니다. learned router 출력은 개선량·비용 회귀이며 성공 확률이나 인과 효과의 증명이 아닙니다.
- **검증 범위:** 저장된 CPU 결과, 사용자 LLM 서버 로그, 실제 FLOWR 생성 품질은 서로 다른 검증 범위입니다. 현재 파일들만으로 모든 모델 조합이나 전체 최적화 경로의 성공을 확인했다고 주장하지 않습니다.
