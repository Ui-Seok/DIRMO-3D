# DIRMO-3D

**DIRMO: Diagnosis-Guided Intervention Routing for 3D Structure-Based Molecular Optimization**

측정값 → 진단 근거 → 동결 LLM의 편집 후보 → router → 동결 FLOWR.root의 3D 생성 → 실제 평가를 연결하는 연구용 구현입니다. CPU orchestrator와 LLM/FM worker는 별도 프로세스를 사용합니다. Conda 환경은 LLM 서버와 training/inference의 두 개로 구성할 수 있습니다. Learned gate 추론은 CPU에서 수행합니다.

필요한 라이브러리와 Conda 설치 방법은 [설치 안내](docs/installation.md)에 정리했습니다. Python 3.11의 `dirmo-llm` 환경에는 LLM 서버용 `requirements-llm.txt`를 설치합니다. Python 3.12의 `dirmo` 환경에는 통합 `requirements.txt`로 CPU 실행·평가, router 학습, FLOWR.root 추론을 함께 설치합니다. FLOWR.root의 Python 3.12 요구사항을 유지하며 upstream 소스를 변경하지 않습니다.

2026-10-08 초기 구현에서는 **코드 작성·검토와 CPU 검증만 수행했습니다. 모델 다운로드 및 LLM/FM GPU 추론은 수행하지 않았습니다.** 이후 사용자가 제공한 `/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_base_spindr.ckpt`를 기본 설정에 연결했습니다. 이 연결 변경에서는 테스트나 모델 로드·추론을 실행하지 않았습니다. 별도 Python 3.12 환경을 지정하기 전에는 FM capability가 `blocked`이며, 실제 checkpoint의 graph/stereo 호환성과 생성 품질은 검증되지 않았습니다.

기존 환경으로 바로 CPU 명령을 사용할 수 있습니다. 아래 명령은 이 디렉터리에서 실행합니다.

```bash
cd /home/uslee/code/DIRMO-3D
bash scripts/dirmo doctor --output reports/doctor.json
bash scripts/dirmo evaluate-initial --manifest examples/local_case.json --output runs/initial
CUDA_VISIBLE_DEVICES= /home/uslee/miniconda3/envs/flowr/bin/python scripts/verify.py
```

`scripts/dirmo`는 `DIRMO_CPU_PYTHON`을 지정하면 해당 환경을 사용하며, 기본값은 기존 `/home/uslee/miniconda3/envs/flowr/bin/python`입니다. 두 Conda 환경 구성에서는 CPU Python과 FM Python에 동일한 `dirmo` Python을 지정합니다. 기존 환경·upstream 소스·데이터를 수정하지 않았습니다. 통합 환경은 FLOWR.root의 PoseBusters fork를 사용하므로 PyPI PoseBusters를 추가하는 `.[evaluation]` extra를 설치하지 않습니다.

LLM은 실제 확인한 `/LocalData1/uslee/checkpoints/Qwen3.5-9B`가 기본입니다. `configs/llm-alternatives.json`의 해당 필드를 `configs/local.json`의 `llm`에 넣으면 Qwen 4B 또는 ChemDFM-v2로 교체할 수 있습니다. 실행할 때 하나만 로드합니다. Qwen은 `Qwen3_5ForConditionalGeneration`, ChemDFM-v2는 Qwen2.5 기반 `AutoModelForCausalLM`을 사용합니다. 기본은 비양자화 BF16(Qwen)/FP16(ChemDFM) 및 SDPA이며, `int4`는 별도 bitsandbytes 설치가 필요한 미검증 옵션입니다. 로컬 tokenizer의 non-thinking template 적용은 CPU에서 확인했습니다.

FLOWR.root checkpoint 경로는 이미 `configs/local.json`에 지정되어 있습니다. 실행 환경 준비 후 최소 설정은 다음과 같습니다.

```bash
# docs/installation.md의 두 Conda 환경을 설치한 뒤 지정합니다.
export DIRMO_LLM_PYTHON=/home/uslee/miniconda3/envs/dirmo-llm/bin/python
export DIRMO_CPU_PYTHON=/home/uslee/miniconda3/envs/dirmo/bin/python
export DIRMO_FM_PYTHON=/home/uslee/miniconda3/envs/dirmo/bin/python
# 다른 checkpoint를 사용할 때만 DIRMO_FM_CHECKPOINT로 경로를 덮어씁니다.
# 사용 가능한 두 장을 확인한 뒤 각각 다른 실제 UUID를 지정합니다.
export DIRMO_LLM_GPU=GPU-<LLM_UUID>
export DIRMO_FM_GPU=GPU-<FM_UUID>
bash scripts/dirmo doctor
bash scripts/dirmo start-workers
bash scripts/dirmo health
bash scripts/dirmo optimize --manifest examples/local_case.json --output runs/rule --router rule
bash scripts/dirmo stop-workers
```

`GPU-<…>`는 설명용 자리표시자입니다. 위 Python 경로는 이 서버의 Conda 기본 env 위치 기준이며, 다른 위치를 사용하면 실제 환경 경로로 바꾸세요. `fm.python`은 기본 config에서 `null`입니다. 호환 환경의 Python 경로를 이 필드 또는 `DIRMO_FM_PYTHON`으로 지정하세요. `fm.checkpoint_path`는 제공된 파일의 실제 경로이며, `checkpoint_sha256`이 `null`이면 worker가 로드 시 실제 파일의 SHA256을 계산해 kernel identity에 기록합니다. 설정의 physical GPU 기본값 0/1도 이번 세션에서 사용 승인을 받은 장치가 아니며, 실제 실행 전에 지정해야 합니다. Worker는 UUID로 CUDA mask를 전달받고 내부에서 `cuda:0`만 사용합니다. 같은 physical GPU로 해석되는 두 selector는 거부합니다. `minimum_free_mib` 검사로 현재 다른 작업의 VRAM 사용을 고려합니다.

서버의 NVIDIA 드라이버는 550.144.03입니다. LLM requirements는 Torch 2.6.0+cu124를 지정하고, training/inference requirements는 upstream의 Torch 2.11 / tensordict 0.12.4 / biotite 1.7.1 조합을 유지하며 Torch GPU wheel을 2.11.0+cu126으로 지정합니다. 후자는 CUDA 12.x minor-version compatibility에 의존하며 모델 실행은 미검증입니다. 전체 uv lock의 정확한 재현은 아니며, 나머지는 upstream의 dependency 범위로 pip가 해결합니다. DIRMO worker는 소스를 `PYTHONPATH`와 별도로 정확한 upstream 경로에서 로드하고, 다른 `flowr` package로의 shadowing을 검사합니다.

구현된 CLI는 다음과 같습니다.

| 명령 | 역할 |
| --- | --- |
| `doctor` | 로컬 모델 경로·revision·환경·GPU·FM blockers 확인 |
| `start-workers`, `stop-workers`, `health` | 모델을 한 번 로드하는 장당 1개 resident worker 관리 |
| `evaluate-initial` | 고정 pocket을 준비하고 초기 ligand의 실제 CPU 평가 저장 |
| `optimize` | rule/learned/LLM/fixed 정책으로 예산 내 반복 |
| `collect` | 동일 state와 공통 seed에서 실제 실행한 branch만 JSONL로 저장 |
| `train-router` | action-conditioned MLP의 개선량·비용 회귀, grouped CV 및 bootstrap |
| `compare` | 동일 executor·목표·예산 상한·seed 규약으로 정책별 별도 run |

```bash
bash scripts/dirmo collect --manifest examples/local_case.json --output runs/branches --dataset runs/branches.jsonl
# 실제 관측 데이터가 있어야 합니다. 여러 kernel이 있다면 하나의 --kernel-id를 지정합니다.
# 기본 FM GPU 학습은 FM worker를 먼저 종료하고 별도 프로세스를 띄웁니다.
bash scripts/dirmo train-router --dataset runs/branches.jsonl --output runs/router.json --device fm
# CPU 검증/소규모 학습도 선택 가능합니다.
bash scripts/dirmo train-router --dataset runs/branches.jsonl --output runs/router.json --device cpu
bash scripts/dirmo start-workers
bash scripts/dirmo optimize --manifest examples/local_case.json --output runs/learned --router learned --router-model runs/router.json
bash scripts/dirmo compare --manifest examples/local_case.json --output runs/compare --policies rule edit pose sequence llm
bash scripts/dirmo optimize --manifest examples/local_case.json --output runs/rule --router rule --resume
```

Branch 데이터가 없으므로 이번에는 학습된 gate artifact를 만들지 않았습니다. 단일 target 데이터는 학습 CLI를 확인할 수 있지만 held-out 일반화 평가를 만들지 않습니다. 학습되지 않은 행동 종류와 다른 kernel의 조건 변경은 learned policy에서 제외합니다. Regression 출력은 성공 확률이 아닙니다. 일부 seed에서 측정이 실패한 branch에는 보상값을 만들어 넣지 않습니다.

입력은 JSON 또는 JSONL manifest입니다. `case_id`, `target_id`, protein/pocket 경로, 정렬된 3D ligand SDF, objective, protection, budgets를 받습니다. 상대 입력 경로는 manifest 위치 기준입니다. 목표의 방향·target·weight·허용 손실은 시작 시 고정합니다. 예제는 기존 로컬 CrossDocked pocket·SDF·PDBQT를 참조합니다. Pocket는 initial pose 기준으로 한 번 준비해 고정하고, Vina box도 initial 좌표에서 한 번 정합니다. 전체 protein이 있으면 `protein_path`로 별도 제공합니다. 초기 SDF는 첫 record를 사용합니다.

보호 원자는 기존 complete atom map을 사용하고, 없으면 initial heavy-atom 순서에 1부터 번호를 부여합니다. `inputs/initial.sdf`로 확인할 수 있습니다. 보호 SMARTS, RDKit feature의 종류·원자 map·절대 위치, ProLIF residue/role 조건을 각각 지정할 수 있습니다. 기본 예제에는 필수 보호 조건을 자동 추가하지 않습니다. 좌표 보존은 protein 좌표계에서 직접 거리로 평가하고 ligand 정렬을 하지 않습니다. ProLIF는 준비된 explicit-H protein이 필요하며 기본 비활성입니다.

FLOWR 경로는 whole-graph `random` prior + Euler만 노출합니다. 마지막 graph inpainting은 원자·결합·전하 등 discrete feature만 고정하며 좌표를 고정하지 않습니다. Draw 하나당 원소·개수·연결·결합 차수·전하·동위원소·전체 stereo identity를 검증합니다. Upstream tensor round-trip에서 보존되지 않는 graph는 생성 전에 거부합니다. RDKit conformer는 편집 graph의 **입력 준비용**으로만 저장하며 최종 FM pose로 취급하지 않습니다. Raw 출력과 map을 부여한 출력은 따로 저장합니다. 자동 repair·relaxation·SMILES uniqueness filtering·내부 oversampling은 사용하지 않습니다.

`CHANGE_GENERATION_CONDITION`은 Euler integration step 수만 변경합니다. `generate_local_fragment`, LiFT semantic comparison, scalar guidance, conformer/harmonic prior, midpoint, `EXTRA_VALIDATION`, relaxation/redocking은 미지원으로 명시합니다. LiFT의 placeholder는 capability error를 반환하며 실제 비교 모델을 구현했다고 주장하지 않습니다.

평가는 RDKit graph properties(QED, RDKit contrib SA, MW/logP/HBD/HBA/TPSA/rotatable bonds/Ro5), 자체 geometry/contact screen, PoseBusters `dock`, Vina **score-only**를 구분합니다. SA는 낮을수록 좋습니다. PoseBusters에는 reference ligand identity/RMSD를 적용하지 않습니다. Raw metric과 목표 가중 utility를 따로 저장합니다. 필수 tool이 누락되거나 실패하면 `missing`/`failed`이고 기본적으로 feasible 후보가 아닙니다. 평가를 끈 demo는 `incomplete_evaluation`을 표시합니다. 외부 receptor preparation의 provenance는 사용자가 기록해야 합니다.

예산에서 `evaluation_calls`는 graph/pose/보호/Vina 등을 포함한 **한 pose 평가 bundle**을 뜻합니다. FM raw draws와 실제 model forward NFE, LLM input+output tokens/호출, evaluation bundles, 전체 wall time을 각각 기록합니다. RPC 전에 비용 상한을 예약하고 실패·OOM·parsing 재시도도 과금합니다. 이미 batch가 1이므로 OOM에서 자동 재시도를 하지 않습니다. Worker 요청의 timeout은 잔여 wall time으로 제한합니다. CPU 평가 bundle의 내부 tool 실행 시간은 완료 후 측정하므로 단일 외부 평가가 deadline을 넘길 수 있는 제한이 있습니다.

각 run은 `manifest.json`, `resolved_config.json`, `inputs/`, `observations.jsonl`, `llm.jsonl`, `actions.jsonl`, `generation/…/{raw.sdf,raw_mapped.sdf,kernel.json}`, `costs.json`, `checkpoint.json`, `result.json`, `final.sdf`를 저장합니다. Model/checkpoint SHA와 revision, 환경·코드·입력 해시, GPU health, peak memory, latency가 실제 실행 시 채워집니다. LLM 캐시는 모델 SHA·revision·generation config·prompt를 키로 하고 cache hit를 별도 기록합니다. Pose는 SMILES 중복으로 제거하거나 캐시하지 않습니다.

Resume은 clean action boundary에서 일관성을 검사합니다. 진행 중이던 원격 요청의 결과를 모르면 자동으로 다시 실행하지 않고 `pending_action`을 확인하도록 중단합니다. 성공은 history에서 target과 보호 조건을 만족하고 초기값보다 최소 utility를 개선한 후보를 선택합니다. 그런 후보가 없으면 original SDF와 `no_improvement`를 반환합니다.

책임별 코드는 `src/dirmo/{chemistry,llm,generation,evaluation,diagnosis,routing,pipeline,data,training}`에 있습니다. 상세 확인 사항과 미검증 경로는 [backbone audit](docs/backbone_audit.md), 검토 결과는 [review](docs/review.md), 측정 로그는 [reports](reports/), 원문 요구사항은 [implementation request](docs/implementation_request.md)에 기록했습니다.
