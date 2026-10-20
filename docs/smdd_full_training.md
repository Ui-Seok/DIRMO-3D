# SMDD-Bench 전체 학습·검증·테스트

이번 실행은 `runs/smdd_lora_02`의 완료된 1-epoch adapter에서 추가 학습했습니다.
SMDD-Bench lead optimization 분자 쌍 340개를 train 272개, validation 34개, test 34개로 나누었고
모든 row를 학습 또는 평가에 사용했습니다. [전체 결과](../reports/smdd-full/REPORT.md)에 표와 그래프가 있습니다.

## 실행 설정

- Base: 로컬 Qwen3.5-9B, BF16, rank-8 LoRA, frozen base/vision tower.
- GPU: physical GPU 4. 메모리 상한 90%, training microbatch 8, 생성 batch 4.
- 기존 1 epoch + 선택된 LR의 1 epoch + epoch 3/4/5 = **선택 경로에서 총 5 epoch**.
- LR 후보 `2e-5`, `5e-5`는 동일한 1-epoch adapter에서 각각 시작합니다.
  두 후보 모두 train 272개 전체를 한 번 학습하고 validation 34개 전체를 생성 평가합니다.
  선택되지 않은 후보의 추가 학습은 최종 경로의 epoch 수에 합산하지 않습니다.
- 각 epoch 시작에 5% warmup 후 해당 epoch의 LR을 유지합니다.
  선택 후 epoch 3/4/5에는 LR을 선택 값의 `0.75`, `0.75²`, `0.75³`로 낮춥니다.
- 초기 pilot은 optimizer 상태를 저장하지 않았으므로 adapter 가중치에서 warm start합니다.
  선택된 LR trial의 optimizer 상태는 후속 epoch로 이어집니다.
- 사용자의 요청에 따라 **validation 악화로 조기 종료하지 않습니다**.
  모든 epoch를 진행하고 best checkpoint와 마지막 checkpoint를 따로 보관합니다.

## 학습과 평가의 차이

`messages`는 입력, `answer`는 witness SMILES 정답입니다. Loss는 정답 completion에만 적용합니다.
학습은 padding을 오른쪽에 추가하고 completion prediction 위치만 vocab head로 계산합니다.
생성은 왼쪽 padding을 사용하고 모델이 자기 출력에 이어서 token을 생성합니다.
Length bucketing으로 padding을 줄이며 매 epoch 모든 train row를 정확히 한 번 사용합니다.

Validation checkpoint 선택 기준은 사전에 다음 순서로 고정했습니다.

1. Reference를 실제로 변경하고 로컬 화학 제약을 모두 통과한 출력 수.
2. 유효한 SMILES 수.
3. 위 수가 같으면 completion loss가 낮은 checkpoint.

화학 검사는 완전한 Markdown fence만 제거한 진단 결과로 비교합니다.
JSON 형식의 엄격한 검사도 별도 기록하며 분자 문법·구조를 자동 수정하지 않습니다.
이 기준의 `local`은 RDKit 물성/필터/유사도 제약을 뜻합니다.
ADMET-AI 목표/유지 조건과 Boltz binding probability는 포함되지 않습니다.
따라서 위 통과율을 SMDD 최적화 성공률로 부르면 안 됩니다.

Test 파일은 best checkpoint를 고정한 뒤 열며 이후 가중치를 갱신하지 않습니다.
최종 base와 best adapter 모두 **train 272 / validation 34 / test 34 전체**를 생성 평가합니다.
별도 `evaluate_smdd_checkpoint_all.py`로 마지막 5-epoch adapter도 모든 split에서 평가합니다.
두 checkpoint는 test 결과를 보기 전에 고정하며 이 비교로 test에 맞춰 모델을 다시 선택하지 않습니다.
Train의 결과는 학습된 데이터에 대한 fitting 진단이고, 일반화 평가는 validation/test에서 합니다.
Epoch별 checkpoint 비교에서는 출력 형식·분자 유효성·로컬 제약과 graph metric delta를 측정합니다.
최종 checkpoint를 고정한 뒤 ADMET-AI로 모든 split의 유효한 출력을 재평가했습니다.
결합 조건은 앞 단계 조건을 통과한 held-out 후보에 Boltz로 별도 검증합니다.

## 코드와 실행

- `src/dirmo/training/smdd_full.py`: batch loss, 생성, LR 비교, 추가 학습, checkpoint 선택, 전체 평가.
- `scripts/train_smdd_full.py`: 실행 인자.
- `scripts/evaluate_smdd_checkpoint_all.py`: 고정된 마지막 checkpoint의 전체 split 평가.
- `scripts/evaluate_smdd_admet.py`: 전체 split의 목표·유지 조건, baseline/witness 재현 audit.
- `scripts/evaluate_smdd_boltz.py`: 앞 단계 조건을 통과한 held-out 후보의 결합 조건 검증.
- `src/dirmo/data/smdd.py`: 기존 system prompt와 input 구성.
- `tests/test_smdd_full.py`: batch label 위치, padding, 데이터 coverage, 선택 기준 검사.

```bash
cd ~/code/DIRMO-3D
CUDA_VISIBLE_DEVICES=4 .venv-sft/bin/python scripts/train_smdd_full.py \
  --output runs/smdd_full_01/experiment \
  --total-epochs 5 --learning-rates 2e-5 5e-5 \
  --microbatch 8 --generation-batch 4 --memory-fraction 0.90 \
  > runs/smdd_full_01/training.log 2>&1
```

상위 `runs/smdd_full_01` 폴더가 먼저 있어야 로그를 redirect할 수 있습니다.
기존 출력 폴더를 덮어쓰지 않으므로 새 실행은 새 output을 지정합니다.

## 산출물

```text
runs/smdd_full_01/
├── training.log
└── experiment/
    ├── manifest.json
    ├── events.jsonl
    ├── epoch_01_validation/
    ├── lr_trials/
    │   ├── lr_2e-05/{adapter,validation}/
    │   └── lr_5e-05/{adapter,validation}/
    ├── epoch_03_validation/
    ├── epoch_04_validation/
    ├── epoch_05_validation/
    ├── checkpoints/
    ├── best_adapter/
    ├── last_adapter/
    ├── lr_trials.json
    ├── history.json
    ├── selection.json
    ├── final/
    │   ├── adapter/{train,validation,test}/
    │   ├── base/{train,validation,test}/
    │   └── last_adapter/{train,validation,test}/
    ├── full_results.json
    └── report.json
```

각 평가 폴더의 `predictions.jsonl`에 모든 입력 과제의 응답 원문과 검사 결과가 있고,
`report.json`에는 전체 및 단백질별 집계가 있습니다. LR은 비교한 후보 내에서 선택한 값입니다.

## PLIP 특징을 추가한다면

PLIP은 PDB complex에서 수소결합, 소수성 접촉, salt bridge, pi stacking 등의 상호작용을
규칙으로 검출합니다. 거리만 있는 현재 pocket 요약보다 결합 유형과 관여 원자를 설명하는
입력 특징으로 사용할 수 있습니다. [공식 문서](https://github.com/pharmai/plip/blob/master/DOCUMENTATION.md)

추천 입력은 **reference complex의** 상호작용 유형, ligand atom map, 단백질 잔기,
거리·각도, 사용한 도구 버전과 좌표 출처입니다. OpenBabel/PDB atom 번호를 RDKit atom map으로
검증해 연결해야 합니다. 원자 번호를 그대로 같다고 가정하면 안 됩니다.
Witness의 상호작용과 변경 후 물성은 분석/정답 annotation에 둘 수 있지만 현재 ligand의
input에 넣으면 미래 정보 누출이 생깁니다.

PLIP의 상호작용 수는 결합 친화도나 ADMET 목표 달성 여부를 대신하는 값이 아닙니다.
SMDD의 complex가 예측된 pose라는 점과 protonation/geometry 처리 방법도 기록해야 합니다.
이번 실행은 기존 입력을 유지합니다. PLIP 추가 효과는 같은 split에서 특징 유무를 비교하는
별도 학습 실험으로 확인해야 합니다. 현재 SMILES 문법 문제에 도움이 된다고 단정할 수 없습니다.

## ADMET oracle 추가 평가

`requirements-admet.txt`와 `.venv-admet`는 SMDD가 사용하는 ADMET-AI **1.4.0**용 별도 CPU 환경입니다.
GPU의 Qwen 환경은 변경하지 않습니다. V2는 다시 학습된 모델로 예측값이 달라 기존 baseline과
직접 비교하지 않습니다. [ADMET-AI 공식 설명](https://github.com/swansonk14/admet_ai/blob/main/README.md)

`scripts/evaluate_smdd_admet.py`는 전체 checkpoint 평가가 완료된 뒤 실행합니다.
단백질 binding 관련 조건은 미측정으로 남기고, 원래 목표와 hold의 방향·변화량 기준을 적용합니다.
Reference의 재예측값과 원본 baseline 차이, witness의 ADMET 조건도 별도 audit에 기록합니다.
이 oracle은 checkpoint 선택에 사용하지 않습니다.

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-admet/bin/python scripts/evaluate_smdd_admet.py \
  --experiment runs/smdd_full_01/experiment \
  --output runs/smdd_full_01/admet
```

`pending_boltz.json`은 로컬·ADMET 조건을 통과해 Boltz 검증이 추가로 필요한 후보입니다.
`full_success_known_false`는 이미 필수 조건 하나 이상을 실패한 출력이고,
`full_success_pending_boltz`는 결합 관련 조건을 아직 확인하지 못한 출력입니다.

ADMET에는 생성 응답의 원문 `product_smiles`를 사용하며 graph 검사용 canonical SMILES로
재정렬하지 않습니다. Legacy Chemprop의 local ChiralTag 특징 때문에 같은 분자의 표기가 예측값에
영향을 줄 수 있습니다. Canonical 입력의 이전 계산은 `admet_canonical_diagnostic/`에 보관했습니다.
원본 witness.smi 340개 중 9개도 현재 oracle에서 일부 목표/hold 조건을 재현하지 못했습니다.
차이의 원인은 확정하지 않았습니다. 임계값은 원본 그대로 유지했고 재현 차이는 audit에 기록했습니다.

## Boltz held-out 평가

별도 `.venv-boltz` 환경과 `requirements-boltz.txt`를 사용합니다.
Boltz 2.2.1, CUDA 12.4용 torch 2.6.0을 사용하며 optional CUDA extension kernels는 비활성화합니다.
Boltz의 click 고정 버전과 상속한 Hub CLI의 충돌은 이 환경의 Hub 버전을 고정해 해결했습니다.
Qwen training/serving 환경은 유지됩니다.

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-boltz/bin/python scripts/evaluate_smdd_boltz.py \
  --memory-fraction 0.60 --splits validation test
```

첫 실행은 다른 GPU 작업과 공유하기 위해 allocator 상한을 GPU 전체의 27%로 제한했습니다.
BRD4는 완료됐지만 BRAF 구조 생성이 그 상한에서 OOM을 일으켰습니다.
이후 GPU 4에 약 34GiB가 비어 있어서 상한을 60%로 올려 BRAF를 같은 조건으로 재실행했습니다.
첫 시도는 `boltz_heldout/report_first_attempt.json`에 보관했습니다.
재실행은 완료된 후보를 재사용합니다. 현재 여유 메모리를 확인하고 상한을 설정해야 합니다.
구조/affinity sample 각각 10개, sampling step 각각 200, seed 42, pocket potentials를 유지합니다.
FASTA의 모든 chain과 task.yaml의 정확한 baseline_affinity를 사용합니다.
단백질 MSA는 Boltz가 공식 MSA server에서 계산합니다.

앞 단계에서 필수 조건을 하나라도 실패한 출력은 Boltz를 실행해도 전체 성공이 될 수 없어
실행을 생략하며 전체 평가 분모에는 포함합니다. Train의 남은 후보는 fitting 진단으로 보관하고
held-out 결합 평가에 포함하지 않습니다. 결합 계산 실패는 성공/실패 대신 `unmeasured`로 기록합니다.
로그는 `runs/smdd_full_01/boltz_heldout/<model>_<task>/boltz.log`,
집계는 `runs/smdd_full_01/boltz_heldout/report.json`에 저장합니다.

이 결과는 한 번의 greedy 응답을 평가한 instruction tuning 실험입니다.
공식 multi-turn agent benchmark 점수나 실험적 결합 친화도 검증으로 해석할 수 없습니다.
