# SMDD-Bench instruction tuning pilot

이번 실험은 SMDD-Bench task 4의 `reference.sdf → witness.smi` 쌍으로 Qwen3.5-9B의
LoRA adapter를 학습합니다. 기존 DIRMO router 학습과는 별도 실험입니다.

## 코드 위치

| 파일 | 역할 / 변경할 설정 |
|---|---|
| `src/dirmo/data/smdd.py` | pair 로딩, **`SYSTEM_PROMPT`**, 입력 구성 **`make_messages()`**, pocket 측정, 데이터 분할, 출력 검증 |
| `scripts/prepare_smdd_data.py` | 데이터 준비 CLI: 경로, split seed |
| `src/dirmo/training/smdd_sft.py` | chat template, completion-only loss, **LoRA target modules**, optimizer, base/adapter 검증 |
| `scripts/train_smdd_lora.py` | epochs, rank, learning rate, accumulation, token 길이, GPU 메모리 제한 CLI |
| `scripts/analyze_smdd_predictions.py` | 저장된 응답에서 형식 문제와 분자 문제를 분리해 분석 |
| `scripts/evaluate_smdd_lora.py` | 저장된 adapter 또는 base의 별도 생성 평가 |
| `requirements-sft.txt` | 신규 Python 3.11 환경의 학습 의존성 |
| `reports/smdd-lora-pilot/build_report.py` | 완료된 결과의 표와 학습 곡선 생성; matplotlib 사용 |
| `tests/test_smdd.py` | 전이적 split 누출, prompt 정보 누출, FASTA chain 선택, 출력 검증, loss 위치 검증 |

## 데이터와 prompt

원본: `/LocalData1/uslee/datasets/smdd-bench/tasks/smdd_004_*/`.
340개 쌍 / pocket 단백질 25종이 있습니다. 단백질 서열, reference 또는 witness의 동일 분자,
비어 있지 않은 Bemis–Murcko scaffold 중 하나라도 공유하면 연결된 전체 그룹을 같은 split에 둡니다.
seed 42에서는 train 272개/18종, validation 34개/4종, test 34개/3종입니다.
이 세 종류의 키에 대한 split 간 겹침은 0입니다. 서열이 비슷하지만 완전히 같지 않은 단백질의
homology clustering은 추가하지 않았으므로 이를 원격 상동 단백질 일반화 검증으로 해석하지 않습니다.

70개 `witness.sdf`에는 공식 `witness.smi`에 없는 입체화학 표기가 추가되어 있습니다.
공식 SMILES를 정답으로 사용하며 SDF와의 원자 연결성만 대조합니다.

입력 `messages`는 system message 하나와 JSON user message 하나입니다.

```json
{
  "reference_mapped_smiles": "[CH3:1][CH2:2][OH:3]",
  "goals": [{"property":"cyp3a4","direction":"minimize","threshold":0.1,"baseline":0.6862}],
  "hold": [{"property":"binding_affinity","direction":"increase","tolerance":0.3,"baseline":-0.7193}],
  "hard_constraints": {"morgan_r2_2048_tanimoto_ge":0.7},
  "pocket": {
    "residue_fields":["chain","seqres_position","aa","reference_min_heavy_distance_A"],
    "residues":[["A",63,"I",3.8]]
  }
}
```

위 예시는 필드를 설명하기 위해 축약한 형식 예시이며 실제 분자/목표와 연결된 과제가 아닙니다.
실제 입력은 전체 hard constraint와 해당 과제의 모든 목표·유지 조건·pocket residue를 포함합니다.
`threshold`는 절대 목표값이 아니라 **baseline에서 변화해야 하는 최소량**입니다.
hold의 `increase`는 증가량 상한, `decrease`는 감소량 상한, `change`는 양방향 변화량 상한입니다.
원래 과제의 예측 ADMET/Boltz 수치를 그대로 사용하고 임의의 utility를 정답으로 만들지 않습니다.
`binding_affinity`는 SMDD의 Boltz 예측값이며 Vina 점수가 아닙니다.

Pocket residue 번호는 1-based SEQRES 위치입니다. FASTA header의 chain을 읽으며 record 순서에
의존하지 않습니다. `complex.pdb`의 대응 잔기 종류가 FASTA와 일치할 때만, 해당 잔기와
현재 reference ligand 사이 최소 heavy-atom 거리를 넣습니다. 불일치하면 거리는 `null`입니다.
이 거리는 현재 pose의 접촉 정보이며 결합 종류나 편집 효과를 단정하는 근거가 아닙니다.

정답은 설명 없이 JSON 한 개입니다.

```json
{"product_smiles":"<canonical isomeric witness SMILES>"}
```

첫 실험은 1:1 pair의 **분자 생성/편집 학습**입니다. 후보 ID 선택 학습에는 여러 후보가 필요하므로
`selected_edit_ids`를 이번 정답으로 쓰지 않습니다. 입력 reference의 atom map은 현재 분자를
구분하는 데 사용하며 출력은 atom map 없는 SMILES입니다. witness 구조·물성·미래 측정 결과는
user prompt에 들어가지 않습니다. 저장된 row의 `witness_metrics` 등은 분석용 annotation입니다.

## 실제 input/output 예시

`runs/smdd_lora_02/examples/`는 train 첫 번째 과제 `smdd_004_A2A_Adenosine_0`을 그대로
분리해 저장한 폴더입니다. 학습기는 이 examples 폴더를 읽지 않고 `data/train.jsonl`을 읽습니다.

- `input_messages.json`: 실제 system/user messages 전체.
- `input_user.json`: user JSON을 들여쓰기하여 표시한 파일.
- `output.json`: official witness에서 만든 정답.
- `training_example.json`: 원본 training row와 annotation 전체.
- `training_chat.json`: 정답 assistant message까지 포함한 대화 예시.
- `rendered_prompt.txt`: 실제 Qwen chat template 적용 후 입력 문자열.
- `training_completion.txt`: loss를 적용하는 정답 문자열과 종료 token.

이 과제의 목표는 CYP3A4 예측값을 baseline 0.6862에서 최소 0.1 낮추는 것입니다.
정답은 phenyl ring의 methyl 치환기가 제거된 아래 분자입니다.

```json
{"product_smiles":"CCCn1c(=O)c2[nH]c(-c3ccccc3)cc2n(CCC)c1=O"}
```

원본 데이터에서 pair를 구성하는 시점은 `prepare_smdd_data.py`를 실행할 때입니다.
`train_smdd_lora.py`는 저장된 row의 `messages`와 `answer`를 읽으며 매 epoch 원본 pair를
다시 탐색하거나 witness를 user input으로 넣지 않습니다.

## 학습 설정

- Base model: 로컬 `Qwen3.5-9B`, BF16, SDPA, thinking 비활성화.
- `tokenizer.apply_chat_template(..., enable_thinking=False)`로 추론과 학습 prompt를 동일하게 만듭니다.
- Loss: assistant의 정답 JSON과 종료 token만 next-token cross entropy로 학습합니다.
  system/user prompt 자체를 정답으로 복사하는 loss는 없습니다.
- LoRA: rank 8, alpha 16, dropout 0.05. text backbone의 full attention과 Gated DeltaNet
  projection에 적용합니다. vision tower, embedding, LM head는 학습하지 않습니다.
- Microbatch 1, gradient accumulation 8, gradient checkpointing, AdamW, 최대 learning rate `1e-4`.
- 첫 pilot: 1 epoch, 34 optimizer updates. validation loss로 best adapter를 저장합니다.
- 입력/정답이 token cap을 초과하면 과제 정보를 자르는 대신 오류로 중단합니다.
- GPU 4의 메모리 할당을 전체의 60%로 제한합니다. 기존 작업 메모리가 변하면 OOM이 발생할 수
  있으며 다른 작업을 종료하거나 다른 GPU로 자동 이동하지 않습니다.
- Qwen의 optional DeltaNet kernels가 없는 환경에서는 PyTorch fallback을 사용합니다.

## 실행

기존 `dirmo-llm` 환경에는 torch 2.6.0+cu124 / transformers 5.15.0이 설치되어 있습니다.
이번에는 이를 공유하는 별도 venv에 PEFT만 추가했습니다. 기존 conda 환경의 패키지는 변경하지
않았으며, `.venv-sft/bin/python`은 `conda activate` 없이 호출할 수 있습니다.

```bash
cd ~/code/DIRMO-3D
/home/uslee/miniconda3/envs/dirmo-llm/bin/python -m venv --system-site-packages .venv-sft
.venv-sft/bin/python -m pip install --no-deps peft==0.21.2
.venv-sft/bin/python -m pip check
```

별도 Python 3.11 conda 환경을 만들 경우에는 `python -m pip install -r requirements-sft.txt`를
사용합니다. 공유 venv에서 전체 requirements를 다시 설치하면 상위 환경과 다른 dependency가
설치될 수 있으므로 이번 설치 방식을 그대로 재현할 때에는 위 PEFT 설치 명령만 사용합니다.

```bash
.venv-sft/bin/python scripts/prepare_smdd_data.py \
  --output runs/smdd_lora_03/data

CUDA_VISIBLE_DEVICES=4 .venv-sft/bin/python scripts/train_smdd_lora.py \
  --data runs/smdd_lora_03/data \
  --output runs/smdd_lora_03/training \
  --epochs 1 --rank 8 --learning-rate 1e-4 --accumulation 8 \
  --eval-generations 8
```

기존 출력 폴더를 덮어쓰지 않습니다. 새 실험은 새 디렉터리를 사용합니다.
prompt를 바꾼 뒤에는 data를 다시 준비해야 합니다. 학습기는 JSONL에 저장된 `messages`를 읽습니다.

## 검증과 로그

공식 SMILES 정답으로 진행한 실험 출력은 `runs/smdd_lora_02/`입니다.
`smdd_lora_01`은 SDF 정답을 사용해 시작했다가 중단한 실행이며 `ABORTED.json`에 이유를 기록했습니다.

| 경로 | 내용 |
|---|---|
| `data/{train,validation,test}.jsonl` | 입력 messages, 정답, 출처, 분석 annotation |
| `data/manifest.json` | split 크기, 중복 검사, witness의 로컬 제약 검사 |
| `training.log` | 실행 전체 stdout/stderr 및 오류 traceback |
| `training/events.jsonl` | base 측정, 학습 update별 loss, epoch 검증, adapter 측정 |
| `training/settings.json` | 실행 인자, GPU UUID, 패키지 버전, 데이터 hash, 생성 검증 sample ID |
| `training/adapter/` | best validation-loss PEFT adapter와 tokenizer |
| `training/base_predictions.jsonl` | 학습 전 생성 결과와 검사 |
| `training/adapter_predictions.jsonl` | 동일 validation prompt에 대한 학습 후 결과와 검사 |
| `training/report.json` | 완료 시 학습/검증 결과 |
| `training/failure.json` | 실패 시 오류 요약 |

```bash
tail -f runs/smdd_lora_02/training.log
```

Validation loss는 34개 모두에서 정답 completion token으로 계산합니다. 생성 검증은 seed 42로
출력을 보기 전에 선택한 validation 8개에 대해 greedy decoding으로 base와 adapter를 비교합니다.
JSON 준수, SMILES 유효성, reference 유사도, RDKit 물성/필터 제약, 실제 변경 여부와 witness
exact match를 보고합니다. 같은 분자를 그대로 반환하면 일부 제약을 통과할 수 있지만 목표를
달성했다는 뜻은 아닙니다. witness exact match도 유일한 성공 기준이 아닙니다.

Test 34개는 학습기에서 읽지 않고 추후 최종 평가용으로 남깁니다.
ADMET-AI 목표/유지 조건과 Boltz binding probability는 이 pilot에서 새 출력에 대해 측정하지
않으므로 보고서에서 `null`로 남깁니다. `local_hard_constraints_passed`를 전체 benchmark 성공률로
해석하면 안 됩니다. 다음 단계는 이 oracle들을 연결한 평가와 여러 후보를 이용하는 학습입니다.

학습이 완료된 뒤 저장된 adapter의 validation 전체를 별도로 평가하려면:

```bash
CUDA_VISIBLE_DEVICES=4 .venv-sft/bin/python scripts/evaluate_smdd_lora.py \
  --data runs/smdd_lora_02/data --split validation \
  --adapter runs/smdd_lora_02/training/adapter \
  --output runs/smdd_lora_02/full_validation
```

`--adapter`를 생략하면 base model을 평가합니다. `--split test`는 최종 test 평가를 명시적으로
선택할 때 사용합니다. 이번 학습 실행에서는 이 test 명령을 실행하지 않습니다.

저장된 출력에서 Markdown fence만 제거해 형식 문제와 분자 문제를 분리하려면:

```bash
.venv-sft/bin/python scripts/analyze_smdd_predictions.py \
  --data runs/smdd_lora_02/data/validation.jsonl \
  --predictions runs/smdd_lora_02/training/base_predictions.jsonl \
  --output runs/smdd_lora_02/base_diagnostic.json
```

SMILES 문법, 결합, 원자 종류, 잘린 응답은 자동 보정하지 않습니다.

이 adapter는 `product_smiles` 응답을 학습하며 기존 서버의 `Proposal` 응답과 형식이 다릅니다.
현재 LLM worker에 바로 적용하려면 응답 변환 및 atom correspondence 검증을 추가해야 합니다.

참고: [SMDD-Bench](https://github.com/t7rs/SMDD-Bench),
[Qwen3.5 Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5),
[PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora).
