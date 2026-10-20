# 학습 없는 분자 편집·도구 피드백 실험

기존 Qwen3.5-9B HTTP worker를 그대로 사용한다. LoRA나 encoder를 추가하지 않고
validation 34개 전체에 대해 짧은 편집 가설과 완성된 분자를 생성한다.
이 실험은 SMDD molecular editing 루프이며 FLOWR pose 생성이나 Vina 평가는 호출하지 않는다.
실제 결과는 [실험 보고서](../reports/smdd-feedback/REPORT.md)에 기록한다.

## 처리 과정

1. Original ligand의 RDKit 물성과 ADMET-AI 1.4.0 예측을 계산한다.
2. Current ligand의 mapped SMILES, 원자 연결, 작용기, 원자별 Crippen logP 기여를 구조화한다.
3. LLM이 원자 ID, 한 문장 편집 가설, 편집된 전체 `product_smiles`를 반환한다.
4. RDKit이 SMILES를 실제 분자 그래프로 만들고 물성·필터·original 대비 Tanimoto를 검사한다.
5. 분자가 변경되었고 로컬 조건을 통과하면 ADMET-AI로 목표·유지 조건을 검사한다.
6. 실패한 조건과 계산값을 LLM에 다시 보내 수정한다. 최대 3회 제안한다.
7. 선택된 RDKit·ADMET 통과 후보만 별도 Boltz 단계에서 결합 조건을 검사한다.

LLM이 적는 편집 가설은 검증 전 가설이다. 자유로운 SMILES 생성이므로 사전에 나열된
C/N/F 후보 중 하나를 고르는 방식이 아니다. 선언한 원자 위치와 실제 graph 변화의
정확한 일치는 현재 자동 검증하지 않는다. 실제 product graph와 수치 조건은 도구로 검사한다.
Crippen 기여는 계산 모델의 원자 기여이며 ADMET 변화나 결합 성능의 인과 증거는 아니다.

매 라운드 current molecule의 원자 번호를 새로 1부터 부여한다. Original ligand의 유사도 기준과
공식 목표 baseline은 수정 후에도 바뀌지 않는다. Pocket 거리는 reference 구조에서 얻은 값이며
생성된 분자의 3D 상호작용으로 해석하지 않는다. Witness, witness 물성, 정답, task 경로와 ID는
LLM 입력에서 제외한다. Test와 train은 이번 개발 실험에서 사용하지 않는다.

## 입력과 출력

입력 JSON에는 다음 필드가 있다.

- `original_molecule`: original SMILES, RDKit 물성, 현재 oracle로 다시 예측한 ADMET 값.
- `current_molecule`: mapped SMILES, 물성, 원자 표, 작용기, 측정했으면 ADMET 예측.
- `goals`, `hold`: 공식 original baseline 기준 개선량과 허용 악화량.
- `hard_constraints`, `reference_pocket`: SMDD 제약과 reference pocket 정보.
- `proposal_number`: 제안 순서.
- `tool_feedback`: 수정 방식의 두 번째 제안부터 실제 검사 결과와 이전 출력 목록.

아래는 출력 형식만 보여주는 예시다. 실제 입력·출력은 실험 보고서의 예시 파일에 있다.
번호는 반드시 해당 입력의 current molecule에 있는 번호여야 한다.

```json
{
  "focus_atom_maps": [7],
  "edit_hypothesis": "Replace the methyl group at atom 7 with a hydroxymethyl group to test a reduction in lipophilicity.",
  "product_smiles": "완전한 유효 SMILES"
}
```

ADMET 수치는 LLM이 만드는 출력에 포함하지 않는다. Oracle 입력에는 LLM이 출력한 원래
SMILES token을 그대로 사용한다. Legacy Chemprop의 stereo feature 때문에 같은 graph라도
SMILES 표기 순서가 달라지면 예측이 달라질 수 있어 oracle 전에 canonicalize하지 않는다.
이것은 실제 분자 성능 변화의 증거가 아니므로 raw SMILES와 canonical graph를 모두 저장한다.
공식 baseline과 original 재예측의 차이도 별도 audit에 남긴다.

## 비교 방법

`independent`는 매번 original에서 시작하며 평가 피드백을 받지 않는다.
`feedback`은 직전 후보의 실제 검사 결과를 받고 안전하게 parse된 분자를 수정한다.
두 방식 모두 과제당 3개의 제안을 사용한다. 첫 제안은 같은 입력·응답을 공유하며,
이후 방식마다 2번씩 요청하므로 validation 34개에서 실제 HTTP 요청은 170회다.
Worker는 매 요청 같은 seed를 쓰므로 independent 제안은 `proposal_number`를 바꿔 입력을 변화시킨다.
그럼에도 같은 분자를 반복 생성할 수 있으며 숨겨진 재시도나 추가 sampling은 하지 않는다.

후보 선택은 순서가 고정된 조건 비교다: 전체 통과 → RDKit/ADMET 통과 → 로컬 통과 →
유효 graph → original과 다름 → 실패한 ADMET 조건 수가 적음 → 먼저 생성됨.
임의의 가중 utility나 witness 유사도로 고르지 않는다. 선택된 후보 한 개씩 Boltz를 평가한다.
피드백 방식은 feedback 제공과 current 분자 변경을 함께 적용하므로 두 효과를 분리한 실험은 아니다.

## 실행

기존 서버가 실행 중이어야 한다. 서버 상태는 다음 명령으로 확인한다.

```bash
cd /home/uslee/code/DIRMO-3D
python scripts/check_llm.py --health-only
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv-admet/bin/python scripts/run_smdd_feedback.py \
  --output runs/smdd_feedback_02 --rounds 3
```

ADMET 실행 환경은 [requirements-admet.txt](../requirements-admet.txt)를 사용한다.
빈 `CUDA_VISIBLE_DEVICES=`는 CPU oracle이 GPU를 사용하지 않게 한다. LLM은 이미 별도 프로세스에서
GPU 4로 실행되므로 이 설정이 서버 GPU를 변경하지 않는다.
작은 디버깅 실행에는 `--limit 2`를 붙일 수 있다. 기존 폴더는 덮어쓰지 않으며 `--resume`은
dataset·실행 스크립트·prompt 모듈·server config·model·oracle identity가 같을 때 완료된 요청/계산을 재사용한다.
결과를 알 수 없는 HTTP 요청이 남으면 중복 호출하지 않고 중단한다.

Boltz 검증은 [requirements-boltz.txt](../requirements-boltz.txt)의 별도 환경에서 실행한다.

```bash
CUDA_VISIBLE_DEVICES=4 .venv-boltz/bin/python \
  scripts/evaluate_smdd_feedback_boltz.py --run runs/smdd_feedback_02 \
  --memory-fraction 0.27 --max-parallel-samples 1
```

SMDD와 같은 seed 42, sampling steps 200, 구조·affinity 각 10 draws를 사용한다.
`max-parallel-samples`는 동시에 처리하는 draw 수만 제한한다. GPU 여유가 부족하면 미측정으로
남기며 서버나 다른 작업을 종료하지 않는다. OOM이나 미측정은 통과로 계산하지 않는다.
Binding probability > 0.7과 해당 task의 affinity 목표·유지 조건까지 통과해야 전체 성공이다.
자원이 확보되면 동일 명령을 다시 실행해 완료된 계산을 재사용할 수 있다.
이전 BRAF 실행에서 약 13 GiB 상한으로 OOM이 발생했고 약 29 GiB 상한에서 완료된 기록을
반영해, 총 protein residue가 400 이상인 입력에는 기본적으로 30,000 MiB 여유와 이에 맞는
프로세스 상한을 요구한다. 정확한 필요 메모리의 보장은 아니며 보수적인 실행 조건이다.
긴 protein 후보를 실행할 자원이 확보되면 동일 run에서 `--memory-fraction 0.60`을 사용할 수 있다.
가중치·생성 설정이 같으면 완료된 결과를 재사용한다. 다른 작업을 종료하거나 이 한계를 자동으로 완화하지 않는다.

## 코드와 로그

- [smdd_feedback.py](../src/dirmo/evaluation/smdd_feedback.py): system prompt, 입력, 원자/작용기 분석,
  계획 형식 검증, feedback, 다음 분자 선택, 후보 순위.
- [run_smdd_feedback.py](../scripts/run_smdd_feedback.py): frozen HTTP 호출, CPU ADMET, 비교 루프, 재개, 집계.
- [evaluate_smdd_feedback_boltz.py](../scripts/evaluate_smdd_feedback_boltz.py): 선택된 생존 후보의 결합 검증.
- [test_smdd_feedback.py](../tests/test_smdd_feedback.py): 정보 누출, 기준값 유지, 실제 실패 feedback,
  원래 oracle SMILES 보존, 원자 번호와 다음 분자 안전성 검사.

```text
runs/smdd_feedback_01/
├── protocol.json             # dataset/code/config hash와 사전 고정 예산
├── server_identity.json      # 실제 모델/GPU/worker identity; 인증 token 제외
├── admet_identity.json       # oracle package와 weight hash
├── reference_audit.json      # original 재예측과 공식 baseline 차이
├── admet_cache.json          # 원래 SMILES별 실제 ADMET 예측
├── llm_calls.jsonl           # 실제 입력, raw 출력, token/시간/status
├── attempts.jsonl            # 방식/과제/라운드별 RDKit·ADMET 검사 결과
├── report.json               # RDKit·ADMET 결과; 미측정 Boltz는 null
├── pending_boltz.json        # 선택된 생존 후보, 동일 요청 중복 제거
├── report_with_boltz.json    # Boltz 측정까지 반영한 결과
├── code/                    # 이번 실행에서 사용한 코드 스냅샷
├── code_identity.json        # 스냅샷 파일별 SHA256
└── boltz/                    # 입력 YAML, 실행 로그, 3D 구조, affinity 결과
```

실행 중 상태는 `reports/smdd-feedback/run.log`의 `event: attempt` 행에서 확인한다.
논문식 긴 CoT 문자열을 학습하거나 그것의 타당성으로 통과를 판단하지 않는다.
LLM의 짧은 설명과 실제 graph/물성 검사를 분리해 저장하여 나중에 데이터 구축에 사용할 수 있다.
