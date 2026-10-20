# LiFT의 soft chemical conditioning: 논문·공식 코드 검토

검토일: 2026-10-16. 생성·학습을 실행하지 않은 코드 분석이다.

- 논문: https://arxiv.org/html/2608.31009v1
- 공식 코드: https://github.com/kasurl/LiFT
- 로컬 코드: `/home/uslee/code/LiFT`
- 검토 revision: `7a284d4a542f5437c30b95d0d3402b4b0cd9635a`
- 로컬의 별도 `src/sbmo/` 실험 확장은 근거에서 제외했다. 아래에 인용한 공식 tracked 파일의 수정 내역은 없었다.

## 핵심 결론

LiFT는 SMILES에서 추출한 연속 latent를 조건으로 받는 pocket-conditioned FM이다. 목표 수치를 FM 입력 벡터의 QED/SA 필드로 직접 넣거나, 지정한 분자 graph를 hard clamp하는 방식과 구분한다. 목표는 먼저 LLM prompt에 들어가고, LLM이 제안한 SMILES의 embedding이 FM을 조건화한다. 최종 FM은 좌표뿐 아니라 원자 종류와 결합 종류도 생성하므로 QED·SA 등 graph 기반 descriptor의 분포를 바꿀 수 있다.

논문의 property steering 주장은 개별 분자의 정확한 수치 달성 보장보다 생성 집단의 방향성에 관한 주장이다. 원본 ligand보다 모든 결과가 좋아진다는 뜻도 아니다.

## 실제 데이터 흐름

```text
protein pocket의 텍스트 요약 + 목표 지시 (+ 선택적 reference SMILES)
    → LLM이 SMILES 후보 생성
    → 구문·화학 유효성 검사 및 parser 오류 기반 repair
    → 첫 번째 SMI-TED 인코딩 가능 후보 선택
    → frozen SMI-TED encoder
    → c ∈ R^768
    → SMILES projector/AdaLN + SCDR가 포함된 pocket-conditioned GVP FM
    → 원자 좌표 + 원자 종류 + 결합 종류
```

SMILES embedding은 생성 전에 계산하며 한 번의 flow sampling 동안 동일하게 유지한다. 시간과 중간 ligand 상태에 따라 달라지는 것은 router의 처리·gate 값이다. ODE step마다 LLM을 다시 호출하지 않는다.

### 1. 목표는 먼저 LLM prompt에 들어간다

`scripts/conditions/generate_qed_noref.py:183`의 조건 생성 함수는 pocket의 8 Å interaction-shell 설명을 읽고 SMILES 후보 3개를 요청한다. 해당 script의 197행에는 QED > 0.6, SA < 3.0 지시가 있다. 자연어 reasoning은 SMILES를 제안하기 위한 과정이며 FM에 그대로 전달되는 입력이 아니다.

Balanced, distribution, QED/safety, Vina prompt를 바꿔 다른 SMILES 조건을 만들지만, FM은 이러한 task 이름이나 수치 필드를 직접 읽지 않는다. LLM prompt의 목표치가 실제 후보의 지표를 보장하지도 않는다.

### 2. Frozen SMI-TED가 768차원 조건을 만든다

`scripts/embeddings/encode_condition_smiles.py:218`은 `model.eval()`과 `torch.no_grad()`로 `model.encode([smiles], return_torch=True)`를 호출한다. 249행 이후는 후보 순서를 유지하면서 인코딩 가능한 첫 SMILES를 선택한다. 높은 QED/Vina 순서로 선택하지 않는다. Embedding은 ligand/pocket 식별자에 연결된 `.pt` dictionary로 저장한다.

운영 예외: 이 script의 `FILL_FAILED_WITH_ZERO=True`는 조건 생성에 실패한 항목에 zero vector를 넣을 수 있다. 실제 실행 검토에서는 `embedding_health_report.json`과 실패 목록을 확인해야 한다. Zero condition을 정상적인 목표 조건으로 취급해서는 안 된다.

### 3. AdaLN이 ligand scalar feature를 조절한다

`src/model/global_adaln_fusion_controller.py:35`:

```text
projector(c) = Linear(SiLU(c)) → gamma, beta
h_base = LayerNorm(h) ⊙ (1 + tanh(gamma)) + beta
```

이 projector는 ligand-level embedding을 feature별 scale·shift로 바꾼다. 각 원자에는 소속 분자의 gamma/beta를 broadcast한다. h는 QED·SA 수치가 아니라 모델이 학습한 scalar hidden feature다. 임의 방향의 3D 벡터를 SMILES latent로 직접 더하지 않는다.

Projector의 마지막 linear weight와 bias를 0으로 초기화해 이 경로가 처음에는 원래 scalar LayerNorm과 같게 시작한다. 이후 FM 학습으로 조건의 해석을 배운다. SCDR의 head는 별도로 작은 nonzero weight에서 시작하므로 전체 conditioning module이 모든 설정에서 완벽한 zero module이라고 표현하면 부정확하다.

### 4. SCDR가 중간 상태를 보고 gate를 만든다

같은 파일의 128행 이후:

1. 현재 ligand의 scalar feature와 vector norm을 모은다.
2. Node MLP 뒤 mean/max/std/log(atom count)로 graph 수준 요약을 만든다.
3. SMILES embedding과 flow-time embedding을 결합한다.
4. 시간 gate와 구조-조건 gate로 두 요약을 결합한다.
5. Scalar state, scalar update, vector update의 세 종류 gate를 만든다.

Gate는 graph별·feature channel별 값이며 원자에 broadcast한다. 특정 원자나 편집 영역의 hard mask를 뜻하지 않는다. Scalar state scale은 약 0.5–1.5, scalar/vector update scale은 약 0.1–1.9 범위로 제한된다.

`src/model/dynamics_hetero.py:433`은 ligand GVP block에서 AdaLN을 적용하고, feed-forward scalar/vector update에 gate를 곱한다. 따라서 의미 조건은 node 표현과 geometric update를 통해 이후 좌표·원자·결합 예측에 영향을 준다.

코드의 `progress`와 flow time t는 구분한다. Inference는 `lightning.py:1044`에서 progress=1.0을 전달하므로 warmup 계수는 sampling 중 고정이다. 시간에 따른 처리는 별도의 time embedding과 time gate가 담당한다.

이 router는 DIRMO의 chemical edit/pose resample action 선택 MLP와 역할이 다르다. 각 GVP block 안의 연속적인 feature/update modulation이다.

### 5. FM이 graph와 좌표를 함께 갱신한다

`src/model/dynamics_hetero.py:916` 이후 출력은 좌표 velocity, atom logits, bond logits이다. `src/model/lightning.py:1035`의 한 step은 같은 SMILES embedding과 현재 pocket·ligand·시간을 dynamics에 넣은 뒤 다음을 각각 갱신한다.

- x: continuous coordinate flow
- h: atom-type Markov bridge
- e: bond-type Markov bridge

`simulate():1078`은 이를 반복한다. 개념적으로 모델은 F_theta(X_t,A_t,B_t,pocket,t,c)를 계산한다. 출력 graph를 조건 SMILES와 동일하게 강제하지 않으며, 그 chemical prior를 반영하는 다른 분자가 나올 수 있다.

Default 학습/생성 설정은 `configs/training/lift.yml`, `configs/sampling/lift.yml`이다. 공개 sampling 설정은 조건 embedding 경로를 주고 Euler 500-step을 사용한다. Pocket은 별도 geometric condition으로 계속 들어간다.

## 조건 해석을 어떻게 학습했는가

`scripts/embeddings/encode_dataset_smiles.py:57`은 학습 복합체 ligand SDF에서 SMILES를 추출하고 frozen SMI-TED로 embedding을 만든다. 같은 ligand의 실제 좌표·원자·결합을 FM의 target으로 사용한다.

`src/model/lightning.py:622` 이후는 해당 target과 noise 사이의 중간 상태를 만들고 embedding을 넣어 예측한다. 714–724행의 주 손실은 다음 세 항의 가중합이다.

```text
L = lambda_x * L_coordinate + lambda_h * L_atom + lambda_e * L_bond
```

공개 config의 가중치는 1, 50, 50이다. 이 기본 경로에 QED target regression loss, SA reward, Vina gradient는 없다. 데이터의 분자 의미 embedding과 실제 분자 구조 사이 관계를 조건부 FM으로 학습하는 방식이다.

학습 코드의 기본 optimizer는 `AdamW(self.parameters())`이며, 공개 config는 별도 staged freezing을 설정하지 않는다. 따라서 공개 기본 경로는 SMI-TED를 frozen으로 사용하면서 FM과 conditioning network를 학습하는 설정이다. 선택적인 staged training 함수가 존재한다는 사실과 기본 학습 설정은 구분한다.

논문의 '추가 generator fine-tuning 없음'은 이렇게 학습된 LiFT에서 QED/Vina 등 목표별 inference prompt를 바꿀 때 목표마다 FM을 다시 학습하지 않는다는 뜻이다. 조건을 한 번도 학습하지 않은 FM에 embedding만 추가해도 작동한다는 뜻이 아니다.

## FLOWR.root와 현재 DIRMO에 적용할 때

현재 DIRMO PB 복구는 전체 graph를 고정하고 일부 좌표만 수정한다. 이 고정 계약을 유지하면 soft chemical prior를 넣더라도 graph 기반 QED·SA를 바꿀 수 없다.

고정 core의 graph/좌표는 hard condition으로 유지하고, 편집 영역의 graph/좌표는 생성하도록 열면서, LiFT처럼 별도의 chemical latent를 soft condition으로 주는 조합은 설계 가능하다. 다만 FLOWR.root와 LiFT의 DrugFlow/GVP backbone이 다르므로 projector 및 조건 주입 경로의 학습·정렬이 필요하다. LiFT checkpoint나 projector weight를 바로 공유할 수 있다고 간주하면 안 된다.

LLM은 학습하지 않고 바꿔 사용할 수 있다. LLM별 SMILES는 동일한 frozen chemical encoder를 거치므로 같은 FM interface를 공유한다. 조건 분포 변화에 따른 실제 생성 품질은 별도로 평가해야 한다. 필요한 학습의 우선 대상은 soft condition과 FM의 연결 경로이며, backbone을 얼마나 동결할지는 별도 실험 선택이다.

수치 조건의 정확한 달성이 중요하면 property-vector conditioning, predictor guidance, 후보 평가·선택 등을 별도로 설계해야 한다. 검토한 LiFT 경로는 global SMILES prior에 기반한 trend-level control이다.

## 빠르게 읽을 코드

| 역할 | 파일·시작 행 |
| --- | --- |
| QED/SA 목표 prompt | `/home/uslee/code/LiFT/scripts/conditions/generate_qed_noref.py:183` |
| Frozen encoder와 조건 후보 선택 | `/home/uslee/code/LiFT/scripts/embeddings/encode_condition_smiles.py:218` |
| 학습 target의 SMILES embedding | `/home/uslee/code/LiFT/scripts/embeddings/encode_dataset_smiles.py:57` |
| AdaLN와 SCDR | `/home/uslee/code/LiFT/src/model/global_adaln_fusion_controller.py:13` |
| GVP block에 gate 적용 | `/home/uslee/code/LiFT/src/model/dynamics_hetero.py:433` |
| 학습 손실 | `/home/uslee/code/LiFT/src/model/lightning.py:622` |
| Sampling step와 세 modality 갱신 | `/home/uslee/code/LiFT/src/model/lightning.py:1035` |
