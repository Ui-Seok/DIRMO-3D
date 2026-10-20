# FLOWR.root와 LiFT FM 파라미터 수

측정일: 2026-10-16 (Asia/Seoul). CPU에서 모델을 구성하고 체크포인트를 로드하여 측정했다. GPU 생성, 학습, 환경 패키지 변경은 수행하지 않았다.

| 측정 범위 | 파라미터 수 | 백만 단위 |
| --- | ---: | ---: |
| 현재 DIRMO FLOWR.root 전체 | 32,224,412 | 32.224412 M |
| └ Pocket encoder | 4,328,224 | 4.328224 M |
| └ Ligand decoder 및 출력부 | 27,896,188 | 27.896188 M |
| LiFT의 DrugFlow/GVP FM 부분: 조건 모듈 제외 | 12,101,903 | 12.101903 M |
| LiFT의 AdaLN·SCDR 조건 모듈 | 10,443,220 | 10.443220 M |
| LiFT의 조건 모듈 포함 FM 전체 | 22,545,123 | 22.545123 M |

## 대상 체크포인트

- FLOWR.root: `/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_base_spindr.ckpt`
  - `worker-runtime/config.json`에 저장된 FM 체크포인트 경로다.
  - `flowr_root_v2.ckpt`는 위 표의 FLOWR.root 측정 대상이 아니다.
- LiFT: `/LocalData1/uslee/checkpoints/LiFT/lift.ckpt`
  - 체크포인트의 `predictor_params`는 heterogeneous GVP, 5 layers, node/edge hidden dimensions `[128, 32]`, `smiles_dim=768`, `num_rbf_time=16`, `use_smiles_adaln=True`다.
  - LiFT의 LLM, 별도 frozen SMI-TED encoder, size predictor, `src/sbmo/` 확장 모델은 포함하지 않는다.

## 측정 방법과 확인

기본 집계는 다음과 같다. 공유 파라미터는 `named_parameters()`의 기본 동작에 따라 중복 집계하지 않는다. 추론 시 동결 여부와 관계없이 모델의 전체 파라미터를 센다.

```python
sum(parameter.numel() for _, parameter in model.named_parameters())
```

### FLOWR.root

`/home/uslee/miniconda3/envs/dirmo/bin/python`에서 DIRMO와 같은 `flowr.scriptutil.load_model()` 경로를 사용했다. `CUDA_VISIBLE_DEVICES=''`, `gpus=0`, graph inpainting, Euler 64-step 설정으로 CPU에 로드했으며 forward는 호출하지 않았다.

- 파라미터 tensor: 800개, 총 32,224,412 elements.
- 체크포인트에 저장된 buffer: 18개, 총 201 elements. RBF centers와 cutoff 등이며 파라미터 집계에서 제외했다.
- 체크포인트 `state_dict`의 전체 tensor elements는 32,224,613이다. 이 값을 그대로 파라미터 수로 사용하면 buffer 201개가 포함된다.
- `gen.pocket_enc`와 `gen.ligand_dec`의 파라미터 합이 전체 파라미터 수와 일치했다.
- 모든 파라미터가 CPU에 존재함을 확인했다.

### LiFT

`/home/uslee/miniconda3/envs/lift/bin/python`에서 체크포인트 hyperparameters와 `src/model/lightning.py:273`의 `DrugFlow.init_model()`과 동일한 인자를 사용하여 `DynamicsHetero`를 CPU에 구성했다. 체크포인트의 `dynamics.*`를 prefix 제거 후 `load_state_dict(..., strict=True)`로 로드했다. 따라서 위 수치는 별도 DrugFlow 공개 체크포인트의 크기가 아니라, LiFT에 실제 저장된 FM 구성의 크기다.

- `dynamics` 파라미터 tensor: 1,883개. 일부 0-element dummy parameter도 포함되며 numel 합에는 영향을 주지 않는다.
- `dynamics` 파라미터: 22,545,123개. 해당 모듈에는 persistent buffer가 없다.
- 최상위 `prior_h`는 14-element 고정 buffer이며 집계에서 제외했다.
- 각 GVP layer의 `adaln0_ligand` 및 `adaln1_ligand` subtree를 조건 모듈로 분류했다. 한 layer당 2,088,644개, 5개 layer 합계 10,443,220개다.
- 조건 모듈을 제외한 파라미터는 12,101,903개다. 동일한 설정에서 `use_smiles_adaln=False`로 모델을 구성했을 때도 12,101,903개로 일치했다. 이 추가 구성에는 추론이나 학습을 수행하지 않았다.
- 모든 파라미터가 CPU에 존재함을 확인했다.

FLOWR.root는 LiFT의 조건 모듈 제외 FM보다 약 2.66배, 조건 모듈 포함 FM보다 약 1.43배의 파라미터를 가진다. 두 모델의 architecture는 서로 다르다.
