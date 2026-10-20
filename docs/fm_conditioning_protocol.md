# FLOWR.root 학습·조건·생성 품질 검증 계획

작성일: 2026-10-13. 대상: DIRMO-3D에서 사용하는 FLOWR.root.

**실행 상태:** 원본 checkpoint 두 개를 CPU에서 읽어 hyperparameters와 SHA256을 확인했다. 이 문서의 새 생성·학습 실험은 아직 실행하지 않았다. 아래 표본 수는 제안하는 실험 규모이며 완료 결과가 아니다. 기존 GPU 작업과 서버 설정은 변경하지 않았다.

**후속 실행 기록:** 2026-10-13 사용자 요청으로 GPU 3 생성 실험을 시작했다. 아래의 초안 상태와 구분한다. 전용 runner 및 분석 코드는 추가됐고, 실행 상태와 실제 결과는 [실험 폴더](../runs/fm_quality_gpu3_20261013_v1/REPORT.md) 및 [비교 분석](../runs/fm_quality_gpu3_20261013_v1/ANALYSIS.md)에 기록한다. 이 실행에는 FM 추가 학습이 포함되지 않는다.

## 1. 검증할 질문

1. 학습 때 어떤 입력 조건을 사용하도록 설정했는가?
2. DIRMO가 그 조건을 모델에 정확히 전달하며 출력이 조건을 지키는가?
3. 입력 복사나 일반적인 pose 보정보다 FM 생성이 품질을 높이는가?
4. 이를 통과한 생성기가 분자 최적화에도 도움이 되는가?

조건 보존과 생성 품질은 서로 다른 평가다. 출력에 고정 원자·결합·좌표를 강제로 덮어쓰면 보존율은 높아질 수 있다. 이것만으로 모델이 해당 조건을 학습했거나, 자유 영역을 잘 생성했다고 판단하지 않는다. 먼저 LLM과 Action Router를 거치지 않는 FM 단독 실험을 한다.

## 2. 로컬 checkpoint에서 확인한 학습 설정

원본 증거: [checkpoints.json](../reports/fm-conditioning-audit/checkpoints.json). 추론 로더가 덮어쓴 hparams가 아니라 checkpoint 파일의 `hyper_parameters`를 읽었다.

| 항목 | `flowr_root_base_spindr.ckpt` | `flowr_root_v2.ckpt` |
| --- | --- | --- |
| 저장된 dataset | `spindr` | `combined_prot-lig` |
| 저장 시점 epoch / global_step | 237 / 35,224 | 42 / 495,214 |
| 좌표 loss weight | 3 | 2 |
| 원자 / 결합 / 전하 loss weight | 1 / 3 / 2 | 1 / 3 / 2 |
| ligand–pocket 거리 loss weight | 5 | 2 |
| 결합각 / 결합길이 loss weight | 10 / 5 | 해당 키 미저장 |
| affinity prediction | `False` | `True`, affinity loss weight 1 |
| fragment inpainting | `True` | `True` |
| fragment growing | `True` | 해당 키 미저장 |
| interaction conditional | `True` | 해당 키 미저장 |
| linker inpainting | `False` | `True` |
| legacy scaffold / functional-group inpainting | 둘 다 `True` | 둘 다 `True` |
| whole-graph `graph_inpainting` | 해당 키 미저장 | 해당 키 미저장 |

두 checkpoint 모두 rigid pocket, `coord_scale=1.0`, self-conditioning, unconditional/inpainting 혼합 설정이 저장돼 있다. 좌표는 Gaussian prior에서 linear interpolation, 원자·결합은 `unmask` interpolation을 사용한다. 학습 interpolation의 좌표 노이즈 설정은 0.3, schedule은 `constant_decay`, 시간 분포 parameter는 alpha=1.8, beta=1.0이다. 이 학습 노이즈 값과 DIRMO의 reference-prior 노이즈는 쓰이는 위치가 다르며 같은 실험 조건으로 간주하지 않는다.

저장된 epoch는 snapshot 번호다. 서로 다른 데이터·batch·학습 이력을 가진 두 파일을 epoch 숫자만으로 비교하지 않는다. `epochs=500`도 설정값이지 500 epoch 완료 증거가 아니다. 저장된 조건 flag는 학습 구성의 증거이며 모드별 학습량과 실제 품질까지 알려주지는 않는다. 파일명 `v2`만으로 최신 공개 모델 또는 생성 품질 우위를 보장하지 않는다.

### whole-graph 학습 여부에 대한 정정

현재 DIRMO adapter는 `graph_inpainting` 키 존재 여부로 `graph_condition_support`에 `trained` 또는 `untrained`를 기록한다. 그러나 확인한 upstream `fm_pocket.py`의 선택적 hparams 저장 목록에는 이 키가 빠져 있다. **키 부재만으로 whole-graph 조건 미학습을 단정할 수 없다.** 이 감사에서의 상태는 **학습 여부 미확인**이다. 원래 학습 실행 config·코드 버전·로그를 확보하거나 별도 생성 실험으로 적합성을 확인해야 한다. 기존 로그의 문자열을 확정된 학습 이력으로 사용하지 않는다. 이번 작업에서는 실행 코드와 기존 로그를 수정하지 않았다.

### 학습 방법과 코드 위치

확인한 upstream revision: `739a33c2aab74d074dd24605b9a2e34dc809ae9c`, 경로 `/home/uslee/code/flowr_root`.

- `flowr/data/interpolate.py`: prior, 실제 분자, 시간 t를 사용해 중간 상태를 만들고 조건 mask를 설정한다. 좌표의 기본 선형 경로는 `x_t=(1-t)x_0+t*x_1+noise(t)`이다.
- `flowr/models/fm_pocket.py:training_step`: pocket, 중간 ligand, 시간, fragment mask/mode와 설정된 상호작용 정보를 받아 예측한다. self-conditioning도 여기서 연결한다.
- `flowr/models/losses.py:compute_losses`: 좌표와 원자·결합·전하 등의 손실, 활성화한 geometry/affinity 손실을 계산한다. 확인한 코드의 기본 좌표 학습은 목표 좌표에 대한 data MSE다. velocity loss는 별도 옵션이며 두 checkpoint에 해당 옵션은 저장돼 있지 않다.
- `scripts/train.sh`: upstream 학습 실행 예시다. 현재 예시의 모든 설정이 이 checkpoint를 만든 원래 recipe와 일치한다고 가정하지 않는다.

설명을 위한 손실 형태는 다음과 같다. 실제 활성 항목과 가중치는 checkpoint 및 실행 config를 확인한다.

```text
L = λcoord · MSE(예측 목표 좌표, 실제 목표 좌표)
  + λatom · CE(원자 종류) + λbond · CE(결합 종류)
  + λcharge · CE(전하) + 활성화한 geometry / affinity 손실
```

학습에서는 여러 t의 중간 상태를 통해 목표 분자를 복원하도록 배운다. 추론에서는 이 예측으로 flow를 여러 번 적분한다. **DIRMO의 64/128-step은 추론 적분 횟수다. 학습 epoch와 다르다.** 현재 원본 학습 설정을 DIRMO의 Vina/QED/SA utility를 보상으로 학습한 것으로 해석하지 않는다.

FLOWR.root의 전반적인 방법은 [원 논문](https://www.nature.com/articles/s41467-026-74130-9)을 참고한다. 논문 전체의 학습 단계가 로컬의 각 파일에 모두 적용됐다는 뜻은 아니다.

## 3. 조건 종류와 현재 DIRMO 범위

| 조건 | 현재 검증할 수 있는 범위 | 별도 확인·구현이 필요한 부분 |
| --- | --- | --- |
| Pocket 원자 종류·3D 좌표 | 고정한 pocket을 받아 생성 | 좌표계, protonation, pocket crop의 일관성 |
| Ligand 전체 graph | 원자·전하·결합을 고정하고 좌표 생성 | 해당 checkpoint의 이 사용 방식에서의 품질 |
| Fragment/core graph·좌표 | 지정한 held 영역 고정, free 영역 생성 | mask의 실제 전달, 자유 영역 geometry와 graph 품질 |
| Reference conformer | 초기 좌표 또는 그 주변 노이즈를 prior로 사용 | 정확한 결합 pose를 제공한 효과와 FM의 효과 분리 |
| 특정 protein–ligand interaction | base metadata에 조건 설정 존재 | 현재 adapter의 interaction 입력 경로와 검증 연결 필요 |
| 자연어 요청, QED/LogP 목표 값 | 현재는 외부 평가·선택에 사용 | 목표 값을 모델 조건으로 넣는 입력·학습 또는 guidance 구현 |
| 자유로운 원자 수 증가 / linker 생성 | upstream 및 일부 checkpoint에 관련 모드 존재 | 현재 DIRMO는 `sample_mol_sizes=False`; 자동 지원으로 간주하지 않음 |

`graph` 모드에서는 좌표까지 모두 고정하지 않는다. 그렇게 하면 pose 생성 실험이 입력 복사가 된다. `fragment` 모드에서는 held 원자의 좌표와 discrete feature를 고정하되 free 원자의 graph는 바뀔 수 있다. 두 모드에 동일한 whole-graph 보존 기준을 적용하지 않는다.

**우선 검증할 조건은 pocket + exact graph, pocket + held fragment다.** 상호작용 조건은 별도 입력 경로를 확인한 뒤 추가한다. 조건 의미와 upstream 예시는 [공식 저장소](https://github.com/jule-c/flowr_root)를 참고하되 현재 DIRMO의 지원 범위와 구분한다.

## 4. 서로 분리해서 평가할 과제

### A. 알려진 pose 주변 보정

입력에 기존 bound pose가 들어간다. 기존 구조를 유지하거나 작은 오차를 보정하는 과제다. `reference` noise=0도 이 과제의 조건이 될 수 있다. FM을 거치지 않은 입력 복사와 비교한다. 높은 통과율만으로 새로운 binding pose 예측 능력을 주장하지 않는다.

### B. 같은 graph의 pose 복구·예측 — 1차 주 실험

실험적으로 관측한 원래 결합 pose는 evaluator만 읽는다. FM에는 ligand-only conformer 또는 사전에 정한 잘못된 pose를 입력한다. 초기 conformer는 SMILES에서 ETKDG로 생성하고 pocket box 중심으로 이동하는 방식 등을 사용한다. 회전·torsion·이동의 난이도는 case manifest에 고정한다.

각 case의 잘못된 입력 SDF는 모든 조건과 checkpoint가 공유한다. 정답 full coordinates가 graph conditioning이나 self-conditioning으로 우회 전달되지 않는지 확인한다. 정답 ligand 중심을 초기화에 사용했다면 알려진 site 위치 정보를 사용하는 조건으로 명시한다. 완전한 blind docking이라고 부르지 않는다.

### C. 화학 편집 후 exact graph의 pose 생성

편집 graph를 먼저 고정하고 FM은 그 graph의 좌표를 생성한다. 원자 수·원소·결합이 달라질 수 있으므로 원본 ligand 전체와의 RMSD를 적용하지 않는다. 보호 core의 대응 좌표, 새 분자의 geometry·clash·interaction 및 docking score를 측정한다. 해당 편집 ligand의 실험 결합 pose가 있을 때만 정답 pose 정확도를 추가로 평가한다.

### D. Native fragment 재설계

held core의 graph·좌표를 제공하고 free 영역의 종류·결합·좌표를 생성한다. protected core 유지, 새 graph 생성률, free 영역 품질과 다양성을 평가한다. 정답 free 영역을 입력하지 않는다. 전체 graph 고정 pose 실험과 평균 통과율을 합치거나 단순 우열 비교하지 않는다.

## 5. 실행 순서와 제안 예산

### 단계 0 — provenance 고정: metadata 감사 완료

checkpoint SHA256, source revision, adapter 구현 hash, 환경 버전, solver, step, pocket preparation과 prior 설정을 kernel별로 저장한다. 두 checkpoint의 architecture와 load compatibility도 실제 생성 전에 확인한다. 각자의 affinity head 존재 여부 등에 따른 차이를 임의로 덮어쓰지 않는다.

### 단계 1 — 조건 전달 점검: 6 complexes × 5 seeds

방향족, 전하, 입체중심이 있는 입력을 포함하고 현재 atom/pocket cap 안에서 선정한다. exact-graph와 native-fragment를 각각 점검한다. 서로 다른 checkpoint에서 사용할 때도 동일한 의미의 mask를 명시한다.

1. typed/mapped SMILES와 SDF가 같은 원소·결합·전하·stereo를 나타내는지 확인한다.
2. upstream tensorization 후 graph가 보존되는지 확인한다.
3. 요청한 free atom map이 실제 model mask에 대응하는지 확인한다. fixed count는 `전체 원자 수 - free 원자 수`와 같아야 한다.
4. graph 모드는 discrete feature만 고정되는지, fragment 모드는 held 좌표만 고정되는지 확인한다.
5. 생성 출력의 graph 및 held 영역 보존을 원본 request와 비교한다.

반환된 유효 출력에 mask·계약 위반이 있으면 해당 입력 경로를 수정하고 다음 단계로 간다. molecule build 실패는 조건 전달 실패와 분리해 품질 실패로 기록한다. protocol에 맞지 않는 입력은 적격성 검사 단계에서 기록하며, 실패를 숨기려고 다른 ligand로 몰래 교체하지 않는다.

확인할 artifact: `request.json`, `kernel.json`, `pocket_frame.json`, `fragment_mask.json`, `graph_check.json`, fragment 평가의 `redesign.json`. hard held 좌표 drift 허용치는 예를 들어 0.001 Å로 사전에 고정한다. 이 값은 본 실험의 계약 검사 제안값이며, 일반적인 pose 정확도 기준이 아니다.

### 단계 2 — pose 생성 pilot: 20 pockets × 10 seeds

과제 B에서 같은 graph·잘못된 초기 입력·pocket을 공유하고 다음 세 조건을 비교한다.

| 조건 | 초기 좌표 | 고정할 정보 |
| --- | --- | --- |
| P1 | Gaussian random prior | 전체 discrete graph + pocket |
| P2 | 잘못된 입력 conformer를 reference prior로 사용, 추가 노이즈 0 | 전체 discrete graph + pocket |
| P3 | 같은 입력 conformer + reference prior 노이즈 std 0.3 Å | 전체 discrete graph + pocket |

원본 결합 pose를 제공하는 noise=0 조건은 과제 A의 별도 진단 대조군으로 둔다. P2가 정답 pose를 입력받는 조건이 되지 않도록 한다.

두 checkpoint가 모두 호환되면 `20 × 10 × 3 × 2 = 1,200`회의 FM draw를 계획한다. 우선 64-step으로 고정한다. 먼저 단계 1에서 실측한 메모리·시간으로 전체 예산을 추정한다. 기존 FM 시간은 이 계획의 실행 시간 보장이 아니다. 128-step은 유망한 조건을 고른 후 동일 case/seed의 후속 비교로 둔다.

대조군을 함께 저장한다.

- **입력 그대로:** 잘못된 pose를 복사해 평가한다.
- **같은 prior 그대로:** 각 seed에서 FM에 실제로 전달한 초기 좌표를 FM 없이 평가한다. 특히 P3의 noise 효과와 FM 효과를 분리한다.
- **일반 docking 또는 pose refinement:** 같은 잘못된 입력·pocket·후보 수를 사용하고 별도로 CPU/GPU 시간과 평가 비용을 기록한다. 후보 수 일치가 계산 비용 일치를 뜻하지는 않는다.

### 단계 3 — fragment pilot: 별도 평가

20 pockets에서 고정한 held core를 기준으로 작은 free 영역(예: 1–3 atoms)과 더 큰 free 영역(예: 전체 heavy atoms의 약 20%)을 비교한다. 작은 분자에서 두 mask가 같아지면 그 사실을 기록한다. 중요한 interaction에 관여하는 원자를 보호할지 여부는 evaluator 결과를 보기 전에 고정한다.

각 mask별 10 seeds, 두 checkpoint이면 최대 `20 × 10 × 2 masks × 2 = 800`회의 추가 draw다. 초기 prior 설정을 fragment 실험 내에서 동일하게 고정한다. 현재 고정된 원자 수 안의 재설계 과제이며 원자 수를 늘리는 fragment growing 실험이 아니다. 원본 유지 및 같은 위치에 미리 정의한 graph 편집을 한 대조군도 별도로 평가할 수 있다.

### 단계 4 — 최종 held-out 평가

pilot에서 조건과 평가 규칙을 확정하고 별도의 50–100 pockets로 확인한다. 조건 선택에 쓴 pilot 결과를 최종 test에 다시 포함하지 않는다. 단순 PDB ID가 아니라 protein sequence/target 및 ligand scaffold가 유사한 사례의 누출을 관리한다. 원래 FM 학습 split 정보가 있으면 그와의 overlap도 확인한다. 정보가 없으면 pretraining overlap 미확인으로 명시하며 처음 보는 target에서의 일반화라고 단정하지 않는다.

## 6. 데이터 준비

기존 CrossDocked 입력은 입력 계약·작동 여부와 geometry 평가에 재사용할 수 있다. 그러나 `docked-0` 또는 `tt-min` SDF를 곧바로 실험적으로 관측한 정답 pose로 간주하지 않는다. 이런 SDF와의 RMSD는 우선 입력 일관성 또는 재현 정도를 뜻한다.

pose 정확도 실험에는 출처를 확인할 수 있는 원본 protein–ligand 실험 복합체를 사용한다. 사용 가능한 원본 구조에서 아래를 분리해 저장한다.

```text
case_id/
  pocket.pdb          # 모든 방법이 공유하는 고정 pocket
  ligand_input.sdf    # 잘못된 pose 또는 ligand-only 초기 conformer
  ligand_truth.sdf    # evaluator 전용 실험 pose
  conditions.json    # atom maps, held/free, 보호 조건
```

protein frame, protonation, hydrogen 처리, aromaticity·stereo 표현, water/cofactor 포함 정책과 docking box를 고정한다. pocket이 다른 구조에서 왔다면 protein 정렬을 통해 같은 frame으로 변환하고 그 변환을 저장한다. ligand 자체를 정답에 맞춰 정렬해서 docking RMSD를 계산하지 않는다.

## 7. 품질 지표와 성공 기준

utility 한 숫자로 결론내리지 않고 다음을 따로 보고한다.

| 평가 질문 | 지표 | 분모·주의점 |
| --- | --- | --- |
| 생성됐는가? | molecule build / sanitization / connected 비율 | 실패를 포함한 scheduled draw 및 실제 FM draw 수를 각각 보고 |
| 조건을 지켰는가? | graph·stereo match, held subgraph match, held 최대 drift | exact graph와 fragment 계약을 구분 |
| 물리적으로 타당한가? | PoseBusters geometry·clash 검사, 결합 길이·각·ring geometry | raw 출력에 적용; missing을 통과로 처리하지 않음 |
| pose가 맞는가? | symmetry-aware heavy-atom RMSD, RMSD ≤2 Å와 물리 검사 동시 통과 | 동일 graph와 실험 정답 pose가 있는 경우 |
| fragment를 새로 만들었는가? | 원본과 다른 유효 graph 비율, free 영역 geometry, 고유 graph 수 | unchanged 출력도 보존율과 별도로 집계 |
| 목적 성질이 좋아졌는가? | Vina, QED, SA, LogP, 보호 interaction, 목적별 개선량·Pareto 후보 | 목표 방향과 허용 손실을 사전에 고정 |
| 다양한가? | 서로 다른 graph 수 / 같은 graph의 서로 다른 conformer 수 | noise=0에서 seed만 다른 동일 출력을 독립 다양성으로 세지 않음 |
| 비용은 얼마인가? | 실패 포함 draws, NFE, latency, GPU peak memory, 평가 비용 | LLM 호출이 없는 FM 단독 예산 |

동일 graph에서는 graph 기반 QED·SA·LogP가 pose 변화로 개선되지 않는다. 값이 달라지면 분자 표현이나 graph 변화를 먼저 확인한다. Vina score 개선을 실험 binding affinity 향상으로 해석하지 않는다.

Pose 평가의 제안 primary endpoint는 `graph/계약 유효 + 사전에 정한 물리 검사 통과 + RMSD ≤2 Å`다. 낮은 RMSD와 물리적 타당성을 함께 보는 이유는 [PoseBusters 논문](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc04185a)에 설명돼 있다. PB-valid의 구성 검사와 적용한 PoseBusters 버전·config를 저장하며 DIRMO의 기존 `feasible`과 자동으로 동의어로 사용하지 않는다.

RMSD는 pocket/protein frame에서 계산한다. [RDKit `CalcRMS`](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html)는 ligand를 새로 정렬하지 않고 대칭성을 고려할 수 있다. `GetBestRMS`처럼 ligand 정렬을 하는 함수로 docking 오류를 지우지 않는다. stereo를 보존하는 원자 대응, heavy-atom 범위와 대칭 처리 설정도 고정한다.

분자가 달라진 fragment 출력에는 정답 원본과의 전체 graph/formula 일치 검사를 강요하지 않는다. 적용 가능한 물리 검사 목록을 별도로 정하고, common core의 원자 대응과 drift를 측정한다. 필요하면 full PoseBusters docking suite와 구분한 geometry-only 결과를 보고한다.

Top-1은 정답 RMSD를 보지 않는 사전 고정 score로 후보를 고른 뒤 평가한다. 10개 중 정답 RMSD가 가장 낮은 결과는 `oracle best-of-10` coverage로 별도 표시한다. 이를 실사용 Top-1 성능으로 보고하지 않는다.

## 8. 후처리·통계·실패 분석

raw 단계에서는 valence repair, relaxation, docking 재최적화, oversampling/filtering을 끈다. discrete/held 조건을 적용하는 sampler 동작은 별도로 기록한다. 후처리의 효과를 알고 싶으면 raw 결과를 보존한 뒤 모든 비교 방법에 같은 처리를 적용해 before/after를 평가한다. 후처리로 좋아진 결과를 순수 FM 결과와 섞지 않는다.

조건 비교는 같은 pocket·case·seed의 짝 비교로 한다. 평균·중앙값·차이와 target/pocket 단위 bootstrap 신뢰구간을 보고한다. 같은 pocket의 10 seeds를 독립적인 10개 target처럼 취급하지 않는다. 20 pockets pilot의 차이만으로 일반적 우위를 확정하지 않는다.

실패 원인은 다음 단계별로 기록한다.

```text
입력 적격성/graph–SDF 불일치
→ tensorization/좌표계/mask 전달 실패
→ FM 실행 또는 raw molecule build 실패
→ graph·보호 조건 위반
→ geometry/clash 실패
→ 평가 도구 오류 또는 정답 정보 부재
→ 유효하지만 목적 개선 없음
```

model load/입력 거부 등 preflight 실패는 FM raw draw가 아니다. 별도 scheduled attempt denominator를 유지한다. GPU에서 실행하고 실패한 draw는 성공 결과로 대체하지 않고 실제 draw budget에 포함한다. metric 측정이 누락된 결과를 성공으로 처리하지 않으며 누락 사유를 보고한다.

기존 분석에는 93 targets/279 paired draws에서 64-step feasible 72.4%, 128-step 67.4%가 기록돼 있다. reference 주변 과제에서 얻은 제한된 결과이며 다른 prior의 품질을 대표하지 않는다. graph/SDF 및 재검증 불일치도 있었다. [기존 실험 분석](../reports/experiment-analysis-2026-10-12/REPORT.md)을 사용해 입력·adapter 오류부터 분리한다.

## 9. 조건 fine-tuning을 진행할 기준과 방법

입력·mask 전달이 정상인데 FM이 no-FM prior 또는 docking 대조군보다 반복적으로 낮은 품질을 보일 때 target task에 맞춘 fine-tuning을 고려한다. 인터페이스 오류를 학습으로 해결하려고 하지 않는다.

**Pose 생성용 학습 예제**는 `(pocket, fixed ligand graph, noisy/prior coordinates, target bound coordinates)`로 만든다. crystal complex 한 개로도 prior/noise를 달리해 여러 예제를 만들 수 있으므로 initial/optimized ligand pair가 필수는 아니다. discrete graph를 조건으로 전달하고 좌표는 생성 대상에 남긴다. 학습 noise·초기 시간·mask 의미가 실제 추론과 일치하도록 구성한다. 전 좌표를 고정해 정답을 복사하는 경로를 만들지 않는다.

**Fragment 생성용 학습 예제**는 `(pocket, held core graph+coordinates, free mask, target full ligand)`로 만든다. held/free 선택을 여러 영역에 대해 반복하되 split은 복합체/target/scaffold 단위로 먼저 나눈다. held 영역을 복사하는 손실만으로 전체 loss가 낮아지지 않도록 free 영역의 성능을 따로 기록한다. discrete fixed-loss masking을 사용한다면 옵션과 실제 적용을 명시한다.

이 학습은 조건에 맞는 유효한 3D 구조 분포를 학습한다. `QED를 높여라` 같은 목적 개선을 자동으로 학습하는 것은 아니다. 목표 값 기반 조건 생성에는 그 목표 입력을 받는 경로와 적합한 학습 예제가 추가로 필요하다.

기존 checkpoint에서 시작하고 architecture·vocab·coordinate scale을 맞춘다. 원래 학습 lr을 그대로 최적 fine-tuning lr이라고 쓰지 않는다. 작은 validation 비교로 lr과 조건 mixture를 선택하고 test는 최종 비교에만 사용한다. checkpoint의 affinity head는 별도의 평가 대상으로 두고 pose 품질 검사의 대체 기준으로 사용하지 않는다.

fine-tuning 전후에 동일 held-out cases, seeds, draw 수와 raw 평가를 사용한다. validation loss와 함께 조건 보존·생성 유효성·pose 정확도 또는 fragment 품질을 평가한다. 낮은 loss만으로 품질 개선을 결론내리지 않는다.

## 10. DIRMO에서 실행하려면 필요한 구현

현재 `collect`는 Planner를 구성하고 LLM 호출을 포함할 수 있다. FM 단독 평가를 보장하는 `--no-llm` 등의 CLI는 현재 없다. 이를 우회하는 [benchmark runner](../scripts/benchmark_fm_conditions.py)와 [평가 구현](../src/dirmo/evaluation/fm_quality.py)을 후속 작업에서 추가했다. 입력/native pose를 분리하고 실제 sampler의 초기 좌표 및 held mask를 기록한다. 각 checkpoint/mode는 별도 프로세스에서 실행하며 최대 두 프로세스가 지정 GPU를 공유한다. CPU complex 전처리만 cache/deep copy하고 sampling은 매 draw마다 다시 seed를 설정한다.

새 실험 실행 예시(기존 디렉토리를 덮어쓰지 않는다):

```bash
conda activate dirmo
cd ~/code/DIRMO-3D
python scripts/benchmark_fm_conditions.py \
  --output runs/fm_quality_new --gpu 3 --cases 20 --seeds 10 \
  --gpu-concurrency 2
```

중단된 실험은 원래 설정으로 `--resume`을 붙인다. 이미 기록된 draw는 재생성하지 않는다. 다른 살아 있는 동일 실험 프로세스와 동시에 resume하지 않는다. 미완료 draw artifact가 있으면 자동 재추출하지 않고 점검을 요구한다. 모델과 GPU UUID 설정은 runner가 처리하므로 명령 앞에 별도의 CUDA export가 필요하지 않다.

최종 비교 보고서 생성:

```bash
python scripts/analyze_fm_quality.py --run runs/fm_quality_new
```

미완료 실행의 중간 보고서는 `--allow-incomplete`로 만들 수 있으며 결과의 상태를 명시한다. PDBbind molfile query 수소는 heavy graph와 좌표를 보존해 정리하고, SDF roundtrip에서 typed stereo graph가 바뀌는 입력은 적격성 검사에서 제외·기록한다. Vina용 receptor는 별도로 Meeko에서 준비하며 표준 단백질 잔기와 명시적 HIE template를 사용한다. 준비가 실패한 case도 FM 실험에는 남기고 Vina/docking을 missing으로 기록한다.

재사용할 코드:

- `src/dirmo/generation/flowr.py`: 실제 checkpoint load, graph/fragment prior, raw 생성과 조건 artifact.
- `src/dirmo/evaluation/metrics.py`: 기존 metric 평가 구성. PoseBusters/RMSD 범위는 추가 연결·검증 필요.
- `src/dirmo/schemas.py`, `src/dirmo/config.py`: generation request와 kernel/config 정의.

runner의 구현 순서는 다음과 같다.

1. case별 input/truth를 분리한 manifest와 고정된 condition grid를 읽는다.
2. LLM·router를 우회해 한 condition당 정확히 한 raw draw를 실행하고 실패도 저장한다.
3. 실제 sampler 초기 좌표를 저장해 no-FM prior 대조군을 만든다.
4. 모든 원본 출력에 조건·geometry·RMSD 및 목적 metric을 적용한다.
5. `draws.jsonl`, `failures.jsonl`, `summary.csv`, case/target별 통계 보고서를 만든다.

각 row에는 최소한 case/target/scaffold ID, checkpoint SHA, implementation hash, condition ID, prior/noise/steps/seed, held/free maps, 실제 raw draw/NFE, failure stage, 개별 metric과 latency가 있어야 한다. evaluator만 truth pose에 접근하도록 생성 입력 경로와 구분한다.

checkpoint audit → 조건 전달 점검 → pose/fragment pilot → 별도 held-out 확인 → 필요 시 task별 fine-tuning 순으로 진행한다.
