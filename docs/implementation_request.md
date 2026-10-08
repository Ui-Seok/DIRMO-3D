# DIRMO-3D 구현을 위한 Codex 프롬프트

작성일: 2026-10-08. 아래 「Codex에 전달할 프롬프트」의 내용을 그대로 전달한다.

## Backbone 선택 근거

- 첫 구현의 주 생성기는 **FLOWR.root의 공개 사전학습 모델**로 한다. 연구의 초점은 새로운 FM을 처음부터 학습하는 것이 아니라, 측정과 진단에 따라 화학적 편집 또는 동일 graph의 3D 재생성을 선택하는 정책이다.
- FLOWR.root 공개 코드에는 `graph_inpainting`으로 화학 graph를 고정하고 좌표를 생성하는 경로와, fragment 교체를 위한 경로가 있다. 단, 코드에 경로가 있다는 사실은 특정 checkpoint에서의 품질이나 stereo 보존을 보장하지 않는다. 실제 checkpoint와 소규모 입력으로 반드시 검증한다.
- 공개 README에서 generation-only checkpoint는 `flowr_root_spindr_base.ckpt`로 안내한다. 이를 우선 확인하되 graph conditioning과의 호환성을 실제 검사한다. joint checkpoint로 바꾸는 경우 별도 kernel로 기록하고 데이터를 섞지 않는다.
- **LiFT는 비교 대상으로 유지한다.** LiFT는 DrugFlow 기반이며 SMILES embedding을 soft semantic prior로 사용한다. 제안 SMILES와 최종 graph가 정확히 일치한다고 가정할 수 없으므로 동일 graph 재생성의 기본 구현으로 바로 사용하지 않는다.
- 새로운 FM backbone 설계는 첫 구현의 범위에 포함하지 않는다. 기존 모델에서 개입 선택의 이점을 확보한 후 별도 실험으로 검토한다.
- 이 문서는 공개 코드와 문서를 읽고 만든 구현 요구사항이다. 여기서 checkpoint inference를 실행하거나 성능을 검증한 것은 아니다.

확인한 공식 자료:

- FLOWR.root: https://github.com/jule-c/flowr_root
- 확인한 FLOWR.root revision: `739a33c2aab74d074dd24605b9a2e34dc809ae9c`
- graph mode: `flowr/gen/generate_from_pdb.py`, `flowr/data/interpolate.py`, `flowr/models/mol_builder.py`, `flowr/models/fm_pocket.py`
- LiFT: https://github.com/kasurl/LiFT
- 확인한 LiFT revision: `7a284d4a542f5437c30b95d0d3402b4b0cd9635a`
- LiFT 논문: https://arxiv.org/abs/2608.31009
- DrugFlow: https://github.com/LPDI-EPFL/DrugFlow
- Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B
- ChemDFM-v2.0-14B: https://huggingface.co/OpenDFM/ChemDFM-v2.0-14B

---

## Codex에 전달할 프롬프트

너는 연구용 소프트웨어를 구현하는 엔지니어다. 현재 작업 공간에 **DIRMO-3D**라는 3D structure-based molecular optimization 연구 코드를 구현하라. 계획 문서나 mock pipeline에서 끝내지 말고, 이용 가능한 실제 모델과 단일 protein–ligand 입력으로 작은 end-to-end 실행까지 진행하라. 필요한 자원이 없으면 독립적으로 구현 가능한 부분을 완료하고, 실제 실행의 blocker를 정확히 기록하라.

프로젝트의 작업용 논문 제목은 다음과 같다.

**DIRMO: Diagnosis-Guided Intervention Routing for 3D Structure-Based Molecular Optimization**

### 1. 연구 목표와 범위

입력은 protein/pocket PDB 또는 CIF, pocket에 정렬된 initial ligand SDF, 개선 목표, 보존 조건, 계산 예산이다. 다음 반복을 구현한다.

1. Initial ligand와 pocket을 평가한다.
2. 측정 근거로 현재 bottleneck과 원인 후보를 작성한다.
3. LLM이 실행 가능한 화학적 편집 후보와 검증할 가설을 제안한다.
4. Router가 구체적인 편집, 동일 graph의 3D 재생성, 생성 조건 변경, 종료 중 하나를 선택한다.
5. 동결된 pocket-conditioned FM으로 3D pose를 생성한다.
6. 실제 결과를 평가하고 계획한 편집, 보존 조건, 목표 개선을 확인한다.
7. 실패 근거와 실행 결과를 업데이트하고 예산 내에서 반복한다.
8. 전체 history에서 목표와 보존 조건을 만족하는 best feasible ligand를 출력한다.

초기에는 LLM과 FM을 동결한다. Rule router로 실행 데이터를 먼저 확보하고, 그 데이터로 작은 learned router를 학습하는 경로까지 구현한다. LLM finetuning, FM from-scratch training, 장기 RL은 첫 버전의 범위에 포함하지 않는다.

### 2. 시작 전 점검

- 현재 저장소, AGENTS.md, 설치된 환경, 기존 생성·평가 코드와 모델 위치를 먼저 확인한다. 기존 구현이 있으면 재사용하고 사용자의 수정 사항을 보존한다.
- 사용 가능한 GPU의 UUID, 이름, VRAM, 현재 사용량을 확인한다. 사용자가 사용할 수 있는 A6000 48GB 두 장만 지정해 사용한다.
- 다운로드된 Qwen3.5-4B, Qwen3.5-9B, ChemDFM-v2.0-14B의 실제 로컬 경로를 확인한다. 추측한 경로를 실행 설정에 넣지 않는다. 경로가 없으면 환경변수/설정 입력으로 남기고 필요한 항목을 보고한다.
- 데이터, protein preparation, receptor preparation, 생성 checkpoint, SMI-TED 등 실제 사용 경로의 가용성을 확인한다. 사용 가능한 모델을 중복 다운로드하지 않는다.
- 공식 저장소의 코드와 checkpoint interface를 읽고 revision, 라이선스, 의존성, native conditioning 기능을 기록한다. 대규모 데이터셋이나 전체 모델군의 다운로드는 자동으로 시작하지 않는다. 작은 실행에 필요한 자원부터 사용한다.
- `docs/backbone_audit.md`에 확인한 기능, checkpoint 선택, 알려진 한계, 호환성 결과를 작성한다.

### 3. GPU와 프로세스 분리 — 필수

**GPU 한 장은 LLM 전용, 다른 한 장은 3D 생성 전용으로 사용한다.**

- 기본 physical GPU 설정은 LLM=0, FM=1이며 config/environment에서 변경 가능해야 한다. 가능하면 UUID로 지정하고 실제 매핑을 로그에 남긴다.
- CPU orchestrator가 두 GPU worker를 별도 프로세스와 별도 환경으로 실행한다. 서로 다른 Python/CUDA 패키지가 필요해도 한 환경으로 강제로 합치지 않는다.
- LLM worker는 시작 전에 `CUDA_VISIBLE_DEVICES=<LLM_GPU>`를 설정한다. 프로세스 내부에서는 보이는 GPU가 한 장이므로 `cuda:0`을 사용한다.
- FM worker는 시작 전에 `CUDA_VISIBLE_DEVICES=<FM_GPU>`를 설정하고 내부에서 `cuda:0`을 사용한다. physical GPU=1이라는 이유로 worker 안에서 `cuda:1`을 지정하지 않는다.
- CUDA mask는 CUDA 초기화 전에 child process environment에 전달한다. Worker는 두 GPU 전체를 자동 탐색하거나 모델을 분산시키지 않는다. LLM tensor parallel size는 1이다.
- Orchestrator와 CPU 평가 프로세스는 GPU를 사용하지 않는다. Vina/PoseBusters/RDKit도 기본 CPU 실행으로 설정한다.
- 모델은 각 worker에서 한 번 로드해 resident 상태로 유지한다. 초기 구현은 localhost HTTP 또는 검증된 JSON IPC 중 하나를 일관되게 사용한다. GPU 요청 동시성은 장당 1부터 시작한다.
- Health endpoint/command에서 PID, visible CUDA devices, 실제 GPU UUID, model/checkpoint, peak memory를 확인할 수 있게 한다.
- 기본 추론 batch는 1이다. LLM context/output cap과 FM atom/pocket size, seed 수를 제한한다. OOM 시 제한된 재시도와 batch 감소를 적용하고 모든 실제 시도를 예산에 반영한다.
- 작은 gate 학습은 FM worker를 종료하거나 unload한 뒤 FM GPU에서 수행한다. Gate 추론은 초기에는 CPU에서 수행한다.

### 4. LLM adapter

- 기본 backend는 **로컬 Qwen3.5-9B**다. Qwen3.5-4B, ChemDFM-v2.0-14B를 같은 역할에서 config로 교체할 수 있게 한다. 실행 시 하나의 LLM만 로드한다.
- 모델별 공식 tokenizer, chat template, loading class, decoding 방식과 호환되는 설치 버전을 확인한다. 세 모델을 무조건 같은 loader로 처리하지 않는다.
- 공식 권장 추론 방식 중 실제 A6000에서 작동하는 것을 사용한다. 먼저 비양자화 16bit 추론을 확인하고, 양자화는 별도 config로 둔다. 지원되지 않는 FP8/attention kernel을 강제하지 않는다.
- Qwen의 thinking 여부는 adapter 설정으로 명시하고 실제 적용 여부를 확인한다. 기본 pipeline 검증은 짧은 non-thinking structured output으로 시작한다. Thinking 비교 시 token과 시간을 모두 기록한다.
- ChemDFM-v2.0-14B를 ChemDFM-R로 취급하지 않는다. ChemDFM의 SMILES 입력은 공식 지침에 맞춰 검증·정규화하되 atom mapping은 별도로 유지한다.
- LLM은 수치 지표를 계산하지 않는다. 도구가 계산한 진단 요약과 목표·보존 조건을 읽고 가설과 후보를 제안한다. LLM이 말한 효능·독성·성공 확률은 측정 결과로 기록하지 않는다.
- JSON/Pydantic schema로 `hypotheses`, `evidence_ids`, `edit_candidates`, `expected_tradeoffs`를 검증한다. 재출력/repair는 제한 횟수만 허용하고 원문 및 실패도 저장한다.
- 편집 후보에는 `candidate_id`, `parent_graph_id`, 변경 원자/fragment, mapped product SMILES 또는 명시적인 edit operation, 원자 대응 정보, 보호 조건에 대한 설명을 포함한다.
- RDKit이 실제 product graph를 생성/검증한다. LLM 원자 인덱스와 atom map은 검증 전까지 신뢰하지 않는다. Fragment 대응이 애매하면 `mapping_ambiguous`로 기록한다.
- LLM이 만든 shell command나 임의 Python 코드를 실행하지 않는다. 미리 정의된 edit executor와 evaluation tool만 사용한다.
- 모델별 프롬프트, raw response, 생성 토큰, latency, parsing/chemical validity, 중복 후보를 기록한다. 캐시 키에는 모델 revision과 generation config를 포함한다.

### 5. FM backbone 결정과 adapter

**주 backbone은 FLOWR.root의 공개 pretrained pocket-conditioned 생성기다.**

- 공식 upstream: https://github.com/jule-c/flowr_root
- 처음 확인할 revision: `739a33c2aab74d074dd24605b9a2e34dc809ae9c`. 이를 기준으로 읽되 실제 설치 revision을 고정해 기록한다. 변경한다면 이유와 기능 차이를 기록한다.
- 생성 전용 `flowr_root_spindr_base.ckpt`를 우선 검토한다. 실제 checkpoint hyperparameters, 코드 호환성, graph-conditioned generation을 확인한 뒤 채택한다. Joint checkpoint가 필요하면 다른 kernel로 명시하고 선정 이유를 남긴다. Affinity head는 초기 주 평가 oracle로 사용하지 않는다.
- 공개 코드에는 `graph_inpainting` 경로가 있고 random/conformer/harmonic prior 선택이 있다. CLI 이름을 믿고 끝내지 말고 data construction, atom/bond overwrite, ODE integration, molecule builder까지 실제 경로를 읽는다.
- **Whole-graph conditioning과 fragment coordinate inpainting은 별개의 기능이다.** 모든 fragment 좌표까지 고정해 아무것도 생성하지 않는 동작을 pose 재생성으로 구현하면 안 된다.
- Pocket 좌표와 atom/bond graph를 조건으로 주고 ligand 좌표를 재생성하는 mode를 실제 검사한다. 우선 random graph prior와 upstream에서 지원하는 solver를 사용하고, 고급 prior/solver 조합은 별도 검증 후 추가한다.
- 입력 초기 ligand와 proposed edited graph에 필요한 upstream data object를 만드는 adapter를 구현한다. 새 graph의 conformer가 입력 구조상 필요하면 RDKit conformer는 초기 입력 준비로만 사용하고, 최종 pose는 FM 생성 결과임을 구분한다.
- Backend의 조건 입력은 typed object로 만든다: graph, pocket, reference pose, protected atom mapping, supported prior/constraint parameters. 지원되지 않는 guidance는 명확한 capability error를 내며 무시하지 않는다.
- 임의의 pharmacophore/QED scalar를 FM이 native guidance로 받아들인다고 가정하지 않는다. 첫 버전은 검증된 graph conditioning과 native 생성 조건, 외부 평가로 구현한다. 새로운 velocity guidance 학습은 별도 후속 단계로 분리한다.
- Raw generation과 postprocessed/relaxed 결과를 별도로 저장한다. 기본 평가는 raw pose에서 수행한다. 자동 molecular repair/relaxation이 있다면 설정과 적용 이력, 적용 전후 graph/pose를 기록한다.
- Graph 고정 검증은 element, atom count, connectivity, bond order, formal charge, 지정 stereochemistry를 포함한다. Aromatic representation의 차이는 일관된 정규화 후 비교한다. Atom ordering은 graph isomorphism과 mapping으로 처리한다.
- 지원되지 않는 element/charge/stereo 또는 graph 변경은 명시적 실패다. 예상 graph와 실제 graph가 다르면 `graph_mismatch`로 처리한다. Silent repair로 다른 분자를 같은 후보로 보고하지 않는다.
- Chemical feature 보존과 pocket-relative pose 보존을 각각 확인한다. Protein을 고정한 상태에서 ligand만 최적 정렬해 interaction loss를 숨기지 않는다.
- 공개 CLI의 validity/uniqueness/diversity filtering과 내부 oversampling이 실패 횟수를 숨길 수 있으므로 raw draw 수, build failure, filtered/repaired sample을 추적한다. 같은 graph의 다른 pose를 SMILES 중복이라는 이유로 제거하지 않는다.
- 특정 checkpoint에서 graph mode가 불가능하면 기능을 거짓으로 구현하지 않는다. Independent modules를 완료하고 해당 mode를 disabled/blocker로 기록한다. RDKit embedding이나 docking을 조용히 FM 생성 대신 사용하지 않는다.

공통 FM adapter에는 `capabilities`, `generate_fixed_graph`, `generate_local_fragment`(지원 시), `health`를 둔다. Local fragment generation은 exact graph edit completion과 별도 mode로 구분한다.

LiFT adapter는 후속 비교용으로 둔다. 공식 upstream은 https://github.com/kasurl/LiFT 이고 DrugFlow 기반이다. SMI-TED encoding이 필요한 semantic generation이며 output graph가 제안 SMILES와 동일하다고 가정하지 않는다. 실제 구현한다면 `semantic_joint_generation`으로 기록하고 최종 graph에서 모든 지표를 다시 계산한다. 첫 prototype 완성을 위해 LiFT 전체 재학습이나 새 FM 설계를 시작하지 않는다.

### 6. 입력, 목표, 진단

- 입력 JSONL/manifest에는 `case_id`, `target_id`, protein/pocket path, initial ligand path, objective config, protected features, budgets를 둔다. Coordinate units는 Å이며 입력 pocket–ligand 정렬과 수소/양성자화 처리를 기록한다.
- Pocket는 initial pose를 기준으로 준비한 뒤 반복 사이에 고정한다. 편집 pose마다 pocket를 다시 잘라 평가 환경을 바꾸지 않는다. Protein 전체와 pocket의 용도를 구분한다.
- 개선 목표는 시작 시 고정한다. QED/SA 및 Vina의 방향과 허용 손실을 명시하고, raw metric과 의사결정 utility를 분리한다. SA는 RDKit contrib SA score를 사용한 경우 낮을수록 좋다는 convention을 기록한다.
- 보호할 pharmacophore/interaction은 사용자가 지정하거나 근거와 함께 제안한다. 초기 ligand에서 검출된 모든 feature를 필수 제약으로 만들지 않는다. 증거가 약한 조건은 soft constraint로 구분한다.
- Graph 기반 QED, SA, MW, logP, HBD/HBA, TPSA, rotatable bonds, Rule of Five는 같은 chemical representation에서 pose와 무관하므로 graph 단계에서 계산하고 캐시한다.
- PoseBusters는 사용 버전·preset·reference 입력을 명시한다. 구조와 화학이 달라진 최적화 후보에 reference-ligand identity/RMSD 검사를 잘못 적용하지 않는다. 각 적용 가능한 check와 누락 이유를 저장한다.
- Physical validity는 ligand 내부 geometry와 protein overlap 등을 분해해 저장한다. 측정 가능하면 minimum contact margin, clash severity 등 continuous values도 저장하되 upstream 정의와 자체 구현을 구분한다.
- Pharmacophore는 feature 종류·위치·대응 관계를 저장하고, interaction은 residue ID와 donor/acceptor 등 역할을 포함한다. RDKit feature와 PLIP/ProLIF interaction을 같은 지표로 합치지 않는다.
- Vina는 score-only, local minimization, redocking을 구분한다. 주 pose 평가는 준비한 receptor와 동일한 box의 score-only로 하고, redocking은 선택한 별도 평가로 기록한다. Redocking 결과로 생성 pose 유효성을 대체하지 않는다.
- 외부 평가 dependency/score가 없거나 실패하면 `missing`/`failed`와 사유를 기록한다. 기본적으로 필수 평가가 빠진 후보를 성공으로 채택하지 않는다. Demo에서 평가를 끄면 incomplete evaluation임을 표시한다.
- 단일 pose의 실패를 분자의 근본적 불가능성으로 단정하지 않는다. 같은 graph의 여러 seed 결과와 조건 변경 결과로 recoverability 가설을 갱신한다.
- 성공률의 uncertainty는 충분한 시도 수가 있을 때 Wilson interval 또는 명시한 Beta posterior로 표현한다. 소수 seed 결과는 불확실성이 크다고 표시한다. Softmax routing probability를 성공 확률로 보고하지 않는다.

### 7. 상태, 행동, 반복 제어

상태는 `original_ligand`, `best_feasible`, 현재 평가/복구 중인 `work_item`, 관측된 실패 근거, hypothesis history, 남은 예산을 구분한다. 실패한 edited graph도 work item으로 유지하여 다시 pose를 생성할 수 있어야 한다. Original graph만 반복 생성하여 편집 결과의 복구를 놓치지 않는다.

초기 행동 집합:

- `CHEMICAL_EDIT`: 특정 edited graph 후보를 만들고 FM으로 completion한다. 후보 ID, edit operation, mapping, seed 수를 포함한다.
- `POSE_RESAMPLE`: 현재 work item의 graph를 그대로 유지하고 다른 seed로 좌표만 생성한다.
- `CHANGE_GENERATION_CONDITION`: graph는 유지하고 검증된 prior/constraint parameter를 변경한다. 실제 지원하는 조건과 값만 허용한다.
- `STOP`: best feasible를 반환하거나 개선 실패 상태를 반환한다.
- `EXTRA_VALIDATION`: schema에는 확장 가능하게 두되 첫 router에서는 비활성화한다. 활성화할 때는 즉각적인 물성 변화가 아닌 후속 의사결정 가치와 비용을 고려하는 별도 설계를 요구한다.

진단과 행동에 대한 evidence ID를 연결한다. 설명은 관측된 사실과 검증할 가설을 분리한다. 단순 서술을 causal proof로 표시하지 않는다.

보호 graph가 깨지거나 graph가 invalid인 후보는 제외한다. Geometry가 실패한 유효 graph는 제한된 recovery attempt를 허용한다. Success/budget/maximum rounds/plateau로 종료하며, history의 마지막 후보가 아닌 best feasible를 출력한다. 개선한 후보가 없으면 original과 `no_improvement` 상태를 반환하고 최적화 성공으로 기록하지 않는다.

Budget는 FM raw draws/NFE, LLM token/호출, 평가 호출, wall time을 각각 관리한다. LLM parsing retry, OOM retry, 내부 filtering/oversampling도 비용에 포함한다. 하나의 request가 내부에서 무제한으로 성공 샘플을 모으게 하지 않는다.

### 8. Router와 학습 데이터

- 먼저 transparent rule router를 구현한다. 예: graph objective 미달이면 구체적인 edit 후보를 검토하고, graph는 목표를 만족하지만 geometry가 실패하면 제한된 pose recovery를 시험한다. 규칙을 ground-truth 원인 라벨로 취급하지 않는다.
- Learned router는 작은 action-conditioned MLP로 시작한다. 상태 수치, uncertainty, history summary, 남은 budget, 행동 종류와 구체적인 edit/생성 파라미터를 입력으로 받는다. Attention은 선택적 후속 ablation이다.
- 각 행동의 예상 개선과 예상 비용을 예측하고 infeasible action mask 이후 선택한다. Token prediction이나 임의 failure class만 학습하여 행동 효과를 안다고 주장하지 않는다.
- Data collector는 일부 동일한 state snapshot에서 여러 실행 가능한 행동을 branch로 시험한다. 동일 kernel과 공정한 seed/budget protocol에서 실제 결과를 기록한다. 선택하지 않은 행동의 outcome을 만들어내지 않는다.
- 기록 필드: pre-action state ID, target/series, action ID와 parameters, model/kernel ID, observation-before, resulting graph/poses, post-action observations, feasibility, utility change, compute cost, seed, evaluation status.
- 생성 kernel ID에는 repo revision, checkpoint SHA256, preparation, prior, solver/NFE, repair/relaxation 설정을 포함한다. Kernel이 달라지면 reward label을 같은 것으로 취급하지 않는다.
- Supervised regression 또는 ranking으로 실제 행동 결과를 학습한다. 단 한 번의 noisy outcome을 확정적인 최적 행동 라벨로 만들지 않는다. 반복 seed와 cluster-level bootstrap으로 비교를 지원한다.
- 초기 rule router 실행 결과만으로 모든 행동을 비교할 수 있다고 가정하지 않는다. Branch trials의 coverage를 보고한다. 선택한 행동만 관측하는 online 학습은 후속 bandit mode로 구분한다.
- State-derived feature만 모델 입력으로 사용하고 action 후 결과를 입력에 섞지 않는다. Standardization/calibration은 train data에서만 수행한다.
- Target 또는 ligand series 단위의 group split을 사용한다. 같은 state의 branch와 seed는 같은 split에 둔다. 데이터가 적으면 grouped CV를 사용하며 사례 수와 불확실성을 함께 보고한다.
- `RuleRouter`, `LearnedRouter`, `LLMRouterBaseline`, fixed-action/fixed-sequence baseline을 같은 executor interface로 교체할 수 있게 한다.

### 9. 구조와 실행 명령

작업 공간에 맞춰 구조를 조정하되 다음 책임을 분리한다.

- `config`: 모델 경로, GPU UUID, 서비스, 생성·평가, objective/budget 설정.
- `schemas`: 입력, state, action, evidence, observations, result.
- `llm`: 모델별 adapter, prompt, structured output parser.
- `chemistry`: graph edit/validation/mapping, properties, pharmacophore.
- `generation`: FLOWR.root adapter, capability audit, 별도 FM worker/client.
- `evaluation`: PoseBusters, interaction, Vina, 실제 필요시 strain adapter.
- `diagnosis`: evidence assembly, hypotheses, recoverability statistics.
- `routing`: rule/learned/baseline policies, feasibility mask.
- `pipeline`: bounded loop, budget, checkpoint/resume, cache.
- `data`와 `training`: branch data collector, group split, router training/evaluation.
- `cli`와 `scripts`: doctor, workers, optimize, collect, train-router, compare.

CLI는 최소 다음 기능을 제공한다. 프로젝트에서 실제 구현한 명령을 README에 적는다.

1. `doctor`: 경로/환경/GPU/backend capabilities 확인.
2. `start-workers` / `stop-workers`: 두 프로세스의 GPU 분리와 health 확인.
3. `evaluate-initial`: initial ligand와 objective 상태 평가.
4. `optimize`: 단일 입력 또는 manifest에서 bounded optimization.
5. `collect`: state별 여러 개입의 branch dataset 수집.
6. `train-router`: 실제 수집 데이터로 MLP 학습.
7. `compare`: 고정 executor와 budget으로 정책 비교.

각 run에는 resolved config, input checksum, 모델/checkpoint revision, 환경 정보, actions/evidence/metrics JSONL, raw/final SDF, 비용 내역, final status와 resume state를 저장한다. Resume과 cache hit를 distinguish하고 checkpoint/state hash가 다르면 오래된 outcome을 재사용하지 않는다.

### 10. 단계별 실행과 검증

1. Audit와 CPU graph/evaluation 모듈을 구현하고 실제 initial ligand로 실행한다.
2. GPU 0에서 Qwen3.5-9B를 로드해 측정 근거에 대한 후보 JSON을 받고 RDKit 검증까지 수행한다.
3. GPU 1에서 실제 FM checkpoint를 로드해 unedited graph와 적어도 한 edited graph의 좌표를 생성한다. 전체 graph identity와 stereo, pocket coordinate frame을 검사한다.
4. Rule router로 짧은 전체 반복을 실행한다. 시작값은 한 target, round cap 2, 후보 cap 2, graph별 seed cap 2이며 전체 raw FM draw cap도 별도로 둔다. 너무 작은 예산 때문에 모든 경로가 실행되지 않으면 어떤 경로를 확인했는지 명시한다.
5. 소규모 branch data를 수집하고 learned router 학습 CLI가 실제 데이터를 처리하는지 확인한다. 작은 smoke dataset의 성능을 연구 결과로 주장하지 않는다.

의미 있는 테스트는 GPU 분리/physical→logical mapping, schema/invalid edit, graph 보존과 atom mapping, budget 상한과 실패 비용, missing metric 처리, best-feasible와 no-improvement 반환, group split leakage, resume 일관성에 집중한다. CPU unit test와 실제 모델 integration test를 구분한다. Mock은 unit test에만 사용하고 실제 integration 성공으로 보고하지 않는다.

실제 GPU peak memory와 LLM/FM/evaluation latency를 기록한다. 같은 graph의 여러 seed가 실행 가능하고 graph는 유지되는지 확인한다. Checkpoint가 없거나 dependency가 막히면 실행하지 못한 경로와 필요한 최소 자원을 정확하게 보고하고, 구현 가능한 나머지 부분은 완료한다.

### 11. 완료 보고

코드 파일을 만들고 필요한 작은 실행을 수행한 뒤 다음을 보고한다.

- 실제 구현한 파이프라인과 구현된/미지원된 mode.
- 사용한 모델 경로, FM checkpoint와 kernel ID, physical GPU 매핑.
- 설치·실행 명령과 수정할 최소 설정.
- 실제 실행 로그/출력 경로, graph 보존 결과, 평가 결과, peak VRAM과 비용.
- 통과한 검사와 실행하지 못한 검사, 남은 blocker.
- Rule router와 learned router의 상태를 구분한 설명.

대규모 실험을 자동으로 시작하지 말고, 작은 real-model end-to-end 실행을 완료한 뒤 reproducible한 실험 명령을 제공하라. 모델 성능, 물성, 불확실성, 비교 결과를 임의로 채우지 말고 모든 주장에 실제 실행 근거를 연결하라.
