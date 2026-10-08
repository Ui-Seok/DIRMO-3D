# 구현 검토 — 2026-10-08

이번 작업 범위는 사용자 후속 요청대로 코드 작성·검토다. LLM/FM GPU integration과 branch 수집·실데이터 gate 학습은 수행하지 않았다.

## 반영한 검토 사항

- Orchestrator 진입에서 CUDA mask를 비우며, 모델 package는 worker의 UUID mask 검사를 마친 후 import한다. GPU selector가 같은 physical device를 가리키면 시작을 거부한다. Worker logical device는 항상 `cuda:0`이다.
- LLM의 JSON과 chemical edit는 별도 검증한다. 실행기는 element replacement/terminal atom addition/removal/non-aromatic bond change만 허용하고 임의 코드/명령은 schema에 없다. Atom correspondence/changed maps와 보호 graph를 RDKit으로 확인한다.
- FLOWR graph prior는 discrete graph만 고정하며 pose 좌표는 생성한다. 최종 corrector의 discrete overwrite도 활성화한다. Fragment coordinate fixing과 혼동하지 않는다. Random/Euler 외 기능은 schema/capability에서 제한한다.
- Raw graph는 atom ordering과 aromatic representation을 정규화하고 graph isomorphism을 통해 비교한다. Charge/stereo/결합 차이 및 새 stereo specification은 explicit graph mismatch다. Symmetric atom ordering은 fixed frame displacement가 가장 작은 correspondence를 선택한다.
- 실패한 edited graph의 입력 conformer를 work item으로 남긴다. 다음 recovery는 그 edited graph를 사용한다. 입력 conformer 자체를 최종 FM 출력으로 채택하지 않는다.
- 필수 평가 누락/실패가 success로 흐르지 않는다. Vina는 CPU score-only이고 PoseBusters는 `dock` preset을 사용한다. PoseBusters의 renamed output 열을 읽도록 수정했다.
- Raw metric과 objective utility, target satisfaction, hard preservation, feasible 여부를 분리한다. History 전체에서 target을 만족하는 개선 후보를 선택하고, 마지막 후보를 무조건 반환하지 않는다.
- 실패·OOM·parsing retry도 비용에 포함한다. RPC 전 상한을 예약한 checkpoint를 저장하고, backend 실제 비용을 받으면 unused reservation만 환불한다. 이미 batch 1이므로 OOM 자동 반복을 하지 않는다.
- Resume은 input/config/code/model/environment 및 SDF/pocket checksum을 검사한다. 진행 중의 결과 미확인 요청을 자동 replay하지 않는다. LLM cache와 resume event는 따로 기록한다.
- Branch outcome은 실제 실행 결과만 저장한다. Pre-action feature에 objective/budget/uncertainty/구체적인 proposed edit 정보를 넣고 post-action outcome은 입력에 섞지 않는다. State와 target/series의 connected groups로 split leakage를 막는다.
- 같은 checkpoint여도 prior/solver/step/repair/preparation 등이 달라지면 kernel ID가 달라진다. 여러 kernel의 reward label은 자동으로 혼합하지 않는다. 학습된 행동 종류의 coverage 밖에서는 learned policy가 행동을 선택하지 않는다.
- Hypothesis와 outcome을 연결하되 자동 causal proof로 승격하지 않는다. Wilson interval은 적은 seed에서 uncertainty가 크다고 표시한다.

## 남아 있는 실험 전 제한

1. 제공된 FLOWR.root checkpoint `/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_base_spindr.ckpt`를 기본 설정에 연결했다. 별도 Python 3.12 환경은 아직 지정되지 않았으며 resident backend load와 upstream tensor interface는 정적 검토만 완료했다. 사용자 요청에 따라 이번 연결 변경에서는 테스트나 모델 로드·추론을 실행하지 않았다. 이 경로의 runtime bugs/quality/stereo 지원은 실제 integration에서 확인해야 한다.
2. Qwen tokenizer/loading-class import만 확인했다. LLM generation, chemistry proposal 유효율, BF16/FP16 inference와 peak VRAM은 검증하지 않았다. 양자화도 미검증 옵션이다.
3. Receptor/protein 준비는 자동으로 수행하지 않는다. Vina PDBQT와 ProLIF explicit-H protein을 사용자가 준비하고 provenance를 기록해야 한다. Coordinate frame은 입력 측에 책임이 있다.
4. Local fragment, LiFT comparison, scalar guidance, 고급 prior/solver, relaxation/redocking 및 EXTRA_VALIDATION은 명시적으로 미지원이다.
5. 실패해 utility를 측정할 수 없는 branch에는 improvement label이 없다. Learned regression은 측정 가능한 metric utility/cost만 학습하며 feasibility predictor나 calibrated success probability를 제공하지 않는다. 초기 데이터의 coverage bias가 남는다.
6. Learned router는 하나의 kernel artifact와 지원 행동 종류에 한정한다. Condition-changing policy 학습은 kernel별 ensemble 또는 별도 검증된 multi-kernel 설계가 필요하다.
7. Clean-boundary resume은 구현됐지만 interrupted remote action 결과를 자동 회수하는 분산 transaction 기능은 없다. `pending_action` 상태에서 명시적으로 중단해 중복 비용을 방지한다.
8. CPU tool bundle의 wall-time 비용은 실제 완료 시 측정된다. 전체 loop와 RPC는 cap을 검사하지만 단일 CPU 평가의 실행 시간에는 process-level hard deadline이 없다.
9. FM model forward NFE가 wrapper의 step/recycle 상한을 넘으면 실행을 중단한다. 신뢰할 수 있는 pinned backend의 상한을 사용하며 현재는 checkpoint에서 실제 확인하지 못했다.

검증 로그는 `reports/unit-tests.log`, 요약은 `reports/unit-tests.json`에 저장한다. 단위 검사에서 사용하는 mock worker·synthetic molecule은 실제 모델 성능 근거가 아니다. 로컬 초기 ligand의 CPU 평가 로그는 별도로 `reports/cpu-validation-final/`에 저장한다.
