# 편집 graph의 FLOWR.root 좌표 생성 비교

실행: `runs/fm_edited_graph_gpu3_100_20261016_v1`. GPU3의 기존 작업은 유지한다.

원본 graph + 같은 말단 C-C 탄소를 N/O/F로 바꾼 세 편집 graph, 각 8개 시작 pose, whole/core 두 FM 조건을 사용한다. 100 pockets × 4 graphs × 8 draws × 2 conditions = 6,400개 예정 슬롯이다. FLOWR.root base와 64-step으로 고정하며 FM/LLM 학습과 LLM 호출은 없다.

기존 2,000개 PDBbind2019 복합체 pool을 seed 20261016으로 섞고 exact protein sequence를 중복 제거한다. 기존에 정의한 2–8개 원자의 terminal region 안에 편집 가능한 말단 탄소가 있는 사례를 사용한다. 전체 원자 수와 결합을 유지하는 N/O/F 교체 규칙은 지표를 보기 전에 고정하며 QED/Vina로 순위를 정하지 않는다. PB/생성 결과를 본 뒤 cohort나 편집을 교체하지 않는다. 선택 전의 가장 가까운 protein–ligand 거리가 공유결합 반경 합의 1.2배보다 짧으면 보수적으로 제외한다. 이는 공유결합의 확인 annotation이 아니라 매우 짧은 접촉의 적격성 제외다. 제외 목록과 각 입력 해시를 저장한다.

편집 target은 conformer 없는 RDKit binary graph와 atom order로 저장한다. Mapped SMILES는 사람이 확인할 label로도 기록한다. 일부 분자의 mapped-SMILES round-trip이 입체화학을 바꿀 수 있어 그 경로로 모델 입력을 복원하지 않는다. Graph-only ETKDGv3로 각 seed의 conformer를 만들고 보호 영역의 국소 anchor에 배치한 뒤 보호 좌표를 정확히 넣는다. 원본 자유 영역 XYZ를 이 함수에 전달하지 않는다. 두 FM 조건은 이 초기 SDF를 공유한다. Whole도 초기 core 위치 정보를 받으며, integration 중 hard hold 여부만 다르므로 blind docking 비교가 아니다.

Whole: 편집 후 전체 graph 고정, 전체 XYZ 자유. Core: 같은 전체 graph 고정, 기존 core XYZ 고정, terminal region XYZ 자유. 조건 SMILES를 global embedding으로 바꾸는 LiFT 실험이 아니다. 자유 영역 graph를 FM이 재설계하는 실험과도 구분한다.

같은 초기 입력 자체, 전체 좌표 UFF200, core 보호 UFF200을 CPU 대조군으로 평가한다. UFF는 ligand-only이며 단백질 에너지 항이 없다. 각 초기 입력 및 FM/UFF 결과에서 PB와 Vina를 평가하므로 원본 100개를 포함한 예정 측정 슬롯은 16,100개다. 동일 graph의 PB 50-conformer 기준 에너지를 엄격한 identity cache로 재사용한다. PB 측정은 별도 CPU process의 600초 제한을 사용하고 시간 초과를 미측정으로 기록한다.

주 endpoint: 전체 예정 FM 슬롯에 대한 PB dock 20-check 통과 + target graph/입체화학 보존. Core 조건은 보호 heavy XYZ 최대 변위 ≤0.001 Å도 요구한다. 모든 후보 통과율과 동일 graph의 8개 중 성공 pocket 비율을 함께 보고한다. 원본과 편집 graph의 QED/SA/LogP 등은 미리 기록한다. 이러한 descriptor의 변화는 graph 편집의 효과이며 FM pose 생성의 효과로 돌리지 않는다. 새로운 편집 분자의 실험 pose가 없으므로 원본 전체 ligand RMSD를 pose 정확도로 사용하지 않는다.

Vina는 standard amino acid receptor의 score-only 대리 점수다. Meeko template/HIE/Gasteiger, 원본 protein heavy XYZ, 물·보조인자·비표준 잔기 제외, native 중심 고정 40 Å box를 사용한다. 수용체 준비 실패를 임의 residue 삭제로 통과시키지 않는다. 누락을 명시하고 유효한 paired subset에서만 Vina 변화량을 비교한다.

코드: `scripts/benchmark_edited_graph.py`, `src/dirmo/evaluation/edited_graph.py`. 기존 `PoseRepairBackend`를 재사용한다. 실제 모델 입력 mask와 tensor round-trip, 최종 SDF graph/stereo/보호 XYZ를 확인한다. Graph와 core 고정이 맞아도 PB·affinity 개선은 보장되지 않는다.

상태는 `controller.json`, `gpu_worker.json`, `summary.json`, `REPORT.md`에서 확인한다. 원자 대응과 edit은 `cases/*/case.json`, 시작 pose는 `cases/*/inputs`, raw FM 결과는 `draws`, 전체 측정은 `evaluations/*/result.json`에 저장한다. 모든 실패 슬롯을 예정 분모에 유지하며 같은 폴더로 재개할 수 있다.

2026-10-18 재개에서는 기존 100개 cohort, 400개 target graph, 시작 pose, seed와 FM sampler를 그대로 유지했다. 400개 graph의 radical/isotope 적격성을 사전 검사했으며, 저장된 5edl 입력의 네 graph가 radical 전자를 포함했다. 해당 입력은 변경하지 않고 64개 예정 FM 슬롯을 `unsupported_input`으로 남긴다. 기존 실패 64개와 중단 당시 상태는 `recovery/resume_20261018_v2`에 보관했다. 한 입력의 tensor round-trip 불일치는 batch를 한 개씩 확인한 뒤 해당 입력에 기록하고 다음 입력을 진행한다. 그 외 예상하지 못한 sampler/프로그램 오류의 중단 검사는 유지한다. 생성 처리 완료 수에는 unsupported 사례도 포함되므로 실제 SDF 수와 각 실패 수를 함께 읽어야 한다. 이전 43개 pocket의 생성 결과와 완결된 CPU 평가 30개는 재사용하며 누락 평가와 후속 생성만 실행한다.
