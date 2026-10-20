# CrossDocked2020: FLOWR.root SBDD 생성 성능 평가

이 실험은 기존 ligand를 개선하는 SBMO 실험과 별개다. 새 ligand의 graph와 좌표를
동시에 생성하는 FLOWR.root의 기본 SBDD 성능을 측정한다. 추가 학습과 LLM 호출은 없다.

## 2026-10-19 실행 설정

- 물리 GPU 5, UUID `GPU-765885b3-d3fe-46b1-bcd3-a18e1130a74e`.
- 현재 보유한 SPINDR generation-only base checkpoint, SHA256
  `78ef9ebe4459df1ce2668685785602a37288594db83d2bb5d970c8bc06e1ce9d`.
- 고정 FLOWR repository revision: `739a33c2aab74d074dd24605b9a2e34dc809ae9c`.
- CrossDocked test 목록 전체 100개 × 100회 시도 = 10,000회, Euler 100-step, batch 32.
- 원본 pocket10에서 ligand까지 7Å 이내인 원자가 있는 잔기를 통째로 추출한다.
  생성 조건은 고정된 protein pocket, 원본 ligand의 heavy-atom 수, 결합 부위 중심이다.
  원본 ligand graph나 원자별 좌표를 고정하지 않는다. 원자 수까지 자유로운 생성은 아니다.
- 성공·PB·중복 여부에 따라 재시도하거나 좋은 결과만 선택하지 않는다. OOM일 때만
  배치를 줄인다. 실패한 최종 슬롯도 10,000회 분모에 포함된다.
- 생성 결과에 valence repair, UFF/MMFF 보정 또는 docking 보정을 적용해 덮어쓰지 않는다.
  Vina local optimization/redocking 및 MMFF 최소화는 평가용 복사본에서 수행한다.

실제 test 목록은 `/LocalData1/uslee/datasets/moljo/data_zip_extract/data/test_list.tsv`이며,
`/LocalData1/uslee/datasets/processed_crossdock_noH_full_temp/test.npz`의 100개 이름과
일치하는지 검증한다. 입력은 원래 폴더 구조와 좌표를 유지한
`/LocalData1/uslee/datasets/crossdocked_pocket10`에서 읽는다. CrossDocked reference는
교차 docking pose를 포함하므로 모든 reference를 실험적으로 측정한 crystal pose라고 부르지 않는다.

## 측정 항목과 해석

| 항목 | 정의/주의 |
| --- | --- |
| RDKit validity / connected validity | 파싱·sanitization 성공과 단일 연결 성분 여부를 각각 기록 |
| PB-valid | PoseBusters `dock`, 원본 pocket10을 protein 입력으로 사용; 정답 graph/RMSD 비교 없음 |
| PB 분모 | 모든 생성 시도, RDKit-valid 출력, PB 측정 완료 출력의 비율을 따로 기록; timeout은 측정 누락 |
| Vina score | FM이 생성한 원래 pose를 이동하지 않고 계산 |
| Vina minimize | 생성 pose 근처에서 Vina local optimization 후 계산 |
| Vina dock | 같은 분자를 다시 docking, exhaustiveness 8; 원본 pose 점수와 별개 |
| High affinity | 각 Vina 항목별로 같은 pocket의 reference보다 낮은 점수인 비율; 실험 affinity 아님 |
| QED / SA / LogP | RDKit QED, SA 원점수 1–10(낮을수록 쉬움), Crippen LogP |
| 기타 2D 속성 | MW, HBD/HBA, TPSA, 회전 결합, ring 수, heavy-atom 수, Lipinski 조건 |
| 다양성 / 중복 | pocket별 Morgan radius2/2048-bit Tanimoto 거리와 unique canonical isomeric SMILES 수 |
| Novelty | local CrossDocked train SMILES에 없는 비율; FLOWR 전체 사전학습 대비 신규성은 확인 불가 |
| Strain | MMFF94s에서 생성 pose와 1000-iteration local minimization 후 에너지 차이; 수렴 여부 별도 |
| 결합 길이/각도 W1 | reference와 생성물 분포의 Wasserstein 거리, pooled 및 bond order/중심 원소별 |
| 효율 | batch 시간, forward 횟수, peak GPU allocation; CPU 평가 시간 별도 |

논문 표의 CrossDocked 모델은 CrossDocked 학습 base다. 이번 weight는 SPINDR base이므로
논문의 수치 재현으로 해석하지 않는다. 또한 현재 GenBench3D가 설치되어 있지 않아
strain은 별도 MMFF94s 지표이며, W1도 논문의 binned/weighted 통계와 완전히 같은 구현이 아니다.
실험 affinity, pKa/용해도, xTB/QM/FEP는 측정하지 않는다.

원본 pocket10에서 Open Babel로 rigid PDBQT를 새로 만든다. 100개 전부 준비되었고,
원래 heavy atom의 개수와 양방향 좌표 일치 검증에서 최대 오차는 **0Å**였다.
H 추가와 Gasteiger charge/AutoDock atom typing만 적용하며, 원자 좌표를 최적화하거나
빠진 heavy atom을 복원하지 않는다. 생리적 pH에 따른 protonation을 보장하는 준비법은 아니다.
Open Babel의 aromaticity 경고 등은 각 receptor의 준비 로그에 남긴다.

초기 점검에서 기존 processed PDBQT의 일부 sidechain 좌표가 원래 pocket10과
다른 것을 발견해, 해당 receptor를 사용한 점수는 `superseded_initial_receptor_protocol/`로
분리했다. FM 생성 input과 생성 함수에는 변화가 없어 이미 저장한 500개 raw draw는
보관하고 재사용했다. 기존/새 generation pocket SHA256과 생성 함수 AST가 동일함을 검증했다.
최종 protocol ID는 `crossdocked_sbdd_raw_v2_coordinate_matched`이며,
최종 집계는 새 receptor로 측정한 결과만 읽는다.

## 코드 및 실행

- `scripts/benchmark_crossdocked_sbdd.py`: CLI 진입점.
- `src/dirmo/evaluation/crossdocked_sbdd.py`: test 목록 검증, pocket 준비, GPU 생성 실행,
  CPU 평가, 재시작, 집계. CPU는 최대 12개 pocket을 병렬 처리한다.
- `src/dirmo/generation/de_novo.py`: reference graph를 고정하지 않는 기존 raw generation adapter.
- `tests/test_crossdocked_sbdd.py`: 실패/누락 분모, Vina 비교, 구조 분포, novelty SMILES 검증.

`dirmo` 환경에서 repository 루트를 기준으로 실행한다.

```bash
python scripts/benchmark_crossdocked_sbdd.py run \
  --output /LocalData1/uslee/experiments/DIRMO-3D/crossdocked_sbdd_gpu5_base_20261019_v1
```

같은 명령은 저장된 슬롯을 유지하고 누락된 작업을 이어간다. 설정을 바꾸면 새 output
경로를 사용한다. 실행 스크립트가 물리 GPU 번호를 UUID로 해석하고 생성 process에만
전달하므로 외부에서 `CUDA_VISIBLE_DEVICES`를 설정할 필요는 없다.

## 결과 위치

`/LocalData1/uslee/experiments/DIRMO-3D/crossdocked_sbdd_gpu5_base_20261019_v1`

repository `runs/`에도 같은 이름의 symlink가 있다. home 저장소가 거의 가득 차서
실험 출력, PB cache, 임시 파일은 모두 `/LocalData1`에 저장한다.

- `study.json`: 설정, 입력 hash, checkpoint, 알려진 한계.
- `launch.json`, `status.json`, `generation_worker.json`: process 및 진행 상태.
- `progress.json`: 분자 단위 처리 수 `evaluated_draws`, 처리율 `evaluation_percent`,
  pocket 전체 완료 수 `evaluated_cases`, KST 갱신 시각 `updated_kst` 및
  최근 결과 저장 시각 `latest_evaluation_kst`. 현재 실험에서는 15초마다 갱신한다.
- `kernel.json`, `generation_batches.jsonl`: 실제 sampling 설정과 시간·메모리.
- `controller.log`, `generation.log`, `cases/*/evaluation.log`: 오류/진행 로그.
- `cases/*/generated/000..099/raw.sdf`: 원래 생성 결과; build 실패 슬롯에는 없음.
- `cases/*/generated/*/{result,evaluation,docking}.json`: 생성 상태와 분자별 평가.
- `cases/*/reference_evaluation.json`: 같은 평가 방식으로 측정한 원본 ligand.
- `summary.json`, `molecule_metrics.csv`, `REPORT.md`: 전체 완료 시 집계.
- `source/`: 실행 시작 시 코드 복사본.

`evaluated_cases`는 분자 1개마다 증가하지 않는다. 한 pocket에 배정된 100회 시도를
모두 처리해야 1 증가하므로, 일부 pocket의 Vina redocking이 오래 걸리면 숫자가
한동안 그대로일 수 있다. `evaluated_draws`로 분자 단위 진행 상황을 확인한다.
처리 수에는 생성 실패나 개별 metric 측정 누락으로 기록을 완료한 슬롯도 포함된다.

2026-10-19 실행 중인 controller는 이전 코드가 메모리에 올라 있어 `status.json`에
기존 UTC 형식을 계속 기록한다. 현재 실험의 KST/분자 진행률은 별도 읽기 전용 observer가
`progress.json`에 기록한다. 평가 process는 재시작하지 않았다. 이후 새 실행에서는
`status.json`에도 분자 단위 진행률과 KST 갱신 시각을 함께 기록한다.

관찰 코드: `scripts/watch_sbdd_progress.py`, `src/dirmo/evaluation/sbdd_progress.py`.
필요시 실행하는 observer 명령은 다음과 같으며, 중복 observer는 lock으로 방지한다.

```bash
python scripts/watch_sbdd_progress.py \
  --output /LocalData1/uslee/experiments/DIRMO-3D/crossdocked_sbdd_gpu5_base_20261019_v1
```

PoseBusters와 docking child에는 각각 600초 제한이 있으며 누락 이유를 저장한다.
별도 보고서 생성은 원래 run과 겹치지 않는 시점에 다음 명령으로 할 수 있다.

```bash
python scripts/benchmark_crossdocked_sbdd.py report \
  --output /LocalData1/uslee/experiments/DIRMO-3D/crossdocked_sbdd_gpu5_base_20261019_v1
```

공식 비교 조건: [FLOWR.root 논문 §5.2](https://arxiv.org/html/2510.02578v3#S5.SS2),
[공식 repository](https://github.com/jule-c/flowr_root).
