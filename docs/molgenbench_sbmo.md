# MolGenBench full-ligand SBMO 외부 평가

이 실험은 **완전한 초기 ligand + pocket → 화학 편집 후보 → 동결 Router 선택 → FLOWR.root 좌표 생성**을 평가한다. 공식 MolGenBench의 scaffold-only H2L 결과와 직접 비교하지 않는다. LLM 호출, FM 학습, Router 재학습은 없다.

## 고정한 설정

- 공식 Zenodo record `20233505`, Version 4. 코드 commit `f7b6cec1892cb575dc3adee4f5a71ee16ef04c39`.
- archive에서 reference/pocket/scaffold 파일만 선택 다운로드한다. ZIP CRC32를 검증하고 SHA256을 기록한다. 전체 archive MD5는 선택 다운로드에서 검증했다고 주장하지 않는다.
- CrossDocked 공식 중복 타깃과 로컬 train의 source PDB/정확한 초기 분자 그래프 중복을 제외하고 seed `20261105`로 20타깃, 1 series씩 고정한다. 이후 생성·측정 결과에 따라 대체하지 않는다.
- 최대 8개 화학 편집 × strict/relaxed × 3 seeds, 64 steps. 최대 960 draws. 후보가 없으면 빈 메뉴와 실패 사유를 보존한다.
- strict는 유지 원자의 XYZ를 고정한다. relaxed는 연결 원자 한 개의 XYZ도 해제한다. 이 원자가 MCS에 속할 수 있지만 두 모드 모두 MCS의 원자·결합·지정 입체화학은 보호한다. MCS 전체와 고정 영역의 좌표 이동을 별도로 보고한다.
- 기존 회귀 Router의 3개 MLP ensemble, 원본 유지, QED 규칙, 균등 무작위 정책을 비교한다. 후보를 FM으로 생성하기 전에 선택을 저장하며 결과가 나쁜 후보를 사후 메뉴에서 제외하지 않는다.
- UFF-only는 동일한 편집 그래프, ETKDG 초기 좌표, pocket, 고정 mask를 쓰고 FM을 생략한다. raw/FM+UFF/ETKDG+UFF 결과를 각각 저장한다.
- PB는 기존 DIRMO dock20 계약, Vina는 `score_only`, 고정된 6Å pocket 및 40Å box다. 공식 MolGenBench 전체 평가 프로토콜과 다르다.

## 정보 분리와 학습 쌍

기존 ChEMBL MMP 집계표는 원래 분자/assay provenance가 없어 평가 타깃 제외를 보증할 수 없다. 새 후보용 규칙은 평가 타깃을 제외한 개발용 MolGenBench 구조에서 추출하고, 다른 타깃에 나타나는 평가 분자도 제외한다. 활성값·활성 방향·성공률로 규칙을 고르지 않는다. 이미 학습된 Router의 과거 MMP 노출과 FM 사전학습 중복은 **미확인**이다.

공개 활성 SDF에 원래 assay ID가 없으면 series ID로 대신하지 않는다. 로컬 ChEMBL37에서 분자 InChIKey·타깃·endpoint·nM 값이 일치하고, relation `=`, 단일 단백질, confidence 9, 비변이 assay인 측정을 별도 근거로 찾는다. 같은 assay와 endpoint 내 작은 MMP 변화만 `activity_pairs.jsonl`에 기록한다. 이는 ChEMBL33 원본 추출을 복원했다는 뜻이 아니다. Ki/Kd/IC50/EC50은 섞지 않는다.

이 쌍은 화학 편집의 활성 변화 학습 후보이며 FM 모드/pose의 정답이 아니다. 실제 학습 전에 개발용 타깃을 다시 train/validation으로 분할하고 유사 단백질·scaffold 중복을 점검해야 한다.

## 실행

데이터는 `/LocalData1/uslee/datasets/molgenbench-v4`, 별도 평가 환경은 `/LocalData1/uslee/envs/molgenbench-eval`에 둔다. 기존 `dirmo` 환경은 FM 및 기존 PB/Vina 계약을 실행하며 변경하지 않는다. 별도 환경에는 `requirements-molgenbench.txt`를 설치한다.

```bash
cd /home/uslee/code/DIRMO-3D
python scripts/fetch_molgenbench.py --output /LocalData1/uslee/datasets/molgenbench-v4

/home/uslee/miniconda3/envs/dirmo/bin/python scripts/run_molgenbench_study.py \
  --output runs/molgenbench_sbmo_gpu3_20261105_v1 --gpu 3 --workers 6
```

공식 코드 저장소는 `/LocalData1/uslee/datasets/molgenbench-v4/official-code`에서 위 commit으로 고정한다. 다운로드는 `sources.json`이 있어야 완료다. 이미 실행 중인 workflow를 중복 실행하지 않는다. 정상 재개는 같은 명령이며, 코드 해시가 바뀌면 검토 없이 기존 결과와 혼합하지 않는다.

## 파일과 검증

| 파일 | 역할 |
|---|---|
| `scripts/fetch_molgenbench.py` | 공식 ZIP 범위 다운로드·무결성 기록 |
| `scripts/molgenbench_data.py` | 중복 점검, 평가 타깃 잠금, 개발 규칙, assay 근거 및 학습 쌍 |
| `scripts/molgenbench_study.py` | 초기 ligand/scaffold 변환, 3-seed 입력, 기존 Router 점수와 선택 고정 |
| `scripts/run_molgenbench_study.py` | CPU 점검→GPU 생성→평가→보고서의 재개 가능한 단계 실행 |
| `scripts/molgenbench_activity.py` | 별도 환경에서 공식 정규화 방식으로 알려진 활성 분자 재발견 평가 |
| `scripts/molgenbench_report.py` | 타깃 단위 통계, 원본/생성/UFF 비교, 실패·결측 포함 보고 |
| `tests/test_molgenbench.py` | MCS 보호, 실패/결측, fragment 크기, 기존 Router 실제 특징·예측 재현 |

진행 상태는 `workflow_status.json`, 생성/평가 상세는 `status.json`, `generation_status.json`, `evaluation_status.json`에 기록한다. 시각은 KST(+09:00)다. 각 단계 오류는 `audit.log`, `evidence.log`, `prepare.log`, `generate_evaluate_report.log`와 개별 평가 로그에서 확인한다.

산출물은 `DATA_AUDIT.md`, `REPORT.md`, `external_results.json`, `per_target.csv`, `core_coordinate_drift.json`, `activity_matches.json`, `activity_pairs.jsonl`, `pair_examples.json`이다. 생성 실패와 미측정 metric을 구별한다. 알려진 활성 목록과 불일치는 비활성이 아니라 미확인이며, 원본을 그대로 반환한 자기 일치는 편집 성공에서 제외한다. 타깃별 3개 seed 평균을 비교하며 가장 좋은 seed만 고르지 않는다.
