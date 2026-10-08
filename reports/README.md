# 검증 결과 — 2026-10-08

사용자의 후속 지시에 따라 LLM/FM 모델을 로드하거나 GPU 추론을 실행하지 않았다. 아래는 코드 검토용 CPU 검사이며 최적화 성능 결과가 아니다.

- `unit-tests.log`: **28개 CPU unit test 통과**. GPU mask/physical mapping, graph/stereo/mapping, invalid edit와 보호 조건, 실패 비용/예산, missing metric, best-history/no-improvement, failed edited graph recovery, grouped split/calibration, MLP artifact round-trip/kernel 분리, resume hash 및 pending action을 검증한다. 모델 호출은 명시적인 unit fixture만 사용했다.
- `unit-tests.json`: 실제 test command의 interpreter/exit status/elapsed time. Synthetic branch data와 학습 artifact는 unit test의 temporary directory에서만 사용하고 연구 artifact로 남기지 않았다.
- `doctor.json`: 실제 로컬 모델 경로와 환경 정보. FM checkpoint·Python 3.12 worker 환경은 missing이다. 샌드박스 안에서 NVIDIA driver 접근이 실패한 사실은 물리 GPU가 없다는 의미가 아니다.
- `tokenizer-audit.json`, `planner-prompt.json`: Qwen 9B의 실제 로컬 tokenizer만 로드해 non-thinking template와 예제 prompt 크기를 검사했다. 해당 prompt는 5,595 tokens이고 1,024 output cap을 포함해 8,192 context cap 이내였다. Model weights는 로드하지 않았다.
- `cpu-validation-final/crossdock_14gs_20gs/`: 실제 로컬 초기 SDF/pocket/PDBQT의 CPU 평가. 첫 CLI와 clean-boundary resume 모두 exit code 0이었다.
- `cpu-initial.stdout.json`, `cpu-initial.stderr.log`: 최신 CPU initial evaluation의 stdout/stderr. Upstream RDKit UFF sulfur typing warning은 보존했으며 해당 PoseBusters internal-energy check의 실제 결과도 JSON에 기록했다.
- `archive/`: 개발 중의 이전 CPU 결과. PoseBusters output-name 수정 전 결과를 포함하며 최신 검증 결과로 사용하지 않는다.

최신 초기 입력의 측정값:

| 항목 | 측정값 |
| --- | --- |
| QED | 0.4316851347607166 |
| SA | 2.617427920420134 (RDKit contrib; 낮을수록 좋음) |
| MW | 320.326 |
| logP | 2.2089 |
| Vina | -5.942 kcal/mol, CPU score-only |
| 자체 geometry screen | pass; protein overlap 0 |
| 최소 contact margin | 0.6412554894494442 Å, DIRMO 정의 |
| PoseBusters | dock preset 적용 check 20개 pass |
| hard protection | pass; 예제 필수 조건은 비어 있음 |
| initial feasible | true |
| objective target met | false |
| 평가 비용 | evaluation bundle 1 |
| LLM calls/tokens | 0 / 0 |
| FM draws/NFE | 0 / 0 |
| GPU peak VRAM | 미측정 |

초기 graph ID는 `ffc0c69a96c19c941577e7aeac76ceaa0d73cedd7c6872348c983fc424f0be79`이다. 실제 FM graph 보존 검사는 checkpoint가 없고 모델 실행도 요청 범위에서 제외되어 수행하지 않았다. Rule routing과 learned-router 학습/선택 경로는 코드와 CPU unit tests로 검사했으며 실제 branch 데이터로 학습된 router는 없다.
