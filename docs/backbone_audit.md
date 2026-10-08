# Backbone audit — 2026-10-08

초기 사용자 후속 요청은 FLOWR.root checkpoint를 추후 내려받고 코드 작성과 검토만 진행하는 것이었다. 이후 `/LocalData1/uslee/checkpoints/FLOWR_root`에 checkpoint를 제공하여 `configs/local.json`의 경로를 연결했다. **이 추가 변경에서는 테스트를 실행하지 않았으며, 실제 LLM/FM 모델 로드, GPU 추론, checkpoint compatibility 검사, VRAM 측정도 수행하지 않았다.** 초기 구현의 CPU initial evaluation과 unit tests는 별도 기록한다.

## 실제 로컬 자원

| 자원 | 확인한 위치 / 상태 |
| --- | --- |
| DIRMO-3D | `/home/uslee/code/DIRMO-3D`; 시작 시 빈 디렉터리, git 저장소 아님 |
| FLOWR.root | `/home/uslee/code/flowr_root`; clean working tree, revision `739a33c2aab74d074dd24605b9a2e34dc809ae9c` |
| LiFT | `/home/uslee/code/LiFT`; revision `7a284d4a542f5437c30b95d0d3402b4b0cd9635a` |
| Qwen 9B | `/LocalData1/uslee/checkpoints/Qwen3.5-9B`; 실제 safetensors shards 존재, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Qwen 4B | `/LocalData1/uslee/checkpoints/Qwen3.5-4B`; 실제 shards 존재, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| ChemDFM v2 | `/LocalData1/uslee/checkpoints/ChemDFM-v2.0-14B`; 실제 shards 존재, revision `b3b1d2143bb3e307e6b6dae794f6e9b0a83a45ef` |
| FLOWR.root checkpoint | `/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_base_spindr.ckpt`; 645,981,475 bytes, 기본 config에 연결, 모델 로드 미실행 |
| 구형 FLOWR checkpoint | `/LocalData1/uslee/checkpoints/flowr/{flowr_noHs.ckpt,flowr_withHs.ckpt}`; root checkpoint로 대체하지 않음 |
| 평가 CPU Python | `/home/uslee/miniconda3/envs/flowr/bin/python`; Python 3.11.16, RDKit 2023.9.5, PB 0.3.1, Vina 1.2.6, Meeko 0.8.0, ProLIF 2.0.3 |
| LLM Python 후보 | `/home/uslee/miniconda3/envs/ai_scientist/bin/python`; Python 3.11.15, Torch 2.5.1, Transformers 5.15.0; Qwen loading class import 성공 |
| FLOWR.root Python | 별도 Python 3.12 환경 없음; 기존 `flowr` Python 3.11 환경과 합치지 않음 |
| 입력 사례 | 로컬 `processed_crossdock_noH_full_temp/test/14gs-A-rec-20gs-cbd-lig-tt-min-0-*`의 pocket/SDF/PDBQT |
| receptor 준비 provenance | 기존 PDBQT는 있으나 preparation 상세는 제공되지 않음; metadata에 미확인으로 기록 |
| SMI-TED | 첫 prototype에서는 사용하지 않음; LiFT를 실행하지 않아 availability를 검증하지 않음 |

AGENTS.md는 현재 home/code 및 지정 데이터 경로 검색에서 발견되지 않았다. 기존 다른 프로젝트와 upstream, 모델·데이터 파일을 수정하지 않았다. 신규 환경이나 모델을 다운로드하지 않았다.

초기 `nvidia-smi` read-only 조사에서 RTX A6000 48GB가 7장 있었다. Physical 0의 UUID는 `GPU-26524290-8c72-5bed-e6e9-edd900103a6e`, physical 1은 `GPU-05efc5f4-75b1-0cc7-9883-99e113306a33`였다. 당시 사용량은 각각 20,765/42,419 MiB로, 0/1에 모델 두 개를 바로 올리기에는 여유가 충분하지 않았다. **사용할 두 장은 지정받지 않았고 어떠한 모델 작업도 배치하지 않았다.** 기본 0/1은 config 기본값이며 승인된 실제 실행 매핑을 뜻하지 않는다. 메모리 사용량은 시점에 따라 달라진다. 샌드박스 내 `doctor`의 driver 접근 실패는 물리 GPU 부재와 구분해야 한다.

## FLOWR.root 경로 정적 확인

공식 소스: [FLOWR.root](https://github.com/jule-c/flowr_root/tree/739a33c2aab74d074dd24605b9a2e34dc809ae9c). 로컬 LICENSE는 MIT이다. README의 generation-only checkpoint 이름은 `flowr_root_spindr_base.ckpt`이다. 사용자가 제공한 실제 파일명은 `flowr_root_base_spindr.ckpt`이며, 이름을 바꾸지 않고 해당 파일 경로를 설정에 연결했다. 파일 존재와 크기만 확인했으며 hyperparameters·SHA256·kernel ID는 현재 **미확인/미생성**이다. 파일명만으로 공식 checkpoint와의 동일성이나 호환성을 확정하지 않는다.

확인한 파일과 처리:

1. `flowr/gen/generate_from_pdb.py`: `get_args()`의 native 설정과 whole-graph `graph_inpainting` 선택. 기본 validity/uniqueness/반복 sampling loop는 adapter에서 사용하지 않는다.
2. `flowr/gen/utils.py`: `load_util()` → `ComplexInterpolant`, `load_data_from_pdb()` → `process_complex()`/H 및 aromaticity 처리, `get_dataloader()` → seeded transform. DIRMO는 미리 고정한 pocket에 `cut_pocket=False`를 전달한다.
3. `flowr/data/interpolate.py`: `_build_inference_prior()`의 graph 경로는 전체 atom mask와 `fragment_mode='graph'`를 사용한다. Random prior의 `inpaint_graph()`는 좌표를 랜덤으로 만들고 atomics/charges/bonds를 reference에서 가져온다. Fragment coordinate inpainting과 다르다.
4. `flowr/models/mol_builder.py`: `inpaint_graph()` 기본 feature keys는 discrete features이고 좌표는 제외한다. Build repair가 실제로 graph를 바꿀 수 있으므로 `--no_ligand_valence_repair`를 적용한다.
5. `flowr/models/fm_pocket.py`: Euler에서 graph features를 덮어쓴다. **최종 corrector의 predicted graph는 `final_inpaint`가 꺼져 있으면 덮어쓰지 않는다.** 따라서 DIRMO는 `final_inpaint=True`를 사용하며 `model.feature_keys`에 coords가 들어있으면 거부한다. 출력은 fixed pocket COM을 되돌린다.
6. 같은 `_generate()` 경로에서 `midpoint`는 graph inpainting일 때 명시적으로 오류를 낸다. Euler만 허용한다. `integration_steps` 변경은 request와 resident integrator 양쪽에 적용한다.
7. `flowr/scriptutil.py`: checkpoint hparams와 architecture로 한 번 로드한다. Affinity는 DIRMO oracle로 사용하지 않는다. Rigid pocket와 `coord_scale=1.0` 외 checkpoint는 현재 adapter에서 거부한다.

Runtime에는 reference graph tensor round-trip을 먼저 검사하고 stereo는 reference 3D 좌표에서 재확인한다. Native discrete graph 조건이 지정 stereochemistry를 직접 encoding한다고 가정하지 않는다. Whole-graph random/Euler 경로에 진입한 경우에도 raw output은 독립적으로 chemical graph/stereo 검사를 통과해야 한다. Pocket file과 upstream absolute pocket tensor의 hash를 기록해 편집마다 pocket이 바뀌는지 검사한다. Raw slot/build failure/actual forward NFE를 기록하며 filtered 결과 수로 raw draws를 숨기지 않는다.

아직 검증하지 못한 부분: checkpoint에 대한 load_model kwargs/hparams 호환성, 해당 checkpoint의 hybridization/charge vocabulary, 지원 원소와 stereo tensor round-trip, graph별 실제 build yield, 서로 다른 seed, edited graph의 실제 pose 생성과 pocket-frame 보존, OOM과 실제 peak VRAM. Source interface를 읽은 것만으로 모델 실행이 성공한다고 단정하지 않는다. 기능 실패 시 capability error/graph mismatch로 기록하며 다른 모델·docking·embedding으로 대체하지 않는다.

해당 upstream `pyproject.toml`/`uv.lock`는 Python 3.12, Torch 2.11, tensordict 0.12 계열 및 CUDA 13 GPU index를 명시한다. 기존 Python 3.11/Torch 2.5.1 환경은 이 lock과 다르다. 설치 시 A6000 서버 드라이버와 CUDA wheel을 실제 확인해야 한다. Checkpoint 로드 경로가 사용하는 PyTorch serialization 호환성도 아직 검증하지 않았다.

## LLM 및 비교 모델

[Qwen3.5-9B 공식 card](https://huggingface.co/Qwen/Qwen3.5-9B), 로컬 config와 tokenizer template를 확인했다. Qwen 4B/9B architecture는 `Qwen3_5ForConditionalGeneration`이다. CPU tokenizer에서 `enable_thinking=False`가 닫힌 `<think>` prefix를 생성함을 확인했다. 측정 진단으로 구성한 예제 prompt는 로컬 tokenizer로 길이를 검사했다. 실제 16-bit inference, generation token/latency/VRAM은 미측정이다. 자동 device map/TP/FP8/custom attention kernel을 사용하지 않는다.

[ChemDFM-v2.0-14B 공식 card](https://huggingface.co/OpenDFM/ChemDFM-v2.0-14B)와 로컬 README는 Qwen2.5 기반 non-thinking 모델, AutoModelForCausalLM/FP16 및 canonical SMILES preprocessing을 안내한다. ChemDFM-R로 취급하지 않는다. License는 card상 AGPL-3.0이다. Qwen artifact의 로컬 LICENSE는 Apache-2.0이다. 모델별 라이선스는 코드 license와 별도로 관리해야 한다.

[LiFT](https://github.com/kasurl/LiFT/tree/7a284d4a542f5437c30b95d0d3402b4b0cd9635a)는 DrugFlow에 frozen SMI-TED semantic embedding을 주는 경로이며 로컬 LICENSE는 MIT이다. 최종 graph identity를 보장하는 기본 생성기로 연결하지 않았다. 현재 LiFT adapter는 명시적으로 disabled comparison placeholder이며 실제 semantic generation을 제공하지 않는다.

## 첫 실제 integration을 위한 최소 조건

- 지정 checkpoint의 실제 경로와 SHA256, 별도 호환 Python 3.12 환경.
- 사용 가능한 서로 다른 A6000 UUID 두 개. GPU 0/1을 자동으로 사용하면 안 됨.
- 알려진 preparation의 고정 pocket/receptor/ligand. 현재 예제는 receptor preparation 상세가 부족함.
- Unedited graph의 두 seed, 최소 한 edited graph, stereo 포함 graph의 tensor round-trip과 raw identity 검사.
- 짧은 rule loop 이후 동일 kernel의 실제 branch dataset. 충분한 target/series가 모일 때만 grouped 일반화 평가.

최초 작은 실행의 `kernel.json`, `graph_check.json`, `pocket_frame.json`, worker health와 실제 비용 기록을 이 문서의 미검증 항목과 함께 확인해야 한다.
