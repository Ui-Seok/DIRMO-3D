# Conda 설치 및 라이브러리

두 Conda 환경으로 구성한다. 사용자가 만든 Python 3.11 `dirmo-llm`는 LLM 서버에 사용하고, Python 3.12 `dirmo`에는 CPU orchestrator·분자 평가·router 학습·FLOWR.root 추론을 통합한다. 설치 파일은 `requirements-llm.txt`와 `requirements.txt` 두 개이며, 각 파일은 다른 requirements 파일을 참조하지 않는다.

이번 변경에서는 Conda 환경 생성·변경, 패키지 설치, 테스트, 모델 로드·추론을 실행하지 않았다. 아래 명령은 사용자가 실행할 설치 안내다.

## 현재 드라이버와 보고된 오류

2026-10-08 읽기 전용 확인에서 NVIDIA 드라이버는 550.144.03이었다. 설치돼 있던 `dirmo-llm`의 Torch는 2.14.1+cu130, `dirmo`의 Torch는 2.11.0+cu130이었다. 사용자가 보고한 `found version 12040`은 CUDA driver API 12.4를 뜻하며 NVIDIA 드라이버 release 번호는 아니다. CUDA 13.x는 최소 580 계열 드라이버가 필요하므로 이 조합은 현재 서버에서 초기화할 수 없다([NVIDIA 공식 호환성 표](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)).

처음 LLM requirements의 `torch>=2.5,<3`는 드라이버 정보를 사용하지 않는 pip가 CUDA 13 build를 선택할 수 있게 했으므로 정확한 cu124 pin으로 수정했다. Training/inference도 기존 cu130 pin을 cu126으로 변경했다. 서버 드라이버와 기존 Conda 환경은 이 작업에서 수정하지 않았다. 시스템 CUDA Toolkit을 별도로 설치하는 것으로 커널 드라이버 문제가 해결되지는 않는다.

## 호환성 판단

| 항목 | 정적 확인 결과 | 적용 |
| --- | --- | --- |
| Python | 고정한 FLOWR.root `pyproject.toml`은 `==3.12.*`이며 DIRMO backend도 3.12를 요구한다 | Python 3.11 `dirmo-llm`는 LLM에 사용; FLOWR.root는 3.12 환경 필요 |
| PyTorch | 로컬 Transformers 5.15는 `torch>=2.5`를 허용하고 FLOWR.root는 `torch==2.11.*`를 요구한다 | 범위가 겹치므로 확정된 Torch 버전 충돌은 없음; 두 환경에서 별도로 해결 |
| TensorDict / Biotite | FLOWR.root는 Torch 2.11 ABI에 맞는 tensordict 0.12와 biotite 1.x를 요구한다 | upstream lock의 0.12.4 / 1.7.1을 통합 requirements에 지정 |
| PoseBusters | FLOWR.root package가 수정된 `posebusters` namespace를 함께 설치한다 | 통합 환경에서는 이 fork만 사용하고 PyPI 배포판을 추가하지 않음 |
| Qwen chat template | Transformers 5.15에서 Jinja2는 `chat-template` extra다 | LLM requirements에 `transformers[chat-template]`를 지정 |

Python 3.12로 전체 환경을 통합하는 것은 metadata의 요구 범위상 가능할 수 있다. 실제 통합 조합의 runtime 호환성은 검증하지 않았다. 이번 구성은 현재 `dirmo-llm`의 Python 3.11을 유지하고, LLM 의존성 변경과 FLOWR.root의 ABI·구조 처리 stack을 각각 관리하기 위한 두 환경이다. 두 환경을 쓴다고 모든 dependency 충돌이 해결됐다고 단정하지 않는다.

## 파일과 라이브러리

| 파일 | 환경 | 포함 기능 |
| --- | --- | --- |
| `requirements-llm.txt` | `dirmo-llm`, Python 3.11 | DIRMO 공통 runtime, Qwen3.5/ChemDFM-v2 LLM 서버, Transformers/Jinja2, Accelerate, PyTorch |
| `requirements.txt` | `dirmo`, Python 3.12 | DIRMO runtime, CPU 평가, router 학습, 고정한 로컬 FLOWR.root 전체 native package와 의존성 |

통합 파일은 Pydantic, NumPy, SciPy, pandas, scikit-learn, RDKit, Biopython, Vina, Meeko, Gemmi, ProLIF, MDAnalysis를 선언한다. `-e ../flowr_root`가 Lightning, torch-geometric, torchmetrics, Open Babel, OpenMM, Hydride, LMDB, OmegaConf 등 나머지 native 의존성과 vendored PoseBusters를 설치한다. 전체 목록은 [FLOWR.root pyproject.toml](../../flowr_root/pyproject.toml)에 있다.

`requirements.txt`의 의미가 기존 CPU 전용에서 training/inference 통합으로 바뀌었다. 기존 `requirements-base.txt`, `requirements-training.txt`, `requirements-fm.txt`는 두 독립 파일로 통합했다. 모든 설치는 `/home/uslee/code/DIRMO-3D`에서 실행한다. 두 파일의 `-e .`는 DIRMO package와 CLI를 설치한다.

## LLM 서버 — dirmo-llm 환경

```bash
conda activate dirmo-llm
cd /home/uslee/code/DIRMO-3D
python -m pip install -r requirements-llm.txt
export DIRMO_LLM_PYTHON="$CONDA_PREFIX/bin/python"
```

기존 `dirmo-llm`에 pip가 없다면 먼저 `conda install -n dirmo-llm pip`를 실행한다. 현재 CPU tokenizer/loading class를 검토한 Transformers 5.15.0 API를 고정한다. Transformers가 tokenizers, safetensors, huggingface-hub를 자동 설치하고 `chat-template` extra가 Jinja2를 설치한다. 서버 드라이버 550.144.03에 맞춰 `torch==2.6.0+cu124`와 공식 cu124 wheel index를 지정한다([PyTorch 공식 버전별 설치 안내](https://pytorch.org/get-started/previous-versions/#v260)). 이 버전은 로컬 Transformers 5.15의 `torch>=2.5` 요구 범위를 만족한다. 기존 CUDA 13 Torch도 위 requirements 설치 명령으로 교체할 수 있다. 이 조합의 실제 Qwen 로드·생성은 이번 변경에서 실행하지 않았다.

기본 Qwen weight는 `/LocalData1/uslee/checkpoints/Qwen3.5-9B`다. 현재 adapter는 `Qwen3_5ForConditionalGeneration`으로 텍스트 요청을 처리한다([공식 문서](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5)). BF16/FP16·SDPA를 사용하며 현재 기본 설정에 bitsandbytes, flash-attn, vLLM, FastAPI가 필요하지 않다. `llm.quantization=int4`를 선택할 경우에만 bitsandbytes를 별도로 설치한다. HTTP server/client는 Python 표준 라이브러리를 사용한다.

## Training / inference — dirmo 환경

```bash
# 이미 환경을 만들었다면 생성 명령은 생략합니다.
# conda create -n dirmo python=3.12 pip
conda activate dirmo
cd /home/uslee/code/DIRMO-3D
python -m pip install -r requirements.txt
export DIRMO_CPU_PYTHON="$CONDA_PREFIX/bin/python"
export DIRMO_FM_PYTHON="$CONDA_PREFIX/bin/python"
```

Torch 2.11 / tensordict 0.12.4 ABI 조합을 유지하며 공식 CUDA 12.6 build인 `torch==2.11.0+cu126`을 선택한다([PyTorch 버전별 설치 안내](https://pytorch.org/get-started/previous-versions/#v2110)). 서버의 550 드라이버는 CUDA 12.x minor-version compatibility 범위에 들어간다. 다만 CUDA 12.6의 새 driver API나 PTX JIT에 의존하는 경로는 더 새 드라이버가 필요할 수 있고, FLOWR.root 실제 동작은 미검증이다([NVIDIA 호환성 안내](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)). `--extra-index-url`과 정확한 CUDA version pin으로 GPU build를 선택한다. pip는 upstream의 `[tool.uv.sources]`를 사용하지 않는다. Python 3.11에서 `--ignore-requires-python`을 사용하거나 upstream metadata/backend의 3.12 검사를 제거하지 않는다.

이 통합 파일은 upstream `uv.lock`의 Torch 2.11 / TensorDict 0.12.4 / Biotite 1.7.1을 유지하되 GPU wheel을 cu130에서 cu126으로 변경한다. 나머지 native 의존성은 upstream metadata 범위로 해결한다. 전체 lock export가 아니며 실제 pip resolution은 아직 실행하지 않았다. Python 버전은 requirements가 변경할 수 없으므로 Conda 환경 생성 단계에서 3.12로 지정해야 한다.

통합 환경의 PoseBusters는 FLOWR.root에 포함된 수정 fork다. 별도의 `pip install posebusters` 또는 `pip install -e '.[evaluation]'`는 같은 namespace에 PyPI 배포판을 추가하므로 실행하지 않는다. DIRMO에서 사용하는 `PoseBusters(config='dock')`, `bust(..., full_report=True)` API는 fork 소스에 존재함을 정적으로 확인했다. 기존 CPU 평가의 PyPI PoseBusters 결과와 새 환경의 fork 결과가 같다고 검증한 것은 아니다.

현재 학습 코드는 router MLP 학습이다. Qwen과 FLOWR.root backbone은 동결하며 해당 backbone의 fine-tuning을 제공하는 것은 아니다. Router 학습과 FM 추론은 같은 Python 환경을 사용하되 별도 프로세스에서 실행한다. 기존 CLI는 FM GPU에서 router를 학습할 때 FM worker를 먼저 종료한다. Learned gate의 추론은 CPU NumPy다.

Checkpoint는 이미 `/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_base_spindr.ckpt`로 연결되어 있다. Vina 평가는 Python score-only이고 receptor PDBQT는 별도로 준비되어 있어야 한다. ProLIF 평가를 켜는 경우에는 수소를 포함해 준비된 protein이 필요하다.

## 실행 환경 연결

두 설치 단계를 같은 shell에서 순서대로 실행하면 위의 `DIRMO_*_PYTHON` 변수들이 연결된다. 새 shell에서는 각 환경의 절대 경로를 다시 지정한다. 현재 서버의 Conda 기본 위치를 사용하면 다음과 같다.

```bash
conda activate dirmo
export DIRMO_LLM_PYTHON=/home/uslee/miniconda3/envs/dirmo-llm/bin/python
export DIRMO_CPU_PYTHON=/home/uslee/miniconda3/envs/dirmo/bin/python
export DIRMO_FM_PYTHON=/home/uslee/miniconda3/envs/dirmo/bin/python
```

`scripts/dirmo`는 `DIRMO_CPU_PYTHON`을 사용한다. LLM/FM interpreter는 환경변수 또는 `configs/local.json`의 `llm.python`, `fm.python`으로 지정한다. CPU orchestrator와 FM worker가 같은 environment를 공유해도 별도 프로세스이며 CUDA mask도 각각 설정된다. 실제 실행 시 서로 다른 두 GPU를 `DIRMO_LLM_GPU`, `DIRMO_FM_GPU`로 지정한다. `git`과 NVIDIA 드라이버의 `nvidia-smi`도 필요하다.

이 구성은 Conda가 Python 환경을 만들고 pip가 requirements를 설치하는 방식이다. `conda install --file requirements.txt` 대신 활성화된 환경에서 `python -m pip install -r ...`를 사용한다. Conda 패키지는 pip 설치 전에 준비하고, 이후 변경은 해당 환경의 pip로 관리한다([Conda 공식 안내](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#using-pip-in-an-environment)).

## LLM 서버 주소와 접속

LLM worker의 기본 주소는 `http://127.0.0.1:8765`이며, 서버 컴퓨터 내부에서 접근하는 JSON API다. 루트 `/`에 웹 화면을 제공하지 않는다. `GET /health`, `POST /propose`에는 managed worker의 Bearer token이 필요하다. 일반 브라우저 접속은 이 token을 보내지 않으므로 정상 서버에서는 401 응답을 받는다. 토큰은 `worker-runtime/workers.json`에 저장되며 CLI가 자동으로 사용한다.

실행 때 명령 앞에 붙였던 환경변수를 반복하지 않으려면 저장된 실행 설정으로 상태를 확인한다.

```bash
conda activate dirmo-llm
cd /home/uslee/code/DIRMO-3D
python -m dirmo --config worker-runtime/config.json health
```

## 실행 중인 LLM의 응답 확인

아래 점검 스크립트는 Python 표준 라이브러리만 사용한다. 활성화된 Conda 환경에서 서버 컴퓨터의 프로젝트 디렉터리로 이동해 실행한다. `worker-runtime/workers.json`의 port와 인증 token을 자동으로 읽으며 token을 출력하지 않는다. 저장된 manifest의 worker instance도 확인한다. 다른 runtime 디렉터리를 사용했다면 `--runtime-dir /절대/경로`를 지정한다.

```bash
conda activate dirmo-llm
cd /home/uslee/code/DIRMO-3D

# 서버 상태만 확인; 추론하지 않는다.
python scripts/check_llm.py --health-only

# 실행 중인 모델에 생성 요청을 한 번 보내고 응답을 출력한다.
python scripts/check_llm.py

# 직접 작성한 질문으로 확인한다.
python scripts/check_llm.py --prompt 'Explain the difference between SMILES CCO and CC(=O)O in two sentences.'
```

`--health-only`는 모델이 로드되어 있고 HTTP 상태 응답이 오는지 확인한다. 기본 실행은 system/user 메시지로 `{"ok": true}` 출력을 요청하고 `raw`, 토큰 수, 지연 시간, 메모리 사용량을 출력한다. `generation_ok: true`는 정상 status와 비어 있지 않은 생성 응답을 뜻하며, `json_valid`와 `json_matches_expected`는 별도로 JSON 형식과 요청 준수를 확인한다. 종료 코드 0은 상태 확인 또는 텍스트 생성 성공을 의미하며 JSON 형식 준수까지 보장하지 않는다. 샘플 한 번으로 DIRMO proposal schema나 화학적 정확성이 검증되는 것은 아니다.

이 스크립트는 모델을 새로 로드하거나 서버를 재시작하지 않는다. 실제 생성 요청은 사용자가 기본 실행 또는 `--prompt` 명령을 실행할 때 발생한다. 요청은 CLI의 실험 budget ledger 밖에서 실행되지만 서버 GPU를 사용하며 다른 추론과 같은 lock을 기다린다. 추론 timeout은 기본 600초이고 `--timeout 120` 등으로 바꿀 수 있다. 클라이언트 timeout만으로 이미 시작한 서버 추론이 취소되지는 않는다. HTTP 401은 token 불일치, connection refused는 서버 미실행/주소 문제, timeout은 응답 대기, `status: oom`은 GPU 메모리 부족을 나타낸다. 서버 오류의 상세 원인은 `worker-runtime/llm.log`도 확인한다.

현재 shell에 `DIRMO_*` 환경변수가 남아 있으면 저장된 설정에도 해당 값이 우선 적용되므로, 시작 시와 같은 값을 사용해야 한다.

개인 PC에서 서버의 localhost에 접근하려면 개인 PC의 터미널에서 SSH tunnel을 유지한다. `서버주소`는 실제 SSH 접속 주소로 바꾼다. 이 tunnel은 네트워크 연결만 전달하며 API 인증을 대신하지 않는다.

```bash
ssh -N -L 8765:127.0.0.1:8765 uslee@서버주소
```

2026-10-08 접속 진단에서 포트가 LISTEN 상태인 동안 루트와 인증된 `/health` 요청이 모두 timeout인 것을 관찰했다. 기존 단일 `HTTPServer`는 브라우저가 미리 열어 둔 소켓 때문에 다른 요청도 대기할 수 있다. 실제 당시 대기 중이던 소켓의 원인은 확정하지 않았으며, 이 알려진 대기 문제를 막기 위해 `ThreadingHTTPServer`와 socket I/O timeout을 적용했다([Python 공식 문서](https://docs.python.org/3.11/library/http.server.html#http.server.ThreadingHTTPServer)). 모델 호출과 shutdown은 lock으로 직렬화하며, health 요청은 모델 요청 lock을 기다리지 않는다.

이미 실행된 worker에는 코드 변경이 반영되지 않는다. 저장된 설정과 GPU 선택을 유지하여 다음과 같이 재시작한다. 아래 명령은 이번 수정에서 실행하지 않았으며 새 worker의 실제 로드·응답은 미검증이다.

```bash
python -m dirmo --config worker-runtime/config.json stop-workers --only llm
python -m dirmo --config worker-runtime/config.json start-workers --only llm --timeout 600
python -m dirmo --config worker-runtime/config.json health
```
