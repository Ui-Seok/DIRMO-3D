# FLOWR.root checkpoint metadata 감사

작성일: 2026-10-13.

두 파일의 SHA256을 계산하고, PyTorch를 사용해 `map_location='cpu'`로 원본 checkpoint를 읽었다. GPU 모델 생성·추론·학습은 수행하지 않았다. 원본 `hyper_parameters`와 저장 시점 epoch/global_step은 [checkpoints.json](checkpoints.json)에 있다.

| 파일 | SHA256 | 저장된 학습 dataset | affinity head |
| --- | --- | --- | --- |
| `/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_base_spindr.ckpt` | `78ef9ebe4459df1ce2668685785602a37288594db83d2bb5d970c8bc06e1ce9d` | `spindr` | 없음 (`predict_affinity=False`) |
| `/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_v2.ckpt` | `f28e863b2b208718f3d3f85f09837c2f907a71123436db6f98a25d2f1858b6a0` | `combined_prot-lig` | 있음 (`predict_affinity=True`) |

두 파일 모두 fragment inpainting 및 legacy scaffold/functional-group inpainting이 활성화된 학습 설정을 저장하고 있다. base에는 fragment growing과 interaction conditional, v2에는 linker inpainting도 활성화된 설정이 있다. 모드별 학습량·생성 품질은 metadata만으로 확정할 수 없다.

두 파일 모두 `graph_inpainting` 키는 미저장이다. 확인한 로컬 upstream의 선택적 hparams 저장 목록에서도 이 항목이 빠져 있으므로, 키 부재를 whole-graph 미학습의 증거로 사용하지 않는다. 현재 DIRMO adapter가 기록하는 `untrained` 문자열은 이 존재 여부를 사용한 추정이며 이력 확정이 아니다. 이번 작업에서는 해당 실행 코드나 과거 결과를 변경하지 않았다.

확인한 upstream revision은 `739a33c2aab74d074dd24605b9a2e34dc809ae9c`다. 현재 코드와 저장된 config를 읽은 감사이며 원래 checkpoint를 만든 코드 버전·전체 학습 이력을 복원한 것은 아니다.

학습 방식, 조건 종류, 대조군, 평가 지표와 제안 생성 예산은 [실험 계획](../../docs/fm_conditioning_protocol.md)에 정리했다. 이 계획의 새 GPU 실험은 아직 실행하지 않았다.
