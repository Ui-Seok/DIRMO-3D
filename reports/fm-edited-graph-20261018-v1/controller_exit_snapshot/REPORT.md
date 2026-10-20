# 화학 graph 편집 후 FLOWR.root 생성: 100 pockets

상태: failed. GPU 3; base checkpoint; 64-step.

입력 준비 100/100, GPU 완료 pocket 43/100, FM 결과 슬롯 2816/6400, CPU 평가 완료 24/100.

원본 graph와 terminal C→N/O/F 세 편집 graph를 비교한다. 모든 편집은 원자 수·결합·atom map을 유지하며 지표 순위로 고르지 않았다. 두 FM 조건 모두 편집 후 전체 graph를 고정한다. whole은 전체 좌표 자유, core는 원본 core 좌표 고정이다.

초기 좌표는 graph-only ETKDG 후 고정 core의 국소 anchor에 배치했다. 원본 자유 영역 좌표를 초기화에 사용하지 않는다. 두 FM 조건은 같은 시작 SDF를 공유하므로, whole도 초기 상태의 core 위치 정보를 받는다. 완전한 blind docking 실험이 아니다.

PB는 같은 고정 protein-only pocket의 dock 전체 20 검사다. 성공은 PB+graph/stereo 보존이며 core 조건은 보호 좌표 최대 변위 ≤0.001 Å도 요구한다. 새 편집 분자에는 실험 정답 pose가 없어 native free RMSD를 정확도로 사용하지 않는다.

Vina는 별도 standard-aa receptor의 score-only 대리 점수다. 수용체 준비·PB 실패/누락은 별도로 유지하고 전체 예정 분모에서 제거하지 않는다. 각 graph의 입력/whole-UFF/core-UFF 대조군도 평가한다.

| Graph | 방법 | 평가/예정 | PB 통과 | Graph 보존 | 조건 포함 성공 | 8개 중 성공 pocket | Vina 실측 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C_to_F | fm_core | 198/800 | 117 | 171 | 111 | 23 | 119 |
| C_to_F | fm_whole | 199/800 | 151 | 173 | 142 | 24 | 120 |
| C_to_F | prior | 199/800 | 71 | 189 | 71 | 19 | 120 |
| C_to_F | uff_core | 198/800 | 99 | 181 | 95 | 20 | 119 |
| C_to_F | uff_whole | 199/800 | 55 | 175 | 53 | 14 | 120 |
| C_to_N | fm_core | 227/800 | 141 | 185 | 127 | 26 | 135 |
| C_to_N | fm_whole | 227/800 | 163 | 186 | 151 | 27 | 135 |
| C_to_N | prior | 228/800 | 87 | 210 | 87 | 22 | 136 |
| C_to_N | uff_core | 227/800 | 111 | 201 | 105 | 23 | 135 |
| C_to_N | uff_whole | 227/800 | 59 | 194 | 55 | 15 | 135 |
| C_to_O | fm_core | 224/800 | 123 | 187 | 119 | 25 | 133 |
| C_to_O | fm_whole | 224/800 | 159 | 186 | 148 | 25 | 133 |
| C_to_O | prior | 224/800 | 82 | 208 | 82 | 19 | 133 |
| C_to_O | uff_core | 224/800 | 105 | 203 | 104 | 20 | 133 |
| C_to_O | uff_whole | 224/800 | 57 | 195 | 56 | 17 | 133 |
| original | fm_core | 234/800 | 145 | 206 | 140 | 25 | 143 |
| original | fm_whole | 235/800 | 173 | 201 | 163 | 27 | 143 |
| original | native | 30/100 | 27 | 30 | 27 | 27 | 19 |
| original | prior | 235/800 | 95 | 221 | 95 | 23 | 143 |
| original | uff_core | 234/800 | 116 | 217 | 116 | 23 | 143 |
| original | uff_whole | 234/800 | 57 | 210 | 57 | 18 | 143 |

실행 중 집계는 부분 결과다. 입력 초기화·생성 실패도 전체 예정 FM 슬롯에 남긴다. 사전 contact 기준은 공유결합 확인 자료가 아니라 매우 가까운 native contact를 제외하는 보수적 적격성 기준이다. 원본 PB 통과 여부로 cohort를 바꾸지 않았다. 사전학습 중복은 미확인이다.

코드·조건: study.json, cases/*/case.json. 초기 SDF: cases/*/inputs. FM raw: draws. PB/Vina: evaluations/*/result.json. 로그: gpu.log, preparation/*.log, evaluations/*/worker.log. 실행 재개는 같은 study.json과 저장된 결과를 유지한다.
