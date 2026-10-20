"""Render the saved-observation analysis as CSV, figures and a Korean report."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "summary.json").read_text())

with (ROOT / "datasets.csv").open("w", newline="") as stream:
    names = ["dataset", "targets", "records", "known_raw_draws", "measured", "feasible",
             "improved_feasible", "success_eligible", "targets_with_eligible_candidate"]
    writer = csv.DictWriter(stream, names)
    writer.writeheader()
    for name, value in data["datasets"].items():
        writer.writerow({"dataset": name, **{key: value["total"][key] for key in names[1:]}})

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for ax, run, title in zip(axes[:2], ("sweep", "regions"), ("Whole-graph sweep", "Region redesign")):
    s = data["datasets"][run]["total"]
    counts = [s["records"], s["measured"], s["feasible"], s["success_eligible"]]
    bars = ax.bar(["Recorded", "Measured", "Feasible", "Goal + gain"], counts,
                  color=["#718096", "#4299e1", "#38a169", "#d69e2e"])
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_title(title)
    ax.set_ylabel("Branch records")
    ax.tick_params(axis="x", rotation=18)
paired = data["sweep"]["steps"]
bars = axes[2].bar(["64 steps", "128 steps"],
                  [100 * paired["feasible_64"] / paired["paired_records"],
                   100 * paired["feasible_128"] / paired["paired_records"]],
                  color=["#4299e1", "#e53e3e"])
axes[2].bar_label(bars, fmt="%.1f%%", padding=3)
axes[2].set_ylim(0, 100)
axes[2].set_ylabel("Feasible / paired records (%)")
axes[2].set_title("Same 93 targets, same 3 seeds")
fig.savefig(ROOT / "outcomes.png", dpi=180)
plt.close(fig)

budget = data["sweep"]["allocation_all_actual_draws_corrected"]["3"]
probe = data["sweep"]["matched_probe"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
labels = ["3 candidates\none draw each", "Rank-1 candidate\nthree draws", "Probe two\nthen repeat best"]
bars = axes[0].bar(labels, [budget["mean"][name] for name in
                           ("breadth", "depth_rank1", "probe2_commit")],
                  color=["#38a169", "#718096", "#4299e1"])
axes[0].bar_label(bars, fmt="%.4f", padding=3)
axes[0].set_ylim(0, 0.046)
axes[0].set_title("Offline allocation replay: 69 paired pockets")
axes[0].set_ylabel("Mean retained feasible utility gain")
for i, arm in enumerate(("pose", "edit")):
    values = [max(probe[arm]["grouped_cv_pre_action_r2"].values()),
              probe[arm]["grouped_cv_calibrated_first_draw_r2"]]
    axes[1].bar([i - 0.18, i + 0.18], values, width=0.34,
                color=["#718096", "#4299e1"])
axes[1].set_xticks([0, 1], ["Pose cells", "Edit cells"])
axes[1].axhline(0, color="black", linewidth=0.7)
axes[1].set_ylim(-0.4, 1.05)
axes[1].set_ylabel("Target-grouped out-of-fold R²")
axes[1].set_title("Before-action features vs calibrated first draw")
axes[1].legend([plt.Rectangle((0, 0), 1, 1, color="#718096"),
                plt.Rectangle((0, 0), 1, 1, color="#4299e1")],
               ["Best pre-action model", "One draw + calibration"], loc="lower right")
fig.savefig(ROOT / "budget_and_prediction.png", dpi=180)
plt.close(fig)

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "|" + "|".join("---" for _ in headers) + "|",
                      *("| " + " | ".join(str(v) for v in row) + " |" for row in rows)])

dataset_rows = []
for name in ("gpu_01", "gpu_02", "gpu_03", "gpu_05", "gpu_06", "v2",
             "smoke", "sweep", "region_smoke", "hard_region", "gate_check", "regions"):
    s = data["datasets"][name]["total"]
    dataset_rows.append([name, s["targets"], s["records"], s["measured"], s["feasible"],
                         s["improved_feasible"], s["success_eligible"]])

arm_rows = []
for label, s in data["clean_sweep_arms"].items():
    arm_rows.append([label, s["records"], s["measured"], s["feasible"],
                    s["improved_feasible"], s["success_eligible"]])
policy_rows = [[r["path"], r["status"], r["cost"]["fm_draws"], r["delta"]]
               for r in data["single_target_results"]]
probe_rows = [[arm, s["cells"], f'{s["first_draw_predicts_next_two_r2"]:.3f}',
               f'{s["first_draw_pearson_r_squared"]:.3f}',
               f'{s["grouped_cv_calibrated_first_draw_r2"]:.3f}',
               f'{s["grouped_cv_pre_action_r2"]["ridge"]:.3f}',
               f'{s["grouped_cv_pre_action_r2"]["boosted"]:.3f}']
              for arm, s in probe.items()]
interval = paired["target_bootstrap_interval"]
sweep_cost = data["costs"]["sweep"]
region_cost = data["costs"]["regions"]
budget_rows = [[name, f'{budget["mean"][name]:.5f}', f'{budget["median"][name]:.5f}']
               for name in ("breadth", "depth_rank1", "probe2_commit")]

report = f"""# 저장된 DIRMO-3D 실험 분석 — 2026-10-12

원본 관측을 집계하고 기존 분석의 지표 정의와 예산 계산을 재검토했다. 유효한 생성·개선 후보는 존재하지만, 저장된 완전 최적화 실행은 단일 target에 한정돼 있고 모두 no_improvement였다. 새 GPU 실험이나 LLM/FLOWR 모델 로드는 수행하지 않았다. 예측 비교에는 저장된 93-feature 데이터를 사용하는 CPU 통계 모델만 학습했다.

**분석 자료와 재현 방법**

- analyze_saved_runs.py: 관측 집계, 직접 좌표 비교, target 분리 CV, 예산 배분의 오프라인 재계산.
- summary.json: 모든 수치와 원본 dataset SHA256.
- datasets.csv: 실험별 관측 집계.
- outcomes.png, budget_and_prediction.png: 비교 그림.
- 재현: dirmo 환경에서 이 폴더의 analyze_saved_runs.py 실행 후 build_report.py 실행.
- 기존 runs 및 기존 reports의 분석 결과는 수정하지 않았다.

관측 수는 branch 기록 수다. 실제 FM raw draw 수는 별도로 계산한다. measured는 utility_change가 있는 기록, feasible은 현재 물리 검사·보호 조건·최대 손실 조건의 통과, improved는 feasible이며 utility 증가량이 설정된 최소값 이상인 기록이다. success_eligible은 improved에 모든 수치 목표 충족을 추가한 값이다. 수집된 성공 가능한 후보와 전체 optimize 실행의 성공을 구분해야 한다. 초기 오류 실험의 비용 불명 RPC 예약값을 실제 draw/NFE로 해석하지 않았다.

**실험 이력**

1. gpu_01/gpu_02: 입력 graph/SDF 불일치와 optional hybridization 초기화 오류를 찾은 실행 연결 점검. 측정 가능한 생성 결과는 없다.
2. gpu_03 및 gpu_04: random prior 오류 수정 후 실제 생성과 학습 경로 점검. gpu_04의 추가 수집은 gpu_03/branches.jsonl에 누적됐다. 5건이 생성·평가됐지만 모두 feasible=false였다.
3. gpu_05: reference conformer prior 도입. 5건 중 4건이 feasible. 단, noise=0인 pose 반복은 서로 다른 seed에서도 동일 SDF 좌표와 utility를 반환했다. 독립 확률 표본 4개로 해석할 수 없다.
4. gpu_06: reference prior에 0.3 Å 표준편차의 노이즈 추가. 6건 중 4건이 feasible. seed 42/43의 pose 차이는 64-step에서 RMSD 0.0732 Å, 128-step에서 0.3858 Å였다. 정렬 없이 atom map으로 대응시켜 고정 pocket 프레임에서 비교했다.
5. v2: 목표와 예산을 바꾼 단일-case 수집·학습·정책 비교. 실제 저장된 config/model SHA는 기존 flowr_root_base_spindr.ckpt이며, 현재 configs/gpu-v2.json의 다른 checkpoint와 혼동하면 안 된다. 모든 관측이 feasible이지만 수치 목표와 개선량을 동시에 충족한 후보는 없다.
6. smoke: 3개 target에서 LLM 편집과 별도로 열거한 QED 증가 편집(oracle)의 수집 경로 점검.
7. sweep: 99개 target에 seed 42/43/44를 적용한 297개 case-run. 각 초기 snapshot에서 pose 64-step, pose 128-step, LLM 편집, 최대 3개 oracle 편집을 분기 실행했다. 같은 target/seed의 대안들이 공통 snapshot에서 시작한다. 관측들은 반복 과정의 연속 trajectory가 아니다.
8. region_smoke/hard_region/gate_check: 영역 mask 전달과 graph/fragment 모드의 호환성을 점검한 중간 실험. 초기의 혼합 행동 실패를 최종 fragment 결과와 합쳐 성능 평균을 내면 안 된다.
9. regions: fragment 모드에서 규칙이 고른 말단 원자/치환기 영역을 재설계. 99개 case 중 97개에서 278개 행동이 실행됐다. 생성기의 atom 수는 고정이며, 자유 영역의 원소/결합이 결과로 결정된다. 별도의 정식 정책 비교는 없다.

저장된 sweep과 regions 실행은 모두 Qwen3.5-9B와 동일한 base FLOWR checkpoint(SHA256 78ef9ebe4459df1ce2668685785602a37288594db83d2bb5d970c8bc06e1ce9d)를 사용한다. 각 실험 내부의 저장된 구현 hash도 하나로 일치한다. ChemDFM, Qwen3.5-4B 또는 새 v2 checkpoint 간 성능 비교를 했다는 근거는 없다.

{table(["실험", "target", "기록", "측정", "feasible", "유효 개선", "성공 가능 후보"], dataset_rows)}

**최종 최적화 결과와 후보 발견을 구분**

저장된 완료 result.json 8개는 전부 원본을 선택했다. 서로 다른 8개 target을 시험한 결과가 아니라 같은 14gs_A 입력의 정책/설정별 반복이다. gpu_06/learned에는 완료 result가 없는 중간 checkpoint도 있다.

{table(["완료 결과 경로", "상태", "FM draws", "최종 utility 증가"], policy_rows)}

작은 router 학습은 gpu_03에서 3개, gpu_06에서 4개, v2에서 4개 관측을 사용했다. 모두 단일 target이라 그룹 검증이 불가능했고, 작은 학습 loss는 일반화나 정책 우월성의 근거가 아니다.

한편 sweep에서 같은 14gs_A의 LLM 편집 후보는 QED 0.5101, SA 2.6914, Vina -6.317, utility 증가 +0.10854로 엄격한 조건을 충족했다. 원본은 QED 0.4317, SA 2.6174, Vina -5.942였다. 해당 후보의 경로는 ../../runs/sweep/branches_s0/14gs-A-rec-20gs-cbd-lig-tt-min-0/generation/6c3240b5c2320828b9235cdc/raw_mapped.sdf 이다. 따라서 이 입력에서 개선이 불가능했다고 해석할 수 없으며, 실제 자동 정책이 그 후보를 찾고 선택했는지가 별도 평가 대상이다.

**99-target sweep**

성과를 판단할 때 초기 목표 상태를 추가로 구분해야 한다. 원본이 이미 수치 목표를 만족한 target은 12개였다. 성공 가능 개선 후보가 나온 7개 target 중 5개는 이 집단이고, 원본이 목표 미달이었는데 새 후보가 모든 목표와 최소 개선량을 충족한 target은 2개뿐이다. Regions에서도 성공 가능 개선 후보가 나온 6개 중 새 목표 도달 target은 1개다. 집단별 목록과 초기 상태를 summary.json의 goal_transitions에 기록했다.

전체 1,748건 중 실제 raw draw는 1,617회, graph 검사를 통과해 utility를 측정한 결과는 1,297건이다. feasible은 728건(전체 기록의 41.6%), 유효 개선은 226건, 모든 목표까지 충족한 성공 가능 후보는 30건이다. 적어도 하나의 유효 개선 후보가 나온 target은 47/99, 성공 가능 후보가 나온 target은 7/99다. 이는 기록된 초기-state 분기 탐색의 발견률이며 optimize 성공률이 아니다.

기존 분석 기준대로 원본 feasible=false, 원본 Vina>0 또는 baseline utility 누락인 6개 target을 제외하면 93 target/1,640건이 남는다. 제외 목록은 summary.json에 기록했다. 이 경우 유효 개선 후보는 196건/44 target이고 성공 가능 후보 30건/7 target은 그대로다.

{table(["행동:후보출처:steps", "기록", "측정", "feasible", "유효 개선", "성공 가능"], arm_rows)}

Oracle은 모든 편집을 실제 검증하고 QED 증가량으로 순위를 매겨 고른 대조군이다. LLM 후보와 같은 후보 분포가 아니므로 두 행의 단순 성공률만으로 모델 우열을 판단하면 안 된다. 다만 현재 수집에서는 oracle이 LLM이 놓친 유효 개선을 추가로 확보했다.

큰 실패 요인은 graph 계약 불일치와 Vina 허용 손실 초과다. 전체 기록에는 graph_mismatch 320건, 사전 capability_error 131건이 있다. 후자 중 109건은 typed graph와 reference SDF의 불일치, 22건은 upstream tensorization 보존 실패였다. graph_mismatch 320건 중 195건은 orchestrator의 재검증에서 실패했다. 이는 단순한 계산량 부족과 구별해 구현/입력 계약부터 점검해야 하는 부분이다. 평가 실패 원인은 한 관측에서 여러 개가 함께 발생할 수 있다.

**64-step와 128-step의 짝 비교**

초기 입력이 유효한 93 target의 동일 seed 279쌍에서 64-step feasible은 202/279(72.4%), 128-step은 188/279(67.4%)다. 128-64 차이는 -5.02%p이고 target 단위 bootstrap 95% 구간은 [{100 * interval[0]:.2f}, {100 * interval[1]:.2f}]%p다. 두 조건 모두 utility가 측정된 224쌍의 평균 utility 차이도 128-step 쪽이 -0.01195로 낮았다.

측정된 생성의 median FM 시간은 64-step 3.19초, 128-step 6.12초다. 이 데이터에서는 step 증가가 품질을 개선하지 않았고 비용은 증가했다. Euler/random/reference의 다른 조합이나 모든 입력에서 같은 결과가 난다는 뜻은 아니다.

Reference 초기화는 이미 주어진 원본 pose 주변에서 생성/복구하는 과제다. Gaussian random prior에서 pose를 새로 만드는 과제와 난이도가 같지 않으므로 작은 실험의 통과율 차이만으로 FM의 일반적인 생성 능력이 향상됐다고 주장할 수 없다. Graph를 유지하는 pose 행동은 graph 기반 QED도 유지한다.

**사전 예측과 한 번 측정한 뒤의 예측**

기존 보고서의 0.527(edit), 0.945(pose)는 seed 42 결과와 다음 두 seed 평균 사이의 Pearson 상관계수 제곱과 일치한다. 이를 실행 전 모델의 샘플 외 예측 R²와 바로 비교하면 평가 정의가 다르다.

이번 재분석은 정확히 같은 3-seed cell에서 seed 42를 관측하고 seeds 43/44의 평균 utility 증가를 예측했다. 5-fold CV에서 target을 분리하고, 첫 측정값의 선형 보정도 훈련 target에서만 학습했다. 모든 세 draw에 utility가 존재하는 cell에 한정한 분석이다.

{table(["행동", "3-seed cell", "첫 값 직접 예측 R²", "상관계수²", "첫 측정 + CV 보정 R²", "사전 Ridge R²", "사전 Boosted R²"], probe_rows)}

실제 첫 관측이 이후 결과 예측에 유용하다는 신호는 있다. 그러나 edit의 첫 값 직접 예측 R²는 probe seed에 따라 0.258/0.474/0.263으로 달랐다. 첫 측정에 쓴 계산 비용을 포함해야 하므로 이 결과만으로 learned routing이나 probe-then-commit 정책의 효율을 입증할 수 없다. 다른 seed를 probe로 택하는 민감도와 세 역할을 합친 분석도 summary.json에 기록했다. 여기서는 특정 93-feature 표현과 두 사전 예측 모델을 시험했으므로, 다른 진단 feature나 다른 학습 방식까지 무효라고 일반화할 수 없다.

**고정 draw 예산의 재계산**

기존 예산 분석의 음수 개선을 선택하는 오류를 수정해 원본 유지의 이득 0을 항상 포함했다. 같은 oracle 후보군에서 그래프 불일치 등 실제 draw 실패도 이득 0으로 포함하고, raw_draws=0의 사전 거부는 FM draw가 아니므로 제외했다. 정책을 모두 비교할 수 있는 공통 69개 pocket, 최대 3-draw 예산의 결과다.

{table(["예산 배분", "평균 유효 utility 이득", "median"], budget_rows)}

Breadth는 후보 3개에 각 1회, depth는 QED 순위 1 후보에 3회, probe2_commit은 후보 2개를 1회씩 관측하고 유효한 최선 후보에 나머지 1회를 쓴다. Breadth-probe의 평균 차이는 +0.00560이고 pocket bootstrap 95% 구간은 [+0.00203, +0.00977]이다. Breadth-depth의 구간은 0을 포함해 명확한 우위를 주장할 수 없다. 2-draw 비교에서도 breadth-depth 구간은 0을 포함한다.

측정 성공만 포함한 기존 보고서와 같은 표본 기준에서도 원본 유지 오류를 고친 뒤 breadth-probe 차이는 양수였다. 이 결과는 초기 snapshot 후보들의 저장된 draw를 재배분한 오프라인 비교다. 새 관측으로 state가 바뀌는 실제 연속 최적화 정책의 성능이나 최종 target 달성률을 측정한 실험은 아니다.

**영역 재설계**

278회 중 graph/영역 계약을 통과한 출력은 212건, utility 측정은 211건, feasible은 121건(전체의 43.5%), 유효 개선은 39건/29 target, 목표와 최소 개선까지 충족한 후보는 8건/6 target이다. 실패 66건 중 43건은 원본 graph를 그대로 반환해 재설계로 인정하지 못한 경우이고, 21건은 raw molecule build 실패, 2건은 disconnected graph였다.

저장된 redesign.json 212개에서 영역 밖 held 원자의 최대 drift는 모두 0 Å였다. 이는 저장된 SDF 정밀도에서 실제 mask 전달이 작동한 관측이며, 모든 다른 입력에서의 보존 보장은 아니다. 실패·미생성 출력에는 해당 검증 기록이 없다.

이 실험은 전체 graph 고정과 다른 계약을 사용하고 seed도 한 개라 whole-graph sweep과 직접 승률 경쟁을 시킬 수 없다. 보호 조건 목록도 비어 있어 사용자가 지정한 중요 pharmacophore/interaction 보존 성능을 검증한 실험이 아니다.

**비용과 우선 점검**

Sweep case 시간 합은 {sweep_cost["summed_case_seconds"] / 3600:.2f}시간이며 LLM 생성 기록의 합이 {sweep_cost["recorded_llm_seconds"] / 3600:.2f}시간이다. LLM 528회/약 430만 token, FM 1,617 draws/124,113 NFE, 평가 bundle 1,594회였다. 모델 로드 시간은 이 case 합에 포함되지 않는다.

Regions case 시간 합은 {region_cost["summed_case_seconds"] / 3600:.2f}시간이고 LLM 기록이 {region_cost["recorded_llm_seconds"] / 3600:.2f}시간이다. 영역 후보는 말단 구조 규칙으로 정해지고 fragment 모드에서는 LLM의 exact-graph 편집 행동이 제외되는데도 LLM을 호출했다. 이 경로에서는 LLM이 실험 결과에 기여하는 역할을 분명히 하거나 별도 대조군에서 호출 비용을 줄일 여지가 있다.

다음 검증의 우선순위는 사전 graph/SDF 불일치와 worker/orchestrator 재검증 차이 해소, 동일 모델·목표·예산을 고정한 여러 target의 실제 정책 비교, 성공 가능한 편집 후보를 포함한 학습 데이터 확보다. 현재 production feature는 95개이므로 기존 93-feature 실험의 모델/데이터를 그대로 실행하지 말고 버전을 고정하거나 명시적으로 마이그레이션해야 한다.

![관측 단계와 step 비교](outcomes.png)

![예산 배분과 예측 비교](budget_and_prediction.png)
"""
(ROOT / "REPORT.md").write_text(report)
print(json.dumps({"report": str(ROOT / "REPORT.md"), "figures": ["outcomes.png", "budget_and_prediction.png"],
                  "csv": str(ROOT / "datasets.csv")}, ensure_ascii=False))
