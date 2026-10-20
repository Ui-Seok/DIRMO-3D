"""Compare completed, paired pose-repair step sweeps from saved measurements."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEPS = [64, 128, 256]
REPORTS = {64: ROOT / "reports/fm-pose-64-analysis-20261015-final",
           128: ROOT / "reports/fm-pose-128-analysis-20261018-final",
           256: ROOT / "reports/fm-pose-256-analysis-20261018-final"}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def table(rows):
    frame = pd.DataFrame(rows)
    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"] +
                     ["| " + " | ".join(str(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None)])


def cluster_ci(delta, metadata, cluster):
    joined = delta.rename("difference").to_frame().join(metadata[[cluster]])
    aggregate = joined.groupby(cluster).difference.agg(["sum", "count"])
    sums, counts = aggregate["sum"].to_numpy(), aggregate["count"].to_numpy()
    rng = np.random.default_rng(20261018)
    sampled = []
    for _ in range(20):
        index = rng.integers(0, len(aggregate), (100, len(aggregate)))
        sampled.extend(sums[index].sum(axis=1) / counts[index].sum(axis=1))
    return (100*np.quantile(sampled, [.025, .975])).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/fm_pose_gpu4_2000_batch32_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports = {step: read(path / "summary.json") for step, path in REPORTS.items()}
    plans = {step: read(root / f"steps_{step}/study.json") for step in STEPS}
    reference = plans[64]
    for step in STEPS:
        assert reports[step]["status"] == "complete"
        assert reports[step]["cohort"]["completed_cases"] == 2000
        assert len(plans[step]["cases"]) == len(reference["cases"]) == 2000
        for a, b in zip(reference["cases"], plans[step]["cases"]):
            assert a["case_id"] == b["case_id"] and a["graph_id"] == b["graph_id"]
            assert a["inputs"] == b["inputs"] and a["region"] == b["region"]
            assert a["source_hashes"] == b["source_hashes"]
    groups = [{"steps": step, **g} for step, report in reports.items() for g in report["groups"]]
    pd.DataFrame(groups).to_csv(out / "group_metrics.csv", index=False)
    rates = pd.concat([pd.read_csv(path / "case_rates.csv").assign(steps=step) for step, path in REPORTS.items()], ignore_index=True)
    rates.to_csv(out / "paired_case_rates.csv", index=False)
    metadata = pd.read_csv(REPORTS[64] / "native_cases.csv").set_index("case_id")
    paired = []
    for method in ["base", "v2"]:
        for level in ["small", "large"]:
            selected = rates[(rates.method == method) & (rates.difficulty == level)]
            for metric in ["pocket_valid", "full_valid", "primary_success"]:
                wide = selected.pivot(index="case_id", columns="steps", values=metric)
                assert len(wide) == 2000 and not wide.isna().any().any()
                for other in [128, 256]:
                    delta = wide[64] - wide[other]
                    paired.append({"method": method, "difficulty": level, "metric": metric,
                                   "comparison": f"64-minus-{other}", "difference_pp": 100*float(delta.mean()),
                                   "protein_cluster_95ci_pp": cluster_ci(delta, metadata, "target_id"),
                                   "scaffold_cluster_95ci_pp": cluster_ci(delta, metadata, "scaffold")})
    save(out / "paired_step_differences.json", paired)

    columns = ["case_id", "difficulty", "seed", "method", "graph_preserved", "held_preserved", "free_rmsd"]
    common_frames = []
    for step, path in REPORTS.items():
        data = pd.read_csv(path / "metrics.csv.gz", usecols=columns)
        data = data[data.method.isin(["base", "v2"])].copy()
        data.loc[~data.graph_preserved | ~data.held_preserved, "free_rmsd"] = np.nan
        data["steps"] = step
        common_frames.append(data)
    frame = pd.concat(common_frames, ignore_index=True)
    common = []
    for method in ["base", "v2"]:
        for level in ["small", "large"]:
            part = frame[(frame.method == method) & (frame.difficulty == level)]
            wide = part.pivot(index=["case_id", "seed"], columns="steps", values="free_rmsd").dropna()
            common.append({"method": method, "difficulty": level, "same_valid_inputs": len(wide),
                           "median_rmsd": {str(s): float(wide[s].median()) for s in STEPS},
                           "median_paired_change_256_minus_64": float((wide[256]-wide[64]).median())})
    save(out / "common_input_rmsd.json", common)
    computation = []
    for step, report in reports.items():
        for method, audit in report["generation_audit"].items():
            computation.append({"steps": step, "method": method, "scheduled": audit["scheduled"],
                                "raw_sdf": audit["raw_sdf_count"], "statuses": audit["statuses"],
                                "inference_hours": audit["inference_hours"], "oom_retries": audit["oom_retries"],
                                "max_held_tensor_drift": audit["max_held_tensor_drift"],
                                "contract_failures": audit["contract_failure_types"]})
    save(out / "generation_accounting.json", computation)
    summary = {"status": "complete", "analysis_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
               "cases": 2000, "unique_exact_proteins": reference["unique_exact_proteins"],
               "steps": STEPS, "scheduled_fm_slots": 384000, "same_saved_inputs_verified": True,
               "source_reports": {str(s): str(p) for s, p in REPORTS.items()},
               "native": reports[64]["native"], "groups": groups, "paired": paired,
               "common_rmsd": common, "computation": computation}
    save(out / "summary.json", summary)
    figures(out, reports, common)
    write_report(out, root, summary, reports)
    print(json.dumps({"status": "complete", "report": str(out / "REPORT.md"), "paired_inputs_verified": 64000}, ensure_ascii=False))


def figures(out, reports, common):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 170})
    colors = {"base": "#267cb0", "v2": "#bd6684"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for col, level in enumerate(["small", "large"]):
        for row, (metric, title) in enumerate([("pocket_valid_pct", "PoseBusters pass"), ("primary_success_pct", "Strict recovery success")]):
            ax = axes[row, col]
            for method in ["base", "v2"]:
                ys = [next(g[metric] for g in reports[s]["groups"] if g["method"] == method and g["difficulty"] == level) for s in STEPS]
                ax.plot(STEPS, ys, "o-", lw=2, color=colors[method], label=f"FLOWR {method}")
                for s, y in zip(STEPS, ys):
                    ax.annotate(f"{y:.1f}", (s, y), xytext=(3, 7 if method == "base" else -15), textcoords="offset points", fontsize=9)
            for method, color, label in [("input", "#8b9199", "Twisted input"), ("UFF", "#bc9a3e", "UFF control")]:
                if row == 1 and method == "input":
                    continue
                value = next(g[metric] for g in reports[64]["groups"] if g["method"] == method and g["difficulty"] == level)
                ax.axhline(value, ls="--", color=color, label=label, lw=1.3)
            if row == 0:
                ax.axhline(reports[64]["native"]["pocket_valid_pct"], ls=":", color="#55785a", label="Native 88.4%")
            angle = "20-40 degrees" if level == "small" else "50-70 degrees"
            ax.set(xticks=STEPS, ylim=(0, 100), xlabel="Integration steps", ylabel="Scheduled-input rate (%)", title=f"{title} | {level} twist ({angle})")
            ax.legend(fontsize=8, ncol=2, loc="lower left" if row == 0 else "upper right")
    fig.suptitle("Local pose repair | 2,000 complexes | same input across models and steps\n32,000 scheduled poses per model / twist level / step", fontsize=13)
    fig.savefig(out / "step_comparison.png");fig.savefig(out / "step_comparison.pdf");plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, level in zip(axes, ["small", "large"]):
        for method in ["base", "v2"]:
            g = next(x for x in common if x["method"] == method and x["difficulty"] == level)
            values = [g["median_rmsd"][str(s)] for s in STEPS]
            ax.plot(STEPS, values, "o-", color=colors[method], label=f"{method}: {g['same_valid_inputs']:,} common inputs")
        for method, color in [("input", "#8b9199"), ("UFF", "#bc9a3e")]:
            value = next(g["conditional_median_free_rmsd"] for g in reports[64]["groups"] if g["method"] == method and g["difficulty"] == level)
            ax.axhline(value, ls="--", color=color, label=f"{method} baseline (all eligible)")
        ax.set(xticks=STEPS, xlabel="Integration steps", ylabel="Free-region RMSD (angstrom; lower is closer)", title=f"{level} twist: same eligible inputs at all steps")
        ax.legend(fontsize=8)
    fig.suptitle("Distance to the original pose | no ligand alignment; symmetry-aware", fontsize=12)
    fig.savefig(out / "rmsd_comparison.png");fig.savefig(out / "rmsd_comparison.pdf");plt.close(fig)


def write_report(out, root, a, reports):
    metrics = []
    for step in STEPS:
        for method in ["base", "v2"]:
            gs = [next(g for g in reports[step]["groups"] if g["method"] == method and g["difficulty"] == d) for d in ["small", "large"]]
            metrics.append({"Step": step, "모델": method, "작은 비틀기 PB": f"{gs[0]['pocket_valid_pct']:.2f}%", "큰 비틀기 PB": f"{gs[1]['pocket_valid_pct']:.2f}%",
                            "작은 비틀기 종합 성공": f"{gs[0]['primary_success_pct']:.2f}%", "큰 비틀기 종합 성공": f"{gs[1]['primary_success_pct']:.2f}%"})
    controls = []
    for d in ["small", "large"]:
        for method in ["input", "UFF"]:
            g = next(g for g in reports[64]["groups"] if g["method"] == method and g["difficulty"] == d)
            controls.append({"비틀기": d, "방법": method, "PB 통과": f"{g['pocket_valid_pct']:.2f}%", "자유 영역 RMSD 중앙값": f"{g['conditional_median_free_rmsd']:.3f} Å",
                             "종합 성공": "기준선" if method == "input" else f"{g['primary_success_pct']:.2f}%"})
    glossary = [{"용어": "Chemical graph", "쉬운 뜻": "어떤 원자가 어떤 결합으로 연결됐는지 나타내는 분자의 설계도. QED/SA 목표값 자체가 아님."},
                {"용어": "Pose / 좌표", "쉬운 뜻": "그 분자가 단백질 안에서 놓인 위치와 자세."},
                {"용어": "Core / 자유 영역", "쉬운 뜻": "움직이지 못하게 붙잡아 둔 원자들 / 움직여 복구할 수 있는 말단 원자들."},
                {"용어": "Step", "쉬운 뜻": "FM 생성 경로를 몇 구간으로 나누어 계산하는지. 학습 epoch나 데이터 수와 다름."},
                {"용어": "작은/큰 오차", "쉬운 뜻": "말단을 연결한 회전 가능한 결합 주변으로 각각 20–40도 / 50–70도 비튼 입력."},
                {"용어": "PB 통과", "쉬운 뜻": "PoseBusters가 결합 길이·각도·분자 내부 충돌·단백질 충돌 등을 검사해 모두 통과한 상태."},
                {"용어": "RMSD", "쉬운 뜻": "원본 원자 위치와 얼마나 떨어졌는지 나타내는 거리. 여기서는 움직이는 영역만 측정하며 작을수록 원본에 가까움."},
                {"용어": "입체화학", "쉬운 뜻": "연결은 같아도 왼손/오른손처럼 달라질 수 있는 공간 배치. 분자를 통째로 돌리는 것과는 다름."},
                {"용어": "UFF", "쉬운 뜻": "학습 모델 없이 힘장으로 원자 위치를 다듬는 대조군. 여기서는 ligand-only이며 단백질 에너지를 고려하지 않음."},
                {"용어": "종합 성공", "쉬운 뜻": "PB 통과 + 입체화학까지 graph 보존 + 고정 좌표 유지 + 시작보다 자유 영역 RMSD 감소."}]
    runtime = [{"Step": x["steps"], "모델": x["method"], "예정 슬롯": x["scheduled"], "실제 SDF": x["raw_sdf"],
                "정상 계약": x["statuses"].get("ok", 0), "입체화학 등 계약 실패": x["statuses"].get("contract_failed", 0),
                "전처리 실패": x["statuses"].get("failed", 0), "최종 OOM": x["statuses"].get("oom", 0),
                "추론 누적 시간": f"{x['inference_hours']:.2f} h", "OOM 재시도": x["oom_retries"]} for x in a["computation"]]
    failure_rows = []
    for step in STEPS:
        failure = read(REPORTS[step] / "failure_checks.json")
        for level in ["small", "large"]:
            for method in ["base", "v2"]:
                x = failure[f"{level}:{method}:pocket"]
                failure_rows.append({"Step": step, "비틀기": level, "모델": method,
                                     **{label: f"{100*x.get(key, 0)/x['scheduled']:.2f}%" for key, label in [
                                         ("minimum_distance_to_protein", "단백질 최소거리 실패"), ("bond_lengths", "결합 길이 실패"),
                                         ("bond_angles", "결합 각도 실패"), ("internal_energy", "내부 에너지 실패"),
                                         ("internal_steric_clash", "내부 충돌 실패")]}})
    paired_rows = []
    for x in a["paired"]:
        if x["metric"] != "primary_success":
            continue
        ci = x["protein_cluster_95ci_pp"]
        paired_rows.append({"모델": x["method"], "비틀기": x["difficulty"], "비교": x["comparison"],
                            "종합 성공 차이": f"{x['difference_pp']:+.2f}%p", "동일 protein 군집 95% CI": f"[{ci[0]:+.2f}, {ci[1]:+.2f}]%p"})
    rmsds = [{"모델": x["method"], "비틀기": x["difficulty"], "공통 유효 입력": x["same_valid_inputs"],
              **{f"{s}-step": f"{x['median_rmsd'][str(s)]:.3f} Å" for s in STEPS},
              "동일 입력 Δ(256−64) 중앙값": f"{x['median_paired_change_256_minus_64']:+.3f} Å"} for x in a["common_rmsd"]]
    lines = ["# FLOWR pose 복구 64·128·256-step 최종 비교", "",
             "**세 실험은 모두 2,000개 복합체의 평가까지 완료됐다. 이번 국소 pose 복구 조건에서는 base 64-step이 종합 성공률과 계산 비용 측면에서 가장 좋았다.**", "",
             f"분석 시각: {a['analysis_kst']} (KST). 원본: `{root}`.", "",
             "## 무엇을 보기 위한 실험인가", "",
             "분자를 관절이 있는 열쇠에 비유하면, 원래 잘 놓여 있던 열쇠의 끝부분만 일부러 비틀어 둔 뒤, 몸통은 붙잡고 끝을 제자리로 고칠 수 있는지 확인한 실험이다. 그리고 같은 문제를 64·128·256-step으로 풀었을 때 더 오래 계산하면 실제로 더 잘 고치는지도 비교했다.", "",
             "1. PDBbind2019 복합체 2,000개에서 원본 ligand와 protein을 준비했다. 서로 다른 복합체이며 exact protein sequence는 1,458종이다. Refined 674개, other-PL 1,326개다. 사전학습 중복 여부는 미확인이다.",
             "2. 회전 가능한 결합으로 연결된 말단 heavy atom 2–8개를 graph 기준으로 선택했다. PB 점수가 나쁜 fragment를 찾아 정한 영역은 아니다.",
             "3. 원본의 나머지 좌표는 유지하면서 말단을 작은 각도 20–40도 / 큰 각도 50–70도로 비틀었다. 각 수준별 16개의 다른 입력을 저장했다. 각 seed는 서로 다른 비틀기 입력이며 한 입력에서 반복 생성한 16개 후보가 아니다.",
             "4. 전체 chemical graph와 core 좌표, pocket을 조건으로 주고 자유 영역의 좌표만 복구했다. base와 v2 checkpoint, 64·128·256-step에 같은 저장 입력을 사용했다. 세 manifest의 case/graph/region/source hash/입력 목록이 전부 같은지 확인했다.",
             "5. 원본 → 비튼 입력 → UFF 보정 또는 FM 복구 후를 비교했다. 복합체마다 32개 입력 × 모델 2개 = step별 128,000 예정 결과이며 세 step 총 384,000슬롯이다. 실패도 분모에 남겼다. FM/LLM 학습, 화학 graph 편집, LLM 호출은 없다.", "",
             "최근 100-pocket 실험은 원본과 N/O/F 편집 graph 및 whole/core 두 조건을 비교한다. 이 2,000개 step 실험은 **화학 graph가 그대로인 core 고정 국소 pose 복구**만 수행했다. 서로 다른 질문을 보는 실험이다.", "",
             "## 용어를 쉽게 읽는 방법", "", table(glossary), "",
             "RMSD는 protein의 같은 좌표계에서 분자를 따로 겹쳐 정렬하지 않고 계산하며, 대칭적으로 같은 원자 대응은 허용한다. 종합 성공은 입력보다 조금이라도 가까워지면 RMSD 개선으로 센다. 0.1 Å 이상 개선한 별도 집계는 각 step 보고서의 `material_success_0p1A_pct`에 있다. 종합 성공에는 affinity/QED/SA 개선 조건이 들어가지 않는다.", "",
             "## 원본과 대조군", "",
             "원본 실험 ligand는 pocket PB 1,768/2,000 = **88.40%**, 전체 protein PB 1,703/2,000 = **85.15%** 통과했다. 원본부터 실패한 사례도 결과를 본 뒤 제외하지 않았다. Protein-only pocket과 전체 protein의 물리 검사 결과를 분리했다.", "", table(controls), "",
             "## Step·모델별 결과", "",
             "각 표 칸의 분모는 32,000개이며, 전처리·생성 실패를 제외하지 않았다.", "", table(metrics), "",
             f"![Step comparison]({out / 'step_comparison.png'})", "",
             "base는 작은 비틀기에서 64→256-step 종합 성공이 61.90→49.33%, 큰 비틀기에서는 56.19→50.18%로 내려갔다. v2도 같은 방향이며, 이 복구 조건에서는 모든 step에서 base보다 낮았다. v2가 다른 task에서도 열등하다는 뜻은 아니다.", "",
             "같은 복합체의 반복 입력과 유사 protein의 상관을 고려해 exact protein 단위로 재표집한 탐색적 95% 범위다. Scaffold 단위 결과도 paired_step_differences.json에 저장했다. 양수는 64-step이 더 좋다는 뜻이다.", "", table(paired_rows), "",
             "## 원본과의 거리도 같이 봐야 한다", "",
             "아래는 세 step 모두에서 graph/입체화학과 고정 좌표가 보존된 **같은 입력**만 남겨 RMSD를 비교한 것이다. 유효 결과만의 조건부 거리이며 전체 성공률과 구분해야 한다.", "", table(rmsds), "",
             f"![RMSD comparison]({out / 'rmsd_comparison.png'})", "",
             "base의 큰 비틀기에서는 step을 늘리면 원본과의 거리 중앙값이 약간 작아질 수 있지만, PB와 종합 성공은 나빠졌다. 원본과 비슷한 위치로 가는 것과 결합 기하·충돌·입체화학을 모두 만족시키는 것은 별개다. 작은 비틀기에서는 base RMSD도 악화됐다.", "",
             "전체 거리 중앙값의 차이와 입력별 거리 변화의 중앙값은 서로 다른 통계다. base 큰 비틀기의 동일 입력별 256−64 변화 중앙값은 +0.007 Å로, 이 기준에서는 개선으로 해석할 수 없다.", "",
             "## 어떤 검사에서 실패했나", "",
             "한 출력이 여러 검사를 동시에 실패할 수 있으므로 아래 실패율을 합산하지 않는다. 분모는 예정 32,000개이며 분자가 없는 출력은 별도다.", "", table(failure_rows), "",
             "Step 증가가 무엇을 직접 유발했는지 이 최종 결과만으로 확정할 수 없다. 생성 중간 경로의 에너지·결합·좌표 변화를 기록하는 실험이 있어야 원인을 더 분명히 알 수 있다. 현재 결과는 이 초기화·고정 조건·solver에서 step 증가가 품질 향상을 보장하지 않았다는 증거다.", "",
             "## 완료와 실패 기록을 구분", "", table(runtime), "",
             "각 모델·step에서 공통 전처리 실패는 1,344슬롯, 42개 복합체다. 37개 복합체는 graph tensor round-trip, 5개는 pocket SE 원소 vocabulary 문제였다. 256-step에는 이와 별도로 base 96개/v2 25개의 최종 OOM이 있다. 실패를 제외하지 않았으며, 이 기록 때문에 `완료`는 모든 예정 결과를 성공 분자로 만들었다는 뜻이 아니라 모든 슬롯의 처리와 평가가 끝났다는 뜻이다.", "",
             "256-step의 OOM 추가 누락은 base 0.15%, v2 약 0.04%다. 관찰된 64→256 종합 성공 감소는 이를 훨씬 넘으며, OOM만으로 설명되지 않는다. 생성된 출력의 보호 좌표 tensor 최대 이동은 전 step·모델에서 0 Å였다. Graph 계약 실패는 각 audit의 stereo_only/nonstereo 항목으로 구분했다.", "",
             "추론 누적 시간은 각 모델의 추론 구간 경과 시간을 더한 값이다. 같은 GPU에서 모델들이 병행됐으므로 두 모델의 시간을 더해서 실제 실험 총시간으로 해석하지 않는다. GPU/공유 부하도 달라 정밀 속도 benchmark는 아니다. 그래도 step 증가에 따른 계산 비용 증가는 분명하다.", "",
             "## 실용적인 판단", "",
             "- 현재의 core 고정 국소 pose 복구에는 **FLOWR.root base + 64-step**을 기준 설정으로 유지한다. 128/256으로 늘릴 실험적 이득은 없었다.",
             "- 이번 결과는 열쇠 끝을 고치는 능력에 관한 것이다. 새 열쇠를 처음부터 만드는 pocket-only SBDD 성능이나 화학 성질 최적화 성능을 검증한 결과로 확대하지 않는다.",
             "- 입체화학·기하 조건을 통과하는지 검사하고 후보를 여러 개 평가하는 과정이 여전히 필요하다. Step만 늘리는 방식보다 현재 진행 중인 명시적 graph·좌표 조건 비교를 완결하는 것이 우선이다.", "",
             "## 원본 보고서와 데이터", ""]
    for step, path in REPORTS.items():
        lines.append(f"- [{step}-step 상세 보고서]({path / 'REPORT.md'})")
    lines += ["- group_metrics.csv: native subset·full protein·거리·물리 개선/악화 등을 포함한 전체 조건 통계.",
              "- paired_case_rates.csv / paired_step_differences.json: 동일 복합체 paired 분석과 protein/scaffold 군집 CI.",
              "- common_input_rmsd.json: 세 step 공통 유효 입력의 거리 비교.",
              "- generation_accounting.json: 실제 생성 수, 실패 종류, OOM, 계산량."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
