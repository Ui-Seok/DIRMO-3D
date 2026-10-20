"""Plot the unperturbed reference alongside an existing pose-analysis snapshot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("input", "UFF", "base", "v2")
LABELS = ("Native\n(reference)", "Corrupted\ninput", "UFF", "FLOWR\nbase", "FLOWR\nv2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    args = parser.parse_args()
    root = args.analysis.resolve()
    summary = json.loads((root / "summary.json").read_text())
    with (root / "native_cases.csv").open() as handle:
        natives = list(csv.DictReader(handle))
    n = summary["cohort"]["completed_cases"]
    assert len(natives) == n == summary["native"]["n"]
    native_counts = {
        key: sum(row[column] == "True" for row in natives)
        for key, column in (("pocket_valid_pct", "pocket_valid"), ("full_valid_pct", "full_valid"))
    }
    for key, count in native_counts.items():
        assert np.isclose(100 * count / n, summary["native"][key])
    assert all(float(row["free_rmsd"]) == 0 for row in natives)
    groups = {(row["difficulty"], row["method"]): row for row in summary["groups"]}
    expected = n * summary["protocol"]["seeds_per_level"]
    assert all(groups[level, method]["n"] == expected for level in ("small", "large") for method in METHODS)
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "pdf.fonttype": 42})
    snapshot_label = "Interim snapshot" if summary["status"] == "interim" else "Completed cohort"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)
    x = np.arange(len(LABELS))
    colors = ("#2878A5", "#D77D29")
    for ax, level, title in zip(axes, ("small", "large"), ("Small perturbation: |rotation| 20–40°", "Large perturbation: |rotation| 50–70°")):
        ax.axvspan(-0.5, 0.5, color="#EDF2F3", zorder=0)
        for offset, key, label, color in zip((-.19, .19), ("pocket_valid_pct", "full_valid_pct"),
                                             ("6 Å pocket", "Full protein incl. waters / cofactors"), colors):
            values = [summary["native"][key]] + [groups[level, method][key] for method in METHODS]
            bars = ax.bar(x + offset, values, width=.35, color=color, label=label, zorder=3)
            ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], fontsize=10, padding=4)
        ax.set(title=title, xticks=x, xticklabels=LABELS, ylim=(0, 103), xlim=(-.65, 4.6))
        ax.set_yticks(np.arange(0, 101, 20))
        ax.grid(axis="y", alpha=.2, zorder=0)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("PoseBusters physical pass (%)  ↑ higher is better")
    fig.suptitle(f"64-step pose repair: original, corrupted, and repaired ligands\n{snapshot_label} • {n:,} matched complexes", fontsize=15, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .86), ncol=2, frameon=False)
    fig.text(.5, .035, f"Native: {n:,} poses (same reference in both panels). Each other bar: {expected:,} inputs ({summary['protocol']['seeds_per_level']} per complex).\nAll required dock checks must pass. Missing / failed outputs remain in the denominator.", ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(top=.72, bottom=.21, left=.07, right=.99, wspace=.08)
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"native_comparison.{extension}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), sharey=True)
    rmsd_colors = ("#89999F", "#ADB5BD", "#D77D29", "#2878A5", "#7A5195")
    for ax, level in zip(axes, ("small", "large")):
        common = summary["common_rmsd"][level]
        values = [0.] + [common["median"][method] for method in METHODS]
        ax.axvspan(-.5, .5, color="#EDF2F3", zorder=0)
        bars = ax.bar(x, values, color=rmsd_colors, width=.62, zorder=3)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=11)
        ax.set(title=f"{level.capitalize()} perturbation • {common['n']:,} common inputs", xticks=x,
               xticklabels=LABELS, ylim=(0, 2.05), xlim=(-.65, 4.6))
        ax.grid(axis="y", alpha=.2, zorder=0)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Median free-region RMSD to native (Å)  ↓ lower is better")
    fig.suptitle("Pose recovery in the fixed protein frame (64 steps)", fontsize=15, y=.98)
    fig.text(.5, .035, "Same inputs across all four methods; subset with preserved graph / stereochemistry and held coordinates.\nNative is the reference: its RMSD is zero by definition. No ligand realignment.", ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(top=.80, bottom=.24, left=.08, right=.99, wspace=.08)
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"native_rmsd_comparison.{extension}", dpi=200)
    plt.close(fig)

    rows = [{"difficulty": "none", "method": "native", "n": n,
             "pocket_valid_pct": summary["native"]["pocket_valid_pct"],
             "full_valid_pct": summary["native"]["full_valid_pct"], "primary_success_pct": ""}]
    for level in ("small", "large"):
        for method in METHODS:
            g = groups[level, method]
            rows.append({key: ("" if key == "primary_success_pct" and method == "input" else g[key]) for key in rows[0]})
    with (root / "native_comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# 원본을 포함한 64-step pose 복구 비교", "",
             f"앞선 분석과 같은 {n:,}개 완료 복합체의 고정 snapshot이다. 원본은 {n:,}개, 각 오차·방법의 입력은 {expected:,}개다. 현재 진행 중인 실험의 조건이나 결과 파일은 변경하지 않았다.", "",
             "원본은 흐트러뜨리기 전 native 좌표다. 원본 좌표에 UFF를 적용하지 않았다. 원본의 RMSD는 기준 좌표 자신과 비교하므로 0 Å이며, 종합 복구 성공 여부는 적용하지 않는다.", "",
             f"원본 PoseBusters 통과: pocket {native_counts['pocket_valid_pct']:,}/{n:,} ({summary['native']['pocket_valid_pct']:.2f}%), 전체 protein {native_counts['full_valid_pct']:,}/{n:,} ({summary['native']['full_valid_pct']:.2f}%).", "",
             "| 오차 | 방법 | Pocket 물리 통과 | 전체 protein 물리 통과 | 종합 성공 |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        success = "—" if row["primary_success_pct"] == "" else f"{row['primary_success_pct']:.2f}%"
        lines.append(f"| {row['difficulty']} | {row['method']} | {row['pocket_valid_pct']:.2f}% | {row['full_valid_pct']:.2f}% | {success} |")
    lines += ["", "UFF는 ligand 내부 에너지를 줄이는 Universal Force Field 최소화다. 수소를 추가하고 보호 heavy atom 좌표를 고정한 채 최대 200회 최소화한 뒤 수소를 제거한다. Protein 상호작용 에너지 항은 사용하지 않는다. FM 출력에 UFF를 후처리한 결과가 아니라, 같은 흐트러진 입력에 UFF만 적용한 별도 대조군이다.", "",
              "작은/큰 오차는 각각 말단 회전 가능 결합 주위의 절대 회전각 20–40° / 50–70°다. 자유 영역은 metric과 무관하게 결정적으로 선택한 2–8개 heavy atom이며, 나머지 좌표와 화학 그래프는 유지한다. 각 수준에서 seed가 다른 16개의 입력을 만든다.", "",
              "물리 통과는 PoseBusters dock preset의 선택된 20개 이진 검사 전체 통과다. 원본 ligand와의 identity/RMSD 검사는 이 함수에 포함하지 않으며 별도로 계산한다. Pocket은 native 기준 고정 6 Å 영역이고, 전체 protein 검사는 원본 파일의 물·보조인자도 포함한다.", "",
              "종합 성공은 ① graph/입체화학 보존 AND ② 보호 좌표 최대 변위 ≤0.001 Å AND ③ pocket 물리 검사 통과 AND ④ 자유 영역의 native RMSD가 흐트러진 입력보다 감소인 경우다. Vina·QED·SA를 합산한 점수가 아니다. 감소량의 최소 임계값이나 1 Å 이내 조건은 이 종합 성공 정의에 포함하지 않는다.", "",
              "![원본 포함 물리 통과 비교](figures/native_comparison.png)", "",
              "![원본 포함 RMSD 비교](figures/native_rmsd_comparison.png)", "",
              "RMSD 그림은 모든 방법에서 graph/입체화학 및 보호 좌표를 보존하고 RMSD 계산이 가능한 공통 입력만 사용한다. 물리 통과 그림은 생성 실패·누락을 분모에 유지한다. 두 그림의 분모를 구분해야 한다.", ""]
    (root / "NATIVE_COMPARISON.md").write_text("\n".join(lines))
    print(json.dumps({"native_n": n, "native_pocket_pass": native_counts["pocket_valid_pct"],
                      "native_full_pass": native_counts["full_valid_pct"],
                      "physical_figure": str(figures / "native_comparison.png"),
                      "rmsd_figure": str(figures / "native_rmsd_comparison.png")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
