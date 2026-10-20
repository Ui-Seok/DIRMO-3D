"""Compare raw FM draws with matched priors, native/input controls and docking."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dirmo.io import write_json


def bootstrap(values):
    values = np.array(values, dtype=float)
    if not len(values):
        return None
    samples = np.random.default_rng(2026).choice(values, (4000, len(values)), replace=True).mean(1)
    return {"mean": float(values.mean()), "target_bootstrap_95ci": np.quantile(samples, [.025, .975]).tolist()}


def full_protein_check(item):
    molecule_path, protein_path, output = item
    from dirmo.chemistry.graphs import read_sdf
    from dirmo.evaluation.metrics import posebusters
    try:
        result = {"status": "ok", **posebusters(read_sdf(molecule_path), protein_path)}
    except Exception as exc:
        result = {"status": "failed", "passes": False, "reason": f"{type(exc).__name__}: {exc}"}
    write_json(output, result)


def analyze(root, allow_incomplete=False):
    from dirmo.evaluation.fm_quality import aggregate
    from types import SimpleNamespace
    summary = aggregate(SimpleNamespace(output=root))
    if summary["status"] != "complete" and not allow_incomplete:
        raise ValueError("Experiment still running; use --allow-incomplete for an interim analysis")
    plan = json.loads((root / "study.json").read_text())
    rows = [json.loads(line) for line in (root / "draws.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["checkpoint"], row["condition"])].append(row)
    case_by_id = {case["case_id"]: case for case in plan["cases"]}
    full_secondary = {}
    if summary["status"] == "complete":
        jobs = []
        for row in rows:
            raw = row["generation"].get("sdf_path")
            destination = Path(row["directory"]) / "full-protein-posebusters.json"
            if raw and not destination.exists():
                jobs.append((raw, case_by_id[row["case_id"]]["protein_path"], str(destination)))
        for case in plan["cases"]:
            for label, path in [("native", case["truth_sdf"]), ("input", case["input_sdf"])]:
                destination = root / "cases" / case["case_id"] / f"{label}-full-protein-posebusters.json"
                if not destination.exists():
                    jobs.append((path, case["protein_path"], str(destination)))
        if jobs:
            with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
                for _ in pool.map(full_protein_check, jobs, chunksize=8):
                    pass
        for key, group in groups.items():
            count, passes, strict, by_failure = 0, 0, 0, Counter()
            for row in group:
                path = Path(row["directory"]) / "full-protein-posebusters.json"
                if not path.exists():
                    continue
                check = json.loads(path.read_text())
                count += 1
                passes += bool(check["passes"])
                metrics = row["metrics"]
                endpoint = (metrics.get("rmsd", float("inf")) <= 2. if row["mode"] == "graph"
                            else metrics.get("changed_graph", False))
                strict += bool(check["passes"] and metrics.get("contract_valid", False) and endpoint)
                by_failure.update(name for name, passed in check.get("checks", {}).items() if passed is not True)
            full_secondary[":".join(key)] = {"attempts": len(group), "evaluated_raw_molecules": count,
                                            "full_pb_pass": passes, "full_strict_success": strict,
                                            "failure_checks": dict(by_failure)}
    baselines = {"input": [], "native": [], "docking": []}
    for case in plan["cases"]:
        directory = root / "cases" / case["case_id"]
        for key in ("input", "native"):
            path = directory / f"{key}-metrics.json"
            if path.exists():
                baselines[key].append({"case_id": case["case_id"], **json.loads(path.read_text())})
        path = directory / "docking.json"
        if path.exists():
            baselines["docking"].append({"case_id": case["case_id"], **json.loads(path.read_text())})
    baseline_summary = {}
    for key in ("input", "native"):
        records = baselines[key]
        baseline_summary[key] = {"targets": len(records),
                                 "contract_valid": sum(x.get("contract_valid", False) for x in records),
                                 "pb_valid": sum(x.get("pb_valid", False) for x in records),
                                 "pose_success": sum(x.get("pose_success", False) for x in records),
                                 "median_rmsd": float(np.median([x["rmsd"] for x in records if "rmsd" in x])) if any("rmsd" in x for x in records) else None}
    docking = baselines["docking"]
    successful_docking = [x for x in docking if x["status"] == "ok"]
    baseline_summary["docking"] = {"cases": len(docking), "status": dict(Counter(x["status"] for x in docking)),
                                   "actual_pose_count": sum(x["actual_poses"] for x in successful_docking),
                                   "top1_pose_success": sum(x["poses"][0].get("pose_success", False) for x in successful_docking if x["poses"]),
                                   "oracle_any_pose_success": sum(any(p.get("pose_success", False) for p in x["poses"]) for x in successful_docking),
                                   "cpu_seconds": sum(x.get("latency_seconds", 0) for x in successful_docking),
                                   "failures": [{"case_id": x["case_id"], "reason": x.get("reason")} for x in docking if x["status"] != "ok"]}
    details = []
    for (checkpoint, condition), group in sorted(groups.items()):
        by_target = defaultdict(list)
        failures = Counter()
        coordinate_sets = defaultdict(set)
        for row in group:
            by_target[row["target_id"]].append(row)
            checks = row["metrics"].get("posebusters", {}).get("checks", {})
            failures.update(name for name, passed in checks.items() if passed is not True)
            model_file = Path(row["directory"]) / "raw.sdf"
            if model_file.exists():
                from rdkit import Chem
                mol = Chem.SDMolSupplier(str(model_file), removeHs=False, sanitize=False)[0]
                if mol is not None:
                    coordinate_sets[row["case_id"]].add(tuple(mol.GetConformer().GetPositions().round(4).ravel()))
        paired_prior = []
        prior_successes, prior_available = 0, 0
        for samples in by_target.values():
            complete_pairs = [x for x in samples if "prior_metrics" in x]
            if complete_pairs:
                paired_prior.append(np.mean([int(x["metrics"].get("pose_success", False)) -
                                             int(x["prior_metrics"].get("pose_success", False)) for x in complete_pairs]))
                prior_successes += sum(x["prior_metrics"].get("pose_success", False) for x in complete_pairs)
                prior_available += len(complete_pairs)
        top1_available, top1_success = 0, 0
        for samples in by_target.values():
            eligible = [x for x in samples if x["metrics"].get("contract_valid") and x["metrics"].get("pb_valid")
                        and x["metrics"].get("vina", {}).get("status") == "ok"]
            if eligible:
                top1_available += 1
                selected = min(eligible, key=lambda x: x["metrics"]["vina"]["score"])
                top1_success += bool(selected["metrics"].get("pose_success", False))
        valid_fragment = [x for x in group if x["metrics"].get("valid_changed_fragment", False)]
        changes = []
        native_by_id = {x["case_id"]: x for x in baselines["native"]}
        for row in valid_fragment:
            native = native_by_id.get(row["case_id"])
            metrics = row["metrics"]
            if native and native.get("vina", {}).get("status") == "ok" and metrics.get("vina", {}).get("status") == "ok":
                changes.append({"case_id": row["case_id"], "seed": row["seed"],
                                "delta_qed": metrics["properties"]["qed"] - native["properties"]["qed"],
                                "delta_sa": metrics["properties"]["sa"] - native["properties"]["sa"],
                                "delta_logp": metrics["properties"]["logp"] - native["properties"]["logp"],
                                "delta_vina": metrics["vina"]["score"] - native["vina"]["score"]})
        pareto = []
        for candidate in changes:
            # Candidate dominates the native ligand under QED↑, SA↓, Vina↓.
            directions = [candidate["delta_qed"], -candidate["delta_sa"], -candidate["delta_vina"]]
            if all(v >= -1e-8 for v in directions) and any(v > 1e-8 for v in directions):
                pareto.append(candidate)
        details.append({"checkpoint": checkpoint, "condition": condition,
                        "fm_minus_matched_prior": bootstrap(paired_prior),
                        "prior_pose_success": prior_successes, "prior_evaluated": prior_available,
                        "pb_failure_checks": dict(failures),
                        "unique_coordinate_draws_per_case": {k: len(v) for k, v in coordinate_sets.items()},
                        "vina_selected_top1_scored_valid_targets": top1_available,
                        "vina_selected_top1_pose_success": top1_success,
                        "valid_fragment_property_vectors": changes,
                        "fragment_dominates_native_qed_sa_vina": pareto})
    paired_conditions = []
    lookup = {(r["checkpoint"], r["condition"], r["target_id"], r["seed"]): r for r in rows}
    comparisons = [(checkpoint, "random", checkpoint, "reference0") for checkpoint in ("base", "v2")]
    comparisons += [(checkpoint, "reference0", checkpoint, "reference03") for checkpoint in ("base", "v2")]
    comparisons += [("base", condition, "v2", condition) for condition in ("random", "reference0", "reference03", "fragment_small", "fragment_large")]
    for left_checkpoint, left_condition, right_checkpoint, right_condition in comparisons:
        target_deltas = defaultdict(list)
        for row in groups.get((left_checkpoint, left_condition), []):
            other = lookup.get((right_checkpoint, right_condition, row["target_id"], row["seed"]))
            if other:
                endpoint = "pose_success" if row["mode"] == "graph" else "valid_changed_fragment"
                target_deltas[row["target_id"]].append(int(other["metrics"].get(endpoint, False)) -
                                                      int(row["metrics"].get(endpoint, False)))
        paired_conditions.append({"left": f"{left_checkpoint}:{left_condition}", "right": f"{right_checkpoint}:{right_condition}",
                                  "paired_targets": len(target_deltas), "pairs": sum(map(len, target_deltas.values())),
                                  "right_minus_left_primary": bootstrap([np.mean(v) for v in target_deltas.values()])})
    report = {"status": summary["status"], "baselines": baseline_summary, "details": details,
              "paired_conditions": paired_conditions,
              "full_protein_secondary": full_secondary,
              "timing_context": "Two FM processes share GPU 3 with existing jobs; latencies are not isolated benchmarks"}
    write_json(root / "analysis.json", report)
    with (root / "summary.csv").open("w") as stream:
        columns = ["checkpoint", "condition", "attempts", "raw_draws", "nfe", "build_valid_count", "contract_valid_count",
                   "pb_valid_count", "primary_rate", "targets_with_any_primary_success", "median_rmsd_contract_valid_only"]
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["groups"])
    lines = ["# FLOWR 생성 품질 비교 분석", "", f"상태: {summary['status']}; {summary['recorded_attempts']}/{summary['planned_attempts']} attempts.", "",
             "생성은 GPU 3에서 수행했다. LLM/router는 호출하지 않았다. 조건별 실패도 분모에 포함했다. 정답 pose를 주지 않은 pose 과제와 held core를 준 fragment 과제는 분리해서 해석한다.", "",
             "## 대조군", "", "| 대조군 | 평가 targets | PB-valid | pose 성공 |", "| --- | ---: | ---: | ---: |"]
    for key in ("input", "native"):
        b = baseline_summary[key]
        lines.append(f"| {key} | {b['targets']} | {b['pb_valid']} | {b['pose_success']} |")
    dock = baseline_summary["docking"]
    lines.extend(["", f"Docking status: {dock['status']}. 성공한 receptor subset에서 Top-1 pose 성공 {dock['top1_pose_success']}/{dock['status'].get('ok', 0)}, oracle best-of-actual-poses 성공 {dock['oracle_any_pose_success']}. 실제 pose 수 {dock['actual_pose_count']}.",
                  "Vina용 receptor가 준비된 subset의 결과이므로 전체 20 targets FM 결과와 직접 승률 비교하지 않는다.", "",
                  "## FM 결과", "", "| checkpoint | condition | 유효 build | 계약 통과 | PB-valid | primary 성공률 | 1회 이상 성공 target |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for g in summary["groups"]:
        lines.append(f"| {g['checkpoint']} | {g['condition']} | {g['build_valid_count']}/{g['attempts']} | {g['contract_valid_count']} | {g['pb_valid_count']} | {g['primary_rate']:.1%} | {g['targets_with_any_primary_success']}/{g['targets']} |")
    lines.extend(["", "Pose primary = contract + PB-valid + RMSD ≤2 Å. Fragment primary = held contract + PB-valid + changed graph. RMSD는 ligand 정렬 없는 protein frame 값이다.", "",
                  "## 짝 비교", "", "| left | right | targets | right-left primary 차이 (95% target bootstrap) |",
                  "| --- | --- | ---: | --- |"])
    for comparison in paired_conditions:
        delta = comparison["right_minus_left_primary"]
        if delta:
            lo, hi = delta["target_bootstrap_95ci"]
            lines.append(f"| {comparison['left']} | {comparison['right']} | {comparison['paired_targets']} | {delta['mean']:+.1%} [{lo:+.1%}, {hi:+.1%}] |")
    lines.extend(["", "## 실패와 다양성", ""])
    for d in details:
        common = sorted(d["pb_failure_checks"].items(), key=lambda x: -x[1])[:4]
        if d["condition"] == "reference0":
            unique = list(d["unique_coordinate_draws_per_case"].values())
            lines.append(f"- {d['checkpoint']} reference0: raw 좌표 고유 개수 median {np.median(unique) if unique else 'missing'}; seed 수를 그대로 다양성으로 세지 않는다.")
        lines.append(f"- {d['checkpoint']} {d['condition']}: 주요 PB 실패 {common}.")
    lines.extend(["", "## 해석 범위", "", "- 학습 데이터와의 overlap은 미확인이다. 독립 test 일반화 주장이나 fine-tuning 효과를 측정한 결과가 아니다.",
                  "- Pocket은 protein-only이며 water/cofactor가 제외돼 있다. Vina receptor는 표준 단백질 잔기와 명시적 HIE template를 사용했다. receptor 준비 실패를 제외 사유로 숨기지 않았다.",
                  "- Whole-graph 조건의 원래 학습 여부는 metadata만으로 확인할 수 없다. Fragment 조건 설정이 존재해도 held graph와 자유 영역 품질은 별도 평가했다.",
                  "- 두 FM 프로세스와 기존 작업이 GPU 3을 공유했으므로 latency를 단독 GPU 성능으로 해석하지 않는다.",
                  "- score-only Vina는 binding affinity 실험값이 아니다. fragment의 QED/SA/Vina 변화와 native 지배 후보는 analysis.json에 벡터로 저장했다.",
                  "", "![조건별 품질](condition_quality.png)"])
    if full_secondary:
        lines.extend(["", "## 원본 전체 protein의 추가 물리 검사", "",
                      "위 primary를 바꾸지 않고 추가한 검사다. FM에 전달한 crop 대신 원본 protein PDB에 남아 있는 단백질·cofactor·water를 대상으로 같은 PoseBusters dock preset을 적용했다.", "",
                      "| 조건 | 전체 attempts | full PB-valid | full PB + 계약 + 과제 성공 |",
                      "| --- | ---: | ---: | ---: |"])
        for key, values in sorted(full_secondary.items()):
            lines.append(f"| {key} | {values['attempts']} | {values['full_pb_pass']} | {values['full_strict_success']} |")
    (root / "ANALYSIS.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, conditions, title in [(axes[0], ["random", "reference0", "reference03"], "Pose: contract + PB + RMSD <= 2 A"),
                                     (axes[1], ["fragment_small", "fragment_large"], "Fragment: held + PB + changed graph")]:
        for index, checkpoint in enumerate(("base", "v2")):
            values = []
            for condition in conditions:
                g = next((g for g in summary["groups"] if g["checkpoint"] == checkpoint and g["condition"] == condition), None)
                values.append(g["primary_rate"] if g else 0)
            axis.bar(np.arange(len(conditions)) + (index - .5) * .35, values, width=.35, label=checkpoint)
        axis.set_xticks(np.arange(len(conditions)), conditions)
        axis.set_ylim(0, 1)
        axis.set_ylabel("Fraction of scheduled attempts")
        axis.set_title(title)
        axis.legend()
    figure.savefig(root / "condition_quality.png", dpi=180)
    plt.close(figure)
    print(json.dumps({"status": summary["status"], "report": str(root / "ANALYSIS.md")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--watch-parent", type=int, help="Finish analysis automatically after this benchmark process completes")
    args = parser.parse_args()
    if args.watch_parent:
        root = args.run.resolve()
        state = {"status": "waiting", "pid": __import__('os').getpid(), "parent_pid": args.watch_parent}
        write_json(root / "completion-watcher.json", state)
        while True:
            summary_file = root / "summary.json"
            summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
            parent_file = Path(f"/proc/{args.watch_parent}/cmdline")
            try:
                parent_running = b"benchmark_fm_conditions.py" in parent_file.read_bytes()
            except OSError:
                parent_running = False
            if summary.get("status") == "complete" and not parent_running:
                break
            if not parent_running:
                state.update(status="incomplete", reason="Benchmark process exited before all planned draws/evaluations completed")
                write_json(root / "completion-watcher.json", state)
                raise SystemExit(2)
            time.sleep(10)
    analyze(args.run.resolve(), args.allow_incomplete)
    if args.watch_parent:
        state.update(status="complete", finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(root / "completion-watcher.json", state)
