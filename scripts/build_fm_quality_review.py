"""Build the final review with coordinate-restored stereo and matched controls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dirmo.chemistry.graphs import read_sdf, smiles
from dirmo.io import write_json
from review_fm_quality import bootstrap, coordinate_stereo


def build(root, output):
    audit = json.loads((output / "forensic-summary.json").read_text())
    prior_audit_file = output / "fragment-prior-audit.json"
    prior_audit = json.loads(prior_audit_file.read_text()) if prior_audit_file.exists() else None
    if audit["full_protein_files_present"] != audit["full_protein_files_expected"]:
        raise ValueError("Full-protein secondary checks still pending")
    items = [json.loads(s) for s in (output / "coordinate-stereo-audit.jsonl").read_text().splitlines()]
    plan = json.loads((root / "study.json").read_text())
    cases = {c["case_id"]: c for c in plan["cases"]}
    native = {key: json.loads((root / "cases" / key / "native-metrics.json").read_text()) for key in cases}
    groups = defaultdict(list)
    for item in items:
        groups[(item["checkpoint"], item["condition"])].append(item)
    comparisons = []
    for condition in ("fragment_small", "fragment_large"):
        lookup = {(i["checkpoint"], i["target_id"], i["seed"]): i for i in items if i["condition"] == condition}
        paired = defaultdict(list)
        for item in groups[("base", condition)]:
            other = lookup[("v2", item["target_id"], item["seed"])]
            paired[item["target_id"]].append(int(other.get("coordinate_full_success", False)) - int(item.get("coordinate_full_success", False)))
        comparisons.append({"condition": condition, "full_v2_minus_base": bootstrap([np.mean(v) for v in paired.values()])})
    # Docking uses the same eleven prepared-receptor cases. Restore stereo from
    # pose coordinates, but keep its original physics/score measurements.
    docking, scored_case_ids = [], set()
    for key, case in cases.items():
        record = json.loads((root / "cases" / key / "docking.json").read_text())
        if record["status"] != "ok":
            continue
        poses = []
        truth = read_sdf(case["truth_sdf"])
        for atom in truth.GetAtoms():
            atom.SetAtomMapNum(0)
        for index, metric in enumerate(record["poses"]):
            mol = coordinate_stereo(read_sdf(root / "cases" / key / f"docked_{index}.sdf"))
            contract = smiles(mol) == smiles(truth)
            rmsd = float(rdMolAlign.CalcRMS(mol, truth, maxMatches=4096)) if contract else None
            poses.append({"rank": index, "contract": contract, "pb_valid": metric.get("pb_valid", False),
                          "rmsd": rmsd, "success": bool(contract and metric.get("pb_valid", False) and rmsd <= 2.),
                          "vina": metric.get("vina", {})})
        eligible = [p for p in poses if p["contract"] and p["pb_valid"] and p["vina"].get("status") == "ok"]
        selected = min(eligible, key=lambda p: p["vina"]["score"]) if eligible else None
        docking.append({"case_id": key, "poses": poses, "rank1_success": poses[0]["success"],
                        "any_success": any(p["success"] for p in poses),
                        "filtered_top1_success": bool(selected and selected["success"])})
        scored_case_ids.add(key)
    matched_fm = []
    for key, group in groups.items():
        if group[0]["mode"] != "graph":
            continue
        by_case = defaultdict(list)
        for item in group:
            if item["case_id"] in scored_case_ids:
                by_case[item["case_id"]].append(item)
        top1, coverage, available = 0, 0, 0
        for case_id in scored_case_ids:
            samples = by_case[case_id]
            eligible = [x for x in samples if x["coordinate_contract"] and x["original"].get("pb_valid")
                        and x["original"].get("vina", {}).get("status") == "ok"]
            if eligible:
                available += 1
                selected = min(eligible, key=lambda x: x["original"]["vina"]["score"])
                top1 += bool(selected["coordinate_pose_success"])
            coverage += any(x["coordinate_pose_success"] for x in samples)
        matched_fm.append({"checkpoint": key[0], "condition": key[1], "targets": len(scored_case_ids),
                           "filtered_top1_success": top1, "any_success": coverage,
                           "targets_with_selectable_candidate": available})
    vectors, dominating = [], []
    for item in items:
        if not item["coordinate_valid_changed_fragment"]:
            continue
        metric, initial = item["original"], native[item["case_id"]]
        if metric.get("vina", {}).get("status") != "ok" or initial.get("vina", {}).get("status") != "ok":
            continue
        delta = {name: metric["properties"][name] - initial["properties"][name] for name in ("qed", "sa", "logp")}
        delta["vina"] = metric["vina"]["score"] - initial["vina"]["score"]
        candidate = {"checkpoint": item["checkpoint"], "condition": item["condition"], "case_id": item["case_id"],
                     "seed": item["seed"], "directory": item["directory"], "smiles": item["coordinate_smiles"],
                     "delta": delta, "full_pass": item.get("coordinate_full_success", False),
                     "initial": {"qed": initial["properties"]["qed"], "sa": initial["properties"]["sa"],
                                 "logp": initial["properties"]["logp"], "vina": initial["vina"]["score"]},
                     "product": {"qed": metric["properties"]["qed"], "sa": metric["properties"]["sa"],
                                 "logp": metric["properties"]["logp"], "vina": metric["vina"]["score"]}}
        vectors.append(candidate)
        directions = [delta["qed"], -delta["sa"], -delta["vina"]]
        if all(v >= -1e-8 for v in directions) and any(v > 1e-8 for v in directions):
            dominating.append(candidate)
    retained = [x for x in dominating if x["full_pass"]]
    native_full = [json.loads((root / "cases" / key / "native-full-protein-posebusters.json").read_text()) for key in cases]
    verdict = {"generation_complete": True, "attempts": len(items), "full_secondary_complete": True,
               "native_pocket_pb": sum(v["pb_valid"] for v in native.values()),
               "native_full_pb": sum(v.get("passes", False) for v in native_full),
               "docking": {"targets": len(docking), "actual_poses": sum(len(x["poses"]) for x in docking),
                           "coordinate_rank1_success": sum(x["rank1_success"] for x in docking),
                           "coordinate_filtered_top1_success": sum(x["filtered_top1_success"] for x in docking),
                           "coordinate_any_success": sum(x["any_success"] for x in docking), "records": docking},
               "matched_fm": matched_fm, "full_fragment_comparisons": comparisons,
               "measured_property_vectors": vectors, "dominates_native": dominating,
               "dominates_native_full_valid": retained,
               "distinct_dominating_target_graphs": len({(x['case_id'],x['smiles']) for x in dominating}),
               "distinct_full_dominating_target_graphs": len({(x['case_id'],x['smiles']) for x in retained})}
    write_json(output / "comparison.json", verdict)
    lines = ["# GPU 3 FLOWR 생성 품질 — 최종 검토", "", "생성 2,000회와 전체 protein 추가 검사를 완료했다. 원본 저장 결과를 읽어 분석했으며 새 GPU 생성·학습은 하지 않았다.", "",
             "## 핵심 판단", "", "현재 DIRMO 설정에서 v2는 native core를 조건으로 한 fragment 생성에서 base보다 높은 통과율을 보였다. 결합 pose를 제공하지 않는 전체 graph 고정 pose 복구는 두 checkpoint 모두 매우 낮은 성공률이었다. 유효한 새 분자가 나왔다는 사실과 QED·SA·Vina가 좋아졌다는 사실은 별도로 평가했다.", "",
             "## 실험 범위", "", "- PDBbind2019의 적격 복합체 20개, seeds 42–51, Euler 64-step, 두 checkpoint × 다섯 조건 × 200 draws = 2,000 draws / 130,000 NFE.",
             "- LLM/router 호출 0. 추가 학습 없이 기존 weights를 비교했다. checkpoint hash와 실제 GPU 3 UUID는 원본 study/health에 있다.",
             "- Pose 입력은 native 정렬 없는 ETKDG conformer를 pocket 중심에 배치했다. 초기 median RMSD는 4.403 Å, 초기 PB 통과는 0/20이다. Native pose는 evaluator만 읽었다.",
             "- Fragment는 native ligand 전체를 전처리 입력으로 사용하고 지정된 held 영역의 graph·좌표를 고정했다. 자유 영역 prior에도 reference 위치 정보가 관여하므로, core만 제공한 독립 초기화 실험으로 해석하지 않는다. 주어진 정보가 pose 과제와 달라 두 과제의 성공률을 직접 경쟁시키지 않는다.",
             "- Native pose의 PB 통과는 pocket에서 17/20, 원본 전체 protein에서 16/20이었다.", "",
             "## 저장·재읽기 문제를 구분한 분석", "", "134개 pose 출력에서 worker의 graph 통과와 SDF 재읽기 결과가 달랐다. MDL의 STEREOANY 표지가 E/Z 복원을 막는 사례를 확인했다. 원본 좌표를 그대로 둔 복사본에서 bond annotation을 지우고 atom/bond stereo를 XYZ로 다시 계산하면 134건 모두 worker의 graph와 일치했다. Native 20개에도 이 방법을 적용해 모두 원래 graph와 일치함을 확인했다.", "",
             "원래 metric과 SDF는 변경하지 않았다. 재검사는 별도 JSONL로 남겼다. Pose 성공 6건이라는 결론은 바뀌지 않았다. Fragment에서는 30개가 실제 화학 변화 없이 E/Z annotation 누락만으로 새 graph로 집계됐으므로 다음 표는 이를 제외한다.", "",
             "## Pose 결과", "", "성공 = 좌표에서 복원한 exact graph + pocket PB 통과 + protein frame RMSD ≤2 Å. 각 행은 200회다.", "",
             "| 조건 | base | v2 |", "| --- | ---: | ---: |"]
    for condition in ("random", "reference0", "reference03"):
        left = next(g for g in audit['groups'] if g['checkpoint']=='base' and g['condition']==condition)
        right = next(g for g in audit['groups'] if g['checkpoint']=='v2' and g['condition']==condition)
        lines.append(f"| {condition} | {left['coordinate_pose_success']}/200 | {right['coordinate_pose_success']}/200 |")
    lines.extend(["", "base reference03의 6건은 모두 1pot 한 target에서 나왔다. 전체 protein 검사에서도 이 6건은 통과했다. Seed 6개를 서로 다른 target 6개로 해석하지 않는다. Reference0은 모든 target에서 10 seeds가 동일 좌표 1개를 반환해 추가 다양성을 제공하지 않았다.", "",
                  "Gaussian 조건에서는 400개 모두 결합 길이·각 검사에 실패했다. V2 pose 조건은 600개 모두 protein 접촉 검사에 실패했다. NFE와 raw-prior 좌표 차이를 확인해 모델 forward가 실제 실행되고 좌표도 바뀌었음을 확인했다.", "",
                  "이 결과는 잘못 배치한 초기 conformer의 전역 복구 결과다. 기존 bound pose 주변의 작은 보정이나 편집 후 retained-atom alignment를 하는 production 경로의 성능을 직접 측정한 결과가 아니다. Whole-graph 조건의 학습 이력은 원본 metadata만으로 미확인이다.", "",
                  "## Fragment 결과", "", "성공 = held atom/charge/bond·좌표 보존 + PB 통과 + 좌표 stereo를 복원한 실제 graph 변화. 각 행은 200회다. 마지막 열은 원본 protein의 물·보조인자 좌표까지 고정해 검사한 추가 기준이다.", "",
                  "| checkpoint | 자유 영역 | 원래 집계 | 실제 변화 확인 후 | 전체 protein 추가 통과 |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for g in audit["groups"]:
        if not g['condition'].startswith('fragment'):
            continue
        lines.append(f"| {g['checkpoint']} | {g['condition']} | {g['changed_valid_fragment']}/200 | {g['coordinate_changed_valid_fragment']}/200 ({g['coordinate_changed_valid_fragment']/2:.1f}%) | {g['coordinate_full_success']}/200 ({g['coordinate_full_success']/2:.1f}%) |")
    lines.extend(["", "편집 영역은 metric이나 PLIP 결과를 보고 선택하지 않았다. fragment_regions는 ring이 아닌 결합을 잘라 작은 말단 조각을 고르고, 큰 영역은 그 조각에서 이웃 원자로 확장한다. 동률은 원자 index로 결정하며 동일 case의 모든 seeds/checkpoints에서 같은 영역을 사용한다. Murcko scaffold는 복합체 중복 제거에만 사용했다. QED·SA·Vina 목표 또는 개선 방향은 모델에 입력하지 않았다.", ""])
    if prior_audit is not None:
        t = prior_audit['totals']
        lines.extend([f"추가 입력 감사에서 {t['draws']}개 fragment의 최종 held mask는 요청과 일치했다. 그러나 {t['draws_with_free_at_same_native_slot']}회에서 자유 원자의 시작 좌표 일부가 같은 원자의 native 좌표와 0.001 Å 이내로 일치했고, {t['draws_with_free_at_any_native_atom']}회에서는 적어도 하나가 reference의 어느 원자 위치와 일치했다. 총 자유 슬롯 {t['free_slots']}개 중 각각 {t['free_at_same_native_slot']}개와 {t['free_at_any_native_atom']}개다. 후자는 서로 다른 원자 slot으로 복사된 위치도 포함하므로 해당 자유 원자의 정답 pose가 전달됐다는 뜻으로 모두 해석하지 않는다.", "",
                      "Upstream은 자체 fragment mask로 fixed/variable fragment를 구성하고 local prior를 reference의 자유 영역 중심에 배치한다. DIRMO는 이후 요청한 held mask와 held features를 덮어쓰지만, 모든 자유 feature를 독립 prior로 재초기화하지 않는다. 따라서 최종 고정 영역은 맞아도 자유 초기화는 완전히 통제되지 않았다. 기록된 시작 좌표에서 이 효과를 확인했으며 자유 discrete features는 저장되지 않아 사후에 직접 판정하지 않았다. 기존 생성 통과 수는 유지하되, 이를 엄밀한 core-only 조건부 생성 성능이나 모델 능력만의 차이로 해석하지 않는다. 증거는 fragment-prior-audit.json에 있다.", ""])
    lines.extend(["", "좌표 복원 후의 v2-base 차이는 small +14.5%p (target bootstrap 95% CI +4.5~+25.0%p), large +10.5%p (+0.5~+22.5%p)다. 전체 protein 추가 기준에서는 small +9.5%p (95% CI −1.5~+21.0%p), large +7.0%p (−5.0~+20.0%p)다. 추가 기준의 신뢰구간은 0을 포함하므로, 이 20-target 실험만으로 v2의 일반적인 우월성이나 small 영역의 최적성을 확정하지 않는다. 계산값은 comparison.json에 저장했다.", "",
                  "평가 가능한 745 fragment 출력에서 held 좌표 drift는 모두 0 Å였다. Held graph 위반은 base small/large 각각 3개, v2 small 1개, v2 large 0개였다. 좌표 고정은 sampler의 hard copy 효과도 포함하므로 모델 학습 능력의 단독 증거가 아니다.", "",
                  "Full PB에서 통과가 줄어든 주된 추가 원인은 water/protein 접촉이다. 물의 위치를 고정한 판정이며, 실제 binding에서 물이 이동·배출되는 효과를 모사한 결과는 아니다. Missing energy 계산을 통과로 바꾸지 않았다.", "",
                  "## 동일 receptor subset의 docking 비교", "",
                  f"Receptor 준비에 성공한 {len(docking)} targets에서 Vina docking 실제 poses는 {verdict['docking']['actual_poses']}개였다. 좌표 stereo를 복원한 rank-1 성공은 {verdict['docking']['coordinate_rank1_success']}/{len(docking)}, PB/graph 통과 후 Vina 최소값으로 고르는 공통 selector는 {verdict['docking']['coordinate_filtered_top1_success']}/{len(docking)}, oracle any-pose 성공은 {verdict['docking']['coordinate_any_success']}/{len(docking)}이다.", "",
                  "| FM 조건 | 공통 selector Top-1 | oracle any draw |", "| --- | ---: | ---: |"])
    for g in matched_fm:
        lines.append(f"| {g['checkpoint']} {g['condition']} | {g['filtered_top1_success']}/{g['targets']} | {g['any_success']}/{g['targets']} |")
    lines.extend(["", "이 비교는 동일 11 targets의 pocket PB 기준이며 full-protein FM 결과와 혼합하지 않는다. 후보 수와 계산 비용이 정확히 같지는 않다. 준비 실패한 나머지 9개는 숨기거나 FM 집계에서 제외하지 않았다.", "",
                  "## 성질 개선", "",
                  f"실제 새 graph이며 조건/PB를 통과하고 Vina까지 측정된 후보 중 QED↑·SA↓·Vina↓로 native를 지배한 것은 {len(dominating)} draws, {len({x['case_id'] for x in dominating})} targets, {verdict['distinct_dominating_target_graphs']}개의 target/graph 조합이다. Full-protein 추가 통과는 {len(retained)} draws / {verdict['distinct_full_dominating_target_graphs']} target/graph 조합이다.", "",
                  "| checkpoint | 조건 | 유효 새 graph | Vina까지 측정 | 세 지표 native 지배 |",
                  "| --- | --- | ---: | ---: | ---: |"])
    for cp in ('base','v2'):
        for condition in ('fragment_small','fragment_large'):
            group = groups[(cp,condition)]
            v = [x for x in vectors if x['checkpoint']==cp and x['condition']==condition]
            d = [x for x in dominating if x['checkpoint']==cp and x['condition']==condition]
            lines.append(f"| {cp} | {condition} | {sum(x['coordinate_valid_changed_fragment'] for x in group)} | {len(v)} | {len(d)} |")
    lines.extend(["", "V2 small 후보의 measured median 변화는 QED +0.0311, SA +0.1802, Vina +0.390이다. SA/Vina는 낮을수록 좋으므로 유효 생성의 상당수에서 일부 목적은 악화됐다. 높은 생성 통과율이 다목적 최적화 성공을 뜻하지 않는다.", ""])
    example = next(x for x in retained if x['case_id'] == '1e6s' and x['checkpoint'] == 'v2'
                   and x['condition'] == 'fragment_small' and x['seed'] == 45)
    lines.extend(["대표 개선 사례는 1e6s, v2 fragment_small, seed 45다. 전체 protein 추가 검사도 통과했다.", "",
                  "| 지표 | 초기 ligand | 생성 ligand | 해석 |", "| --- | ---: | ---: | --- |"])
    for name, label, interpretation in (("qed", "QED", "높을수록 좋음"),
                                        ("sa", "SA", "낮을수록 좋음"),
                                        ("vina", "Vina score", "낮을수록 좋음"),
                                        ("logp", "LogP", "목표 범위에 따라 판단")):
        lines.append(f"| {label} | {example['initial'][name]:.3f} | {example['product'][name]:.3f} | {interpretation} |")
    lines.extend(["", f"생성 SDF: [{example['directory']}/raw.sdf]({example['directory']}/raw.sdf)", "",
                  "Vina는 계산 점수이며 실험으로 측정한 결합 친화도가 아니다. LogP 감소를 자동으로 개선이라고 판정하지 않았다. 세 지표의 동시 개선 판정에는 QED·SA·Vina만 사용했다.", "",
                  "## 다음 단계", "", "1. SDF E/Z roundtrip 및 fragment 자유 영역의 prior 초기화·mask 경로를 수정·검증한다. Metadata 차이를 화학 편집 학습 label로 사용하지 않고, 요청한 영역의 자유 초기화와 held 보존을 함께 확인한다.",
                  "2. Native core가 있는 지역 탐색에는 v2 fragment를 우선 사용하고 실제 구조 변화·held graph·물리 검사 및 각 metric을 모두 검증한다.",
                  "3. 전체 graph 고정 pose는 입력 prior·time embedding·self-conditioning·stereo 조건의 학습 preset 호환성을 확인한 뒤 같은 graph의 noisy-pose fine-tuning 필요성을 판단한다. 현재 낮은 성공률만으로 LLM tuning에 원인을 돌리지 않는다.",
                  "4. 실제 목적 개선은 metric vector와 Pareto 후보로 평가하고, 별도 target/scaffold 분리 데이터로 조건 선택을 확인한다. Native를 지배한 후보는 Pareto front와 동일한 개념이 아니다.", "",
                  "## 해석 한계와 재현", "", "20 targets의 pilot이며 pretraining overlap은 미확인이다. Protein exact sequence와 non-empty Murcko scaffold를 중복 제거했지만 sequence-cluster split을 검증한 것은 아니다. Scalar utility, 실제 affinity, 전체 optimize 정책의 우월성을 측정하지 않았다. 두 FM 프로세스와 기존 작업이 GPU 3을 공유했으므로 latency는 단독 GPU benchmark가 아니다.", "",
                  "원본의 summary/analysis와 새 검토 결과는 구분해 보관했다. 원본 리포트는 기존 stereo 판정 기준의 집계이며, 본 검토는 좌표 복원으로 교차 확인한 결과다.", "",
                  "```bash", "conda activate dirmo", "cd ~/code/DIRMO-3D",
                  "python scripts/review_fm_quality.py --run runs/fm_quality_gpu3_20261013_v1 --output reports/fm-quality-review-2026-10-13",
                  "python scripts/audit_fm_fragment_conditions.py --run runs/fm_quality_gpu3_20261013_v1 --output reports/fm-quality-review-2026-10-13/fragment-prior-audit.json",
                  "python scripts/build_fm_quality_review.py --run runs/fm_quality_gpu3_20261013_v1 --output reports/fm-quality-review-2026-10-13", "```", "",
                  "![조건별 품질](quality_review.png)", "",
                  "막대는 pocket 기준의 성공률, 검은 가로 표식은 전체 protein 추가 검사까지 통과한 성공률이다. 두 패널의 세로축 범위가 다르다. Pose와 fragment는 제공받은 조건이 달라 성공률을 직접 비교하지 않는다."])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for index, cp in enumerate(('base','v2')):
        pose = [next(g['coordinate_pose_success']/2 for g in audit['groups'] if g['checkpoint']==cp and g['condition']==c)
                for c in ('random','reference0','reference03')]
        axes[0].bar(np.arange(3)+(index-.5)*.35,pose,width=.35,label=cp)
        fragments = [next(g['coordinate_changed_valid_fragment']/2 for g in audit['groups'] if g['checkpoint']==cp and g['condition']==c)
                     for c in ('fragment_small','fragment_large')]
        full = [next(g['coordinate_full_success']/2 for g in audit['groups'] if g['checkpoint']==cp and g['condition']==c)
                for c in ('fragment_small','fragment_large')]
        x=np.arange(2)+(index-.5)*.35
        axes[1].bar(x,fragments,width=.35,label=cp)
        axes[1].scatter(x,full,color='black',marker='_',s=250,zorder=4,
                        label='Full-protein secondary pass' if index == 0 else None)
    axes[0].set_xticks(np.arange(3),['Gaussian','Reference 0','Reference 0.3'])
    axes[0].set_ylim(0,5)
    axes[0].set_title('Pose recovery (known pocket, no native pose)')
    axes[1].set_xticks(np.arange(2),['Small free region','Larger free region'])
    axes[1].set_ylim(0,50)
    axes[1].set_title('Genuine new graph with native held core')
    for axis in axes:
        axis.set_ylabel('Successful scheduled draws (%)')
        axis.legend()
        axis.grid(axis='y',alpha=.2)
        axis.set_axisbelow(True)
    fig.savefig(output/'quality_review.png',dpi=180)
    plt.close(fig)
    print(json.dumps({'status':'complete','report':str(output/'REPORT.md'),'genuine_dominating_draws':len(dominating),
                      'full_dominating_draws':len(retained),'matched_docking_rank1':verdict['docking']['coordinate_rank1_success']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.run.resolve(),args.output.resolve())
