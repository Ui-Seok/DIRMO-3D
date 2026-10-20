"""Render an auditable report of the frozen-LLM validation comparison."""
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

from dirmo.evaluation.smdd_feedback import select_attempt, tool_feedback
from dirmo.io import write_json


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/smdd_feedback_01"
OUT = Path(__file__).resolve().parent


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def main():
    path = RUN / "report_with_boltz.json"
    if not path.exists():
        path = RUN / "report.json"
    report = json.loads(path.read_text())
    attempts = lines(RUN / "attempts.jsonl")
    calls = {r["request_id"]: r for r in lines(RUN / "llm_calls.jsonl")}
    protocol = json.loads((RUN / "protocol.json").read_text())
    n, rounds = report["cases"], protocol["rounds"]
    expected = {(task, mode, r) for task in protocol["task_ids"] for mode in ["independent", "feedback"]
                for r in range(1, rounds + 1)}
    actual = {(r["task_id"], r["mode"], r["round"]) for r in attempts}
    if expected != actual or len(attempts) != len(expected) or len(calls) != n * (2 * rounds - 1):
        raise ValueError("Incomplete, duplicate, or unexpected proposal coverage")
    diagnostics = {}
    per_target = {}
    rescues, examples = [], []
    objective_types = {}
    for mode in ["independent", "feedback"]:
        items = [r for r in attempts if r["mode"] == mode]
        failures = Counter()
        unique = []
        for item in items:
            for failed in tool_feedback(item["assessment"], [])['failed_conditions']:
                name = failed.get("condition", failed.get("property", failed.get("reason")))
                failures[f"{failed['stage']}:{name}"] += 1
        for task in protocol["task_ids"]:
            group = [r for r in items if r["task_id"] == task]
            unique.append(len({r["assessment"]["product_smiles"] for r in group
                               if r["assessment"].get("smiles_valid")}))
        diagnostics[mode] = {
            "failed_conditions": dict(failures.most_common()),
            "copy_original_attempts": sum(r["assessment"].get("smiles_valid") and
                                          not r["assessment"].get("changed") for r in items),
            "mean_unique_valid_graphs_per_case": sum(unique) / n,
            "token_cost_including_shared_first": sum(calls[r["request_id"]]["response"].get("total_tokens") or 0
                                                       for r in items),
        }
        selected = [r for r in report["selected"] if r["mode"] == mode]
        objective_types[mode] = {
            "pre_boltz_pass_with_admet_goal": sum(r["assessment"]["pre_boltz_passed"] and
                r["assessment"].get("admet_conditions", {}).get("admet_goal_count", 0) > 0 for r in selected),
            "pre_boltz_pass_with_binding_only_goal": sum(r["assessment"]["pre_boltz_passed"] and
                r["assessment"].get("admet_conditions", {}).get("admet_goal_count", 0) == 0 for r in selected),
        }
        for target in sorted({r["task_id"].split("_")[2] for r in items}):
            selected = [r for r in report["selected"] if r["mode"] == mode and r["task_id"].split("_")[2] == target]
            per_target.setdefault(target, {})[mode] = {
                "cases": len(selected),
                "pre_boltz_pass": sum(r["assessment"]["pre_boltz_passed"] for r in selected),
                "full_success": sum(r["assessment"]["full_benchmark_success"] is True for r in selected),
                "binding_unmeasured": sum(r["assessment"]["full_benchmark_success"] is None for r in selected),
            }
    for task in protocol["task_ids"]:
        feedback = [r for r in attempts if r["mode"] == "feedback" and r["task_id"] == task]
        first = feedback[0]
        selected = select_attempt(feedback)
        control = select_attempt([r for r in attempts if r["mode"] == "independent" and r["task_id"] == task])
        improved = selected["assessment"]["pre_boltz_passed"] and not first["assessment"]["pre_boltz_passed"]
        if improved:
            rescues.append({"task_id": task, "round": selected["round"],
                            "control_pre_boltz_pass": control["assessment"]["pre_boltz_passed"]})
        def failed_admet(item):
            checked = item["assessment"].get("admet_conditions")
            return (sum(v["passed"] is False for k in ["goals", "holds"] for v in checked[k])
                    if checked else None)
        partial = (failed_admet(first) is not None and failed_admet(selected) is not None
                   and failed_admet(selected) < failed_admet(first))
        local_repair = (not first["assessment"].get("local_hard_constraints_passed") and
                        selected["assessment"].get("local_hard_constraints_passed") and
                        selected["assessment"].get("changed"))
        if improved or local_repair or partial:
            examples.append({"task_id": task, "pre_boltz_rescue": improved,
                             "admet_failure_counts": [failed_admet(first), failed_admet(selected)],
                             "first": first, "selected": selected,
                             "selected_call": calls[selected["request_id"]]})
    examples.sort(key=lambda e: (not e["pre_boltz_rescue"],
                                any(v is None for v in e["admet_failure_counts"]), e["task_id"]))
    write_json(OUT / "summary.json", {"run_report": str(path), "comparison": report["comparison"],
                                      "diagnostics": diagnostics, "per_target": per_target,
                                      "rescues": rescues, "tokens": report["tokens"],
                                      "pre_boltz_objective_types": objective_types})
    write_json(OUT / "examples.json", examples)
    demonstration = examples[0] if examples else {"task_id": protocol["task_ids"][0],
         "selected_call": calls[next(iter(calls))]}
    write_json(OUT / "example_input.json", demonstration["selected_call"]["messages"])
    write_json(OUT / "example_output.json", demonstration["selected_call"]["response"])
    successful = [s for s in report["selected"] if s["assessment"]["full_benchmark_success"] is True]
    unique_success = {}
    for selection in successful:
        key = (selection["task_id"], selection["assessment"]["oracle_smiles"])
        entry = unique_success.setdefault(key, {"task_id": selection["task_id"],
            "product_smiles": selection["assessment"]["oracle_smiles"],
            "modes": [], "round_by_mode": {}, "assessment": selection["assessment"]})
        entry["modes"].append(selection["mode"])
        entry["round_by_mode"][selection["mode"]] = selection["round"]
    write_json(OUT / "successful_predictions.json", list(unique_success.values()))
    if successful:
        chosen = successful[0]
        call = calls[next(r["request_id"] for r in attempts if r["task_id"] == chosen["task_id"]
                          and r["mode"] == chosen["mode"] and r["round"] == chosen["round"])]
        write_json(OUT / "successful_input.json", call["messages"])
        write_json(OUT / "successful_output.json", call["response"])
        content = json.loads(call["messages"][1]["content"])
        molecules = [Chem.MolFromSmiles(content["original_molecule"]["smiles"]),
                     Chem.MolFromSmiles(chosen["assessment"]["oracle_smiles"])]
        Draw.MolsToGridImage(molecules, molsPerRow=2, subImgSize=(500, 320),
                            legends=["Original ligand", "Proposed ligand"],
                            useSVG=False).save(OUT / "successful_pair.png")
    initial = report["comparison"]["independent"]["first_proposal_pre_boltz_pass"]
    independent = report["comparison"]["independent"]
    feedback = report["comparison"]["feedback"]
    labels = ["First proposal", "3 independent proposals", "3 proposals with feedback"]
    values = [initial, independent["selected_pre_boltz_pass"], feedback["selected_pre_boltz_pass"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#8a94a6", "#408bc4", "#27946e"])
    ax.set_ylim(0, n)
    ax.set_ylabel(f"Cases passing RDKit + ADMET (of {n})")
    ax.set_title("Frozen Qwen3.5-9B — SMDD validation; binding checked separately")
    ax.bar_label(bars)
    fig.tight_layout()
    fig.savefig(OUT / "pre_boltz_comparison.png", dpi=160)
    plt.close(fig)
    text = ["# 학습 없는 편집·도구 피드백 validation 결과", "",
            f"Qwen3.5-9B base 모델로 **validation {n}개 전체**를 실행했다. LoRA, encoder, projector, "
            "가중치 업데이트는 사용하지 않았다. Train/test와 witness 정답을 입력에 사용하지 않았다.", "",
            "Original과 current 분자 물성·원자/작용기 정보를 구조화해 전달하고, 한 문장 편집 가설과 "
            "전체 SMILES를 받았다. RDKit·ADMET 실패값을 다음 제안에 넣어 수정했다.", "",
            f"과제당 방식별 {rounds}개 제안을 사용했다. 첫 제안은 공유했고 실제 HTTP 요청은 "
            f"**{len(calls)}회**, 평가 기록은 **{len(attempts)}개**다. 서버는 thinking=false, seed=42, "
            "temperature=0.7, top_p=0.8, top_k=20, BF16이며 physical GPU 4를 사용했다. "
            "ADMET-AI 1.4.0은 격리된 CPU 환경에서 계산했다.", "",
            "## 조건 통과 결과", "",
            "| 과제 단위 지표 | 피드백 없는 3회 제안 | 도구 피드백 3회 제안 |", "|---|---:|---:|"]
    for label, key in [("최초 제안 RDKit·ADMET 통과", "first_proposal_pre_boltz_pass"),
                       ("유효 graph가 한 번 이상 생성됨", "any_valid_graph"),
                       ("변경된 ligand의 로컬 제약 통과가 한 번 이상 있음", "any_changed_local_pass"),
                       ("선택된 후보 RDKit·ADMET 통과", "selected_pre_boltz_pass"),
                       ("Boltz 포함 전체 성공 확인", "full_success_confirmed"),
                       ("선택된 후보의 결합 조건 미측정", "unmeasured_binding")]:
        text.append(f"| {label} | {independent[key]}/{n} | {feedback[key]}/{n} |")
    text.extend(["", "RDKit·ADMET 통과는 전체 성공이 아니다. 전체 성공에는 Boltz binding probability > 0.7과 "
                 "해당 task의 affinity 목표/유지 조건이 추가로 필요하다. 미측정은 실패로 확정하거나 성공으로 계산하지 않는다.", "",
                 "![RDKit·ADMET 비교](pre_boltz_comparison.png)", "",
                 "| 제안 단위 지표 | 피드백 없음 | 도구 피드백 |", "|---|---:|---:|"])
    for label, key in [("정확한 편집 JSON 형식", "plan_valid_attempts"), ("유효한 분자 graph", "valid_graph_attempts"),
                       ("변경 및 로컬 제약 통과", "changed_local_pass_attempts"), ("ADMET 검사 시행", "admet_evaluated_attempts")]:
        text.append(f"| {label} | {independent[key]}/{n*rounds} | {feedback[key]}/{n*rounds} |")
    text.extend(["", "## 결과 해석", "",
                 f"처음에는 RDKit·ADMET를 통과하지 못했지만 feedback 수정 후 통과한 과제는 **{len(rescues)}개**다. "
                 f"최초 제안의 통과 수는 {initial}개, 3회 independent의 통과 수는 {independent['selected_pre_boltz_pass']}개, "
                 f"feedback의 통과 수는 {feedback['selected_pre_boltz_pass']}개다. "
                 "같은 호출 수에서의 관측이며 통계적 유의성이나 일반적인 성능 우위를 보장하지 않는다.", "",
                 f"과제당 유효 graph의 평균 종류 수는 independent {diagnostics['independent']['mean_unique_valid_graphs_per_case']:.2f}, "
                 f"feedback {diagnostics['feedback']['mean_unique_valid_graphs_per_case']:.2f}다. "
                 "Worker가 요청마다 같은 seed를 쓰므로 independent는 제안 순서만 바꿔도 같은 분자를 반복할 수 있다. "
                 "다양한 sampling seed를 사용한 독립 후보 생성과의 비교는 별도 실험이 필요하다.", "",
                 "Feedback 제공과 current ligand 변경을 함께 적용했으므로 두 효과를 분리하지 않았다. "
                 "편집 위치·가설의 타당성을 label로 평가한 실험도 아니다. 실제 도구로 계산한 조건의 통과 여부를 평가했다.", "",
                 f"RDKit·ADMET 통과 후보 중 ADMET 개선 목표가 있는 과제는 independent "
                 f"{objective_types['independent']['pre_boltz_pass_with_admet_goal']}개, feedback "
                 f"{objective_types['feedback']['pre_boltz_pass_with_admet_goal']}개다. Binding만 개선 목표인 과제는 각각 "
                 f"{objective_types['independent']['pre_boltz_pass_with_binding_only_goal']}개, "
                 f"{objective_types['feedback']['pre_boltz_pass_with_binding_only_goal']}개다. "
                 "후자는 ADMET 유지 조건을 통과했을 뿐 개선 목표인 affinity를 확인한 수치가 아니다.", "",
                 "### Target별 선택된 후보", "", "| Target | 개수 | Independent pre-Boltz | Feedback pre-Boltz |", "|---|---:|---:|---:|"])
    for target, modes in per_target.items():
        text.append(f"| {target} | {modes['feedback']['cases']} | {modes['independent']['pre_boltz_pass']} | {modes['feedback']['pre_boltz_pass']} |")
    text.extend(["", "### 많이 실패한 조건", ""])
    for mode in ["independent", "feedback"]:
        listed = ", ".join(f"`{k}` {v}회" for k, v in list(diagnostics[mode]["failed_conditions"].items())[:6])
        text.append(f"- {mode}: {listed}. 한 제안에서 여러 조건이 실패할 수 있다.")
    if (RUN / "boltz/report.json").exists():
        boltz = json.loads((RUN / "boltz/report.json").read_text())
        deferred = [r for r in boltz["results"] if r["status"] != "measured"]
        text.extend(["", "## Boltz 검증", "",
                     f"선택된 후보의 중복을 제거한 요청 {len(boltz['results'])}개 중 "
                     f"{len(boltz['results']) - len(deferred)}개를 측정했고 {len(deferred)}개는 미측정이다. "
                     "Seed 42, sampling steps 200, structure/affinity 각 10 draws를 사용했다.", ""])
        text.extend(["이번 실행은 `--memory-fraction 0.24 --min-free-mib 13000 --max-parallel-samples 1`을 "
                     "사용했다. 초기 GPU context 생성 후 여유가 약 13,448 MiB였으며 작은 protein 입력은 "
                     "프로세스 상한과 1 GiB의 여유 예약 내에서 실행했다. 400 residue 이상 입력은 이전 BRAF OOM "
                     "기록을 반영한 보수적 조건인 30,000 MiB 여유를 만족하지 못해 실행하지 않았다.", ""])
        for result in boltz["results"]:
            checked = result.get("binding_conditions")
            if checked:
                text.append(f"- `{result['task_id']}` ({', '.join(s['mode'] for s in result['selections'])}): "
                            f"binding probability {checked['binding_probability']:.4f}, "
                            f"affinity improvement {checked['affinity_improvement']:.4f}, "
                            f"전체 통과 `{result['full_benchmark_success']}`.")
                if result["full_benchmark_success"]:
                    selected = next(s for s in successful if s["task_id"] == result["task_id"])
                    goals = selected["assessment"]["admet_conditions"]["goals"]
                    for goal in goals:
                        if goal["passed"] is not None:
                            text.append(f"\n`{goal['property']}`: {goal['baseline']:.4f} → {goal['value']:.4f}, "
                                        f"개선량 {goal['improvement']:.4f} ≥ 목표 {goal['threshold']:.4f}.")
                    if config := result["config"].get("affinity_hold_constant"):
                        change = -checked['affinity_improvement']
                        description = (f"악화량 {change:.4f}가 유지 허용치 {config['tolerance']:.4f} 이내다. "
                                       if change > 0 else f"{abs(change):.4f} 개선되어 유지 조건도 통과했다. ")
                        text.append(f"\nAffinity는 {checked['baseline_affinity']:.4f} → {checked['affinity_pred_value']:.4f}로 "
                                    + description +
                                    "이 과제는 affinity 개선을 요구하는 대신 유지 조건을 적용한다.")
                    text.append("")
            else:
                text.append(f"- `{result['task_id']}`: 미측정, `{result.get('reason', 'oracle execution failed')}`. "
                            "세부 실행 로그와 자원 한계는 run/boltz/results.json 참조.")
    if examples:
        text.extend(["", "## 수정 사례", ""])
        for example in examples[:3]:
            selected = example["selected"]
            a = selected["assessment"]
            first_count, selected_count = example['admet_failure_counts']
            admet_description = (f"ADMET 실패 조건 수 {first_count} → {selected_count}." if first_count is not None
                                 else f"초기 ADMET 미측정(로컬 탈락); 수정 후 ADMET 실패 조건 {selected_count}개.")
            text.extend([
                f"- `{example['task_id']}`: round {selected['round']} 후보 선택, "
                f"RDKit·ADMET 전체 통과 `{a['pre_boltz_passed']}`. "
                + admet_description,
                "", f"LLM 편집 가설: {a.get('plan', {}).get('edit_hypothesis', '')}",
                "", f"생성 SMILES: `{a.get('oracle_smiles', '')}`", "",
            ])
        text.append("\n가설은 LLM 출력 그대로이며 원인 설명의 정확성은 검증하지 않았다. 일부 조건 개선과 전체 통과는 다르다.")
    if successful:
        text.extend(["", "![전체 조건 통과 후보와 original의 2D 구조](successful_pair.png)", "",
                     f"위 2D 구조는 `{successful[0]['task_id']}`의 original과 모델 제안이다. "
                     "Boltz 3D 결과와 구분되는 graph 도식이며 설명한 편집 위치의 정확성을 자동 검증한 그림은 아니다.", ""])
    text.extend(["", "## 비용·재현과 파일", "",
                 f"총 input token {report['tokens']['input_tokens']:,}, output token {report['tokens']['output_tokens']:,}, "
                 f"model generation 시간 합계 {report['generation_seconds']/60:.1f}분이다. "
                 "공유된 첫 제안은 실제 요청/전체 token에서 중복 계산하지 않았다.", "",
                 "Oracle의 ADMET 값은 예측값이며 실험 측정이 아니다. 공식 baseline은 그대로 사용했다. "
                 "Reference 재예측 차이는 `reference_audit.json`에 기록했다. 같은 graph라도 원래 SMILES 표기에 따른 "
                 "legacy Chemprop 차이가 가능하므로 예측 숫자 변화만으로 분자 개선의 원인을 단정하지 않는다.", "",
                 "- [실행 및 prompt 코드 안내](../../docs/smdd_feedback.md)",
                 "- [집계 JSON](summary.json), [수정 사례](examples.json)",
                 "- [전체 통과 확인된 출력](successful_predictions.json)",
                 "- [실제 입력 예시](example_input.json), [실제 출력 예시](example_output.json)",
                 "- [프로토콜](../../runs/smdd_feedback_01/protocol.json)",
                 "- [모든 HTTP 입력·출력](../../runs/smdd_feedback_01/llm_calls.jsonl)",
                 "- [모든 도구 검사](../../runs/smdd_feedback_01/attempts.jsonl)",
                 "- [Run 결과](../../runs/smdd_feedback_01/report.json)", ""])
    (OUT / "REPORT.md").write_text("\n".join(text))
    write_json(RUN / "verification.json", {"status": "passed", "coverage": len(expected),
            "actual_requests": len(calls), "no_duplicate_attempts": len(attempts) == len(actual),
            "original_dataset_sha256": protocol["dataset_sha256"], "train_test_used": False,
            "known_full_success": {m: c["full_success_confirmed"] for m, c in report["comparison"].items()},
            "binding_unmeasured": {m: c["unmeasured_binding"] for m, c in report["comparison"].items()}})
    print(json.dumps({"report": str(OUT / "REPORT.md"), "comparison": report["comparison"],
                      "rescues": len(rescues)}, indent=2))


if __name__ == "__main__":
    main()
