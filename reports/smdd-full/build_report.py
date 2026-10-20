"""Build full split tables and epoch plots with the existing dirmo plotting env."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def loss_text(value):
    return "미측정" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/smdd_full_01/experiment"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    report = json.loads((args.run / "report.json").read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    last_path = args.run / "final/last_adapter/report.json"
    if last_path.exists():
        last = json.loads(last_path.read_text())
        if last["status"] != "complete":
            raise ValueError("Last checkpoint evaluation is incomplete")
        for split, result in last["splits"].items():
            report["final"][split]["last_adapter"] = result
    model_names = [name for name in ["base", "adapter", "last_adapter"] if name in report["final"]["test"]]
    history = report["history"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ["train", "validation", "test"]:
        for name in model_names:
            item = report["final"][split][name]
            strict = item["strict"]["counts"]
            diagnostic = item["fence_removed_diagnostic"]["counts"]
            rows.append({"split": split, "model": name, "rows": item["rows"],
                         "completion_loss": item["completion_loss"], "strict_json": strict["json_valid"],
                         "valid_smiles_after_fence_removal": diagnostic["smiles_valid"],
                         "local_passed_after_fence_removal": diagnostic["local_hard_constraints_passed"],
                         "changed_and_local_passed": diagnostic["changed_and_local_passed"],
                         "witness_exact_match": diagnostic["witness_exact_match"], "hit_token_cap": item["hit_token_cap"]})
    with (args.output / "split_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    epochs = [h["total_epoch"] for h in history]
    axes[0].plot(epochs, [h["validation"]["completion_loss"] for h in history], marker="o", label="Validation completion loss")
    axes[0].axvline(report["best_total_epoch"], color="tab:gray", linestyle="--", label="Selected checkpoint")
    axes[0].set(xlabel="Total training epoch", ylabel="Cross entropy", title="Full validation loss", xticks=epochs)
    axes[0].legend(fontsize=8)
    for field, label in [("smiles_valid", "Valid SMILES"), ("changed_and_local_passed", "Changed + local constraints")]:
        values = [h["validation"]["fence_removed_diagnostic"]["rates"][field] * 100 for h in history]
        axes[1].plot(epochs, values, marker="o", label=label)
    axes[1].set(xlabel="Total training epoch", ylabel="Percent of all validation rows", ylim=(0, 100),
                title="Full validation generation", xticks=epochs)
    axes[1].legend(fontsize=8)
    bars = [(-0.25, "base", "tab:orange"), (0, "adapter", "tab:green")]
    if "last_adapter" in model_names:
        bars.append((0.25, "last_adapter", "tab:blue"))
    for shift, name, color in bars:
        values = [report["final"][s][name]["fence_removed_diagnostic"]["rates"]["changed_and_local_passed"] * 100
                  for s in ["train", "validation", "test"]]
        axes[2].bar([i + shift for i in range(3)], values, width=0.25, color=color, label=name)
    axes[2].set(xticks=range(3), xticklabels=["Train (seen)", "Validation", "Test"],
                ylabel="Percent of rows", title="Changed + local constraints")
    axes[2].legend(fontsize=8)
    figure.savefig(args.output / "epochs_and_splits.png", dpi=170)
    plt.close(figure)
    epoch_lines = []
    for h in history:
        counts = h["validation"]["fence_removed_diagnostic"]["counts"]
        loss = h["validation"]["completion_loss"]
        lr = f"{h['learning_rate']:.7g}" if "learning_rate" in h else "이전 실행"
        epoch_lines.append(f"| {h['total_epoch']} | {lr} | {loss_text(loss)} | {counts['smiles_valid']}/34 | {counts['changed_and_local_passed']}/34 |")
    trial_lines = []
    for trial in report["lr_trials"]:
        item = trial["validation"]
        counts = item["fence_removed_diagnostic"]["counts"]
        trial_lines.append(f"| {trial['learning_rate']:g} | {loss_text(item['completion_loss'])} | {counts['smiles_valid']}/34 | {counts['changed_and_local_passed']}/34 |")
    split_lines = []
    for row in rows:
        n = row["rows"]
        split_lines.append(f"| {row['split']} | {row['model']} | {n} | {loss_text(row['completion_loss'])} | {row['strict_json']}/{n} | {row['valid_smiles_after_fence_removal']}/{n} | {row['changed_and_local_passed']}/{n} |")
    target_lines = []
    target_metrics = []
    split_targets = {}
    for split in ["train", "validation", "test"]:
        for name in model_names:
            records = [json.loads(line) for line in (args.run / f"final/{name}/{split}/predictions.jsonl").read_text().splitlines()]
            groups = defaultdict(list)
            for record in records:
                target = record["task_id"].removeprefix("smdd_004_").rsplit("_", 1)[0]
                groups[target].append(record["fence_removed_diagnostic"])
            split_targets[split] = {target: len(items) for target, items in sorted(groups.items())}
            for target, items in sorted(groups.items()):
                entry = {"split": split, "model": name, "target": target, "rows": len(items),
                         "smiles_valid": sum(r["smiles_valid"] for r in items),
                         "changed_and_local_passed": sum(r.get("changed", False) and r["local_hard_constraints_passed"] for r in items)}
                target_metrics.append(entry)
                if split == "test":
                    target_lines.append(f"| {target} | {name} | {entry['rows']} | {entry['smiles_valid']} | {entry['changed_and_local_passed']} |")
    (args.output / "per_target.json").write_text(json.dumps(target_metrics, indent=2) + "\n")
    admet_section = ""
    admet = None
    audit_summary = None
    admet_path = args.run.parent / "admet/report.json"
    if admet_path.exists():
        admet = json.loads(admet_path.read_text())
        if admet["status"] != "complete":
            raise ValueError("ADMET evaluation is incomplete")
        audit = json.loads((admet_path.parent / "reference_and_witness_audit.json").read_text())
        failures = [r for r in audit if not r["witness_admet_conditions"]["all_admet_goals_passed"]
                    or not r["witness_admet_conditions"]["all_admet_holds_passed"]]
        audit_summary = {"rows": len(audit), "witness_admet_failure_rows": len(failures),
                         "witness_admet_failure_task_ids": [r["task_id"] for r in failures],
                         "reference_values_different_gt_1e-3": sum(abs(d) > 1e-3 for r in audit
                            for d in r["reference_deltas_from_published_baseline"].values()),
                         "reference_max_abs_delta": admet["baseline_reference_max_abs_delta"]}
        failure_lines = []
        for item in failures:
            failed = [f"{label}:{c['property']}" for label, key in [("goal", "goals"), ("hold", "holds")]
                      for c in item["witness_admet_conditions"][key] if c["passed"] is False]
            failure_lines.append(f"| {item['task_id']} | {item['split']} | {', '.join(failed)} |")
        lines = []
        test_failure_lines = []
        for split in ["train", "validation", "test"]:
            for name in model_names:
                item = admet["splits"][split][name]
                c = item["counts"]
                lines.append(f"| {split} | {name} | {c.get('all_admet_goals_passed', 0)}/{c.get('eligible_admet_goals', 0)} | "
                             f"{c.get('all_admet_holds_passed', 0)}/{c.get('valid_smiles', 0)} | "
                             f"{item['full_success_known_false']}/{item['rows']} | {item['full_success_pending_boltz']} |")
        for name in model_names:
            outcomes = [json.loads(line) for line in (admet_path.parent / name / "test/conditions.jsonl").read_text().splitlines()]
            for outcome in outcomes:
                if outcome["local_hard_constraints_passed"]:
                    goals = [g["property"] for g in outcome["conditions"]["goals"] if g["passed"] is False]
                    holds = [g["property"] for g in outcome["conditions"]["holds"] if g["passed"] is False]
                    test_failure_lines.append(f"| {name} | {outcome['task_id']} | {outcome['changed']} | "
                                              f"{', '.join(goals) or '없음'} | {', '.join(holds) or '없음'} |")
        admet_section = f"""## ADMET-AI v1.4.0 추가 평가

전체 모델의 유효한 SMILES에 공식 버전 ADMET-AI를 실행했습니다. 값은 실험값이 아니라 모델 예측입니다.
Checkpoint 선택에는 사용하지 않았습니다.
모델 응답의 **원문 product_smiles**로 측정했습니다. RDKit graph 검사용 canonical SMILES로
바꿔 넣지 않습니다. 전체 Markdown fence만 제거할 수 있으며 분자 자체는 수정하지 않았습니다.
기존 canonical 입력 계산은 `admet_canonical_diagnostic/`에 진단용으로 보관했습니다.

| Split | Model | ADMET 목표 전체 통과/측정 가능 | ADMET hold 통과/유효 SMILES | 필수 조건 실패 확인 | Boltz 미검증 후보 |
|---|---|---:|---:|---:|---:|
{chr(10).join(lines)}

ADMET 목표가 없는 binding-only 과제는 ADMET 목표 통과율의 분모에서 제외했습니다.
표의 Boltz 미검증 후보는 이미 로컬·ADMET 조건을 통과한 것으로 결합 관련 검증이 추가로 필요합니다.
다른 필수 조건을 실패한 후보는 Boltz를 실행해도 전체 과제 성공으로 인정할 수 없습니다.
상세 개선량과 hold 실패는 `admet/{{base,adapter,last_adapter}}/{{train,validation,test}}/conditions.jsonl`에 있습니다.

### 로컬 제약을 통과한 test 출력의 실패 조건

변경이 없는 reference 복사도 로컬 제약 통과 수에 들어갈 수 있으므로 변경 여부를 별도로 표시합니다.
각 목표의 수치와 개선량은 원본 conditions.jsonl에 있습니다.

| Model | Test task | Reference 변경 | 실패한 ADMET 목표 | 실패한 hold |
|---|---|---|---|---|
{chr(10).join(test_failure_lines)}

### 원본 label과 baseline 재현 audit

원본 witness.smi {len(audit)}개 중 {len(failures)}개가 현재 ADMET oracle에서 목표 또는 hold 조건을
재현하지 못했습니다. Reference 재예측 {len(audit) * 8}개 속성값 중
{audit_summary['reference_values_different_gt_1e-3']}개가 공개 baseline과 1e-3보다 크게 달랐고,
최대 절대 차이는 {audit_summary['reference_max_abs_delta']:.4f}였습니다.
목표 판정은 공개 baseline을 그대로 사용했습니다. 현재 수치는 공식 조건을 적용한 로컬 재평가이며,
이 재현 차이를 감안해야 합니다. 차이의 원인은 확정하지 않았고 임계값을 임의로 완화하지 않았습니다.

| Witness task | Split | 재현 실패 조건 |
|---|---|---|
{chr(10).join(failure_lines)}

자세한 예측값·차이는 `admet/reference_and_witness_audit.json`에 있습니다.

"""
    boltz_section = ""
    boltz = None
    boltz_path = args.run.parent / "boltz_heldout/report.json"
    if boltz_path.exists():
        boltz = json.loads(boltz_path.read_text())
        successful_predictions = []
        for candidate in boltz["results"]:
            if candidate["full_benchmark_success"] is True:
                name, split, task_id = candidate["model"], candidate["split"], candidate["task_id"]
                output = next(json.loads(line) for line in (args.run / f"final/{name}/{split}/predictions.jsonl").read_text().splitlines()
                              if json.loads(line)["task_id"] == task_id)
                oracle = next(json.loads(line) for line in (admet_path.parent / name / split / "conditions.jsonl").read_text().splitlines()
                              if json.loads(line)["task_id"] == task_id)
                successful_predictions.append({"model": name, "split": split, "task_id": task_id,
                                               "generation": output, "admet": oracle,
                                               "boltz": candidate})
        (args.output / "successful_predictions.json").write_text(json.dumps(successful_predictions, indent=2) + "\n")
        lines = []
        for split in ["validation", "test"]:
            for name in model_names:
                item = boltz["splits"][split][name]
                lines.append(f"| {split} | {name} | {item['full_success']}/{item['rows']} | "
                             f"{item['full_failure']}/{item['rows']} | {item['unmeasured']} |")
        details = []
        for result in boltz["results"]:
            if result["status"] == "measured":
                checked = result["binding_conditions"]
                details.append(f"| {result['model']} | {result['task_id']} | {checked['binding_probability']:.4f} | "
                               f"{checked['affinity_improvement']:.4f} | {checked['all_passed']} |")
            else:
                details.append(f"| {result['model']} | {result['task_id']} | 미측정 | 미측정 | 미확정 |")
        boltz_section = f"""## Held-out Boltz 결합 검증

모든 필수 로컬·ADMET 조건을 통과한 held-out 후보에만 Boltz 2.2.1을 실행했습니다.
앞 단계에서 실패한 출력도 전체 평가 분모에 포함합니다. 상태: `{boltz['status']}`.
단백질 FASTA의 모든 chain과 task.yaml의 pocket, seed 42,
구조·affinity diffusion sample 각각 10, diffusion step 각각 200을 사용했습니다.
로컬 기준 checkpoint를 고정한 뒤 실행했고 이 결합 결과는 선택에 사용하지 않았습니다.
Affinity 개선량은 Boltz의 `affinity_pred_value` (log10 IC50, IC50 단위 μM) 감소량입니다.
ADMET 목표와 hold를 통과한 뒤 결합 확률·친화도 조건까지 통과해야 전체 조건 통과로 집계합니다.

| Split | Model | 전체 조건 통과 | 필수 조건 실패 | 미확정 |
|---|---|---:|---:|---:|
{chr(10).join(lines)}

| Model | Candidate | Binding probability (>0.7 필요) | Affinity 개선량 (>=0.3 필요) | 결합 조건 통과 |
|---|---|---:|---:|---|
{chr(10).join(details)}

이번 측정에서는 마지막 5-epoch 모델의 BRD4 후보 1개가 전체 조건을 통과했고,
로컬 기준으로 선택한 2-epoch 모델의 BRAF 후보는 결합 확률·친화도 목표를 실패했습니다.
따라서 낮은 validation completion loss나 높은 로컬 제약 통과 수가 전체 목표 달성의
최고 checkpoint를 보장하지 않았습니다. 두 checkpoint는 그대로 보관하며 test에 맞춰 다시 선택하지 않습니다.

Train에서 앞 단계를 통과한 후보 {sum(admet['splits']['train'][n]['full_success_pending_boltz'] for n in model_names)}개는
결합 검증을 진행하지 않았으며 train의 전체 최적화 성공률은 미확정입니다.
상세 값, 실제 실행 명령과 실패 로그는 `boltz_heldout/report.json`, 각 후보의 `boltz.log`에 있습니다.
전체 조건을 통과한 예시는 `successful_predictions.json`에 원문 응답, 로컬·ADMET·Boltz 측정을 함께 저장했습니다.
이 평가도 모델 예측 기반이며 실험적 친화도 검증이나 공식 multi-turn leaderboard 결과는 아닙니다.

"""
    interpretation = ""
    if "last_adapter" in model_names:
        train_last = report["final"]["train"]["last_adapter"]
        val_best = report["final"]["validation"]["adapter"]
        val_last = report["final"]["validation"]["last_adapter"]
        test_best = report["final"]["test"]["adapter"]
        test_last = report["final"]["test"]["last_adapter"]
        counts = lambda item: item["fence_removed_diagnostic"]["counts"]
        interpretation = f"""마지막 checkpoint의 train completion loss는 {loss_text(train_last['completion_loss'])},
유효 SMILES는 {counts(train_last)['smiles_valid']}/{train_last['rows']},
witness exact match는 {counts(train_last)['witness_exact_match']}/{train_last['rows']}입니다.
Validation completion loss는 best의 {loss_text(val_best['completion_loss'])}에서
{loss_text(val_last['completion_loss'])}로 바뀌었고, 변경+로컬 제약 통과는
{counts(val_best)['changed_and_local_passed']}/{val_best['rows']}에서
{counts(val_last)['changed_and_local_passed']}/{val_last['rows']}로 바뀌었습니다.
Test의 변경+로컬 제약 통과는 {counts(test_best)['changed_and_local_passed']}/{test_best['rows']}에서
{counts(test_last)['changed_and_local_passed']}/{test_last['rows']}로 바뀌었습니다.
"""
        if val_last["completion_loss"] is not None and val_best["completion_loss"] is not None and val_last["completion_loss"] > val_best["completion_loss"]:
            interpretation += "Train fitting이 좋아지는 동안 validation loss가 상승해 overfitting 경향이 있습니다.\n"
        if counts(test_last)["changed_and_local_passed"] > counts(test_best)["changed_and_local_passed"]:
            interpretation += "일부 test 지표는 좋아졌으므로 모든 test 성능이 악화되었다고 단정할 수 없습니다.\n"
    if admet is not None and all(admet["splits"]["test"][n]["full_success_known_false"] ==
                                admet["splits"]["test"][n]["rows"] for n in model_names):
        interpretation += ("전체 test에서 로컬·ADMET 조건을 함께 통과한 출력은 세 모델 모두 0건입니다. "
                           "Loss 감소나 출력 형식 개선을 목표 달성으로 해석하면 안 됩니다.\n")
    text = f"""# SMDD-Bench 전체 결과

Qwen3.5-9B rank-8 LoRA를 GPU 4에서 선택 경로 기준 총 {report['completed_total_epochs']} epoch 학습했습니다.
이전 1 epoch의 adapter에서 warm start했으며 LR 후보 두 개를 동일한 시작 가중치로 비교했습니다.
Validation이 악화되어도 학습을 중단하지 않았습니다.

- 선택된 peak LR: `{report['selected_peak_learning_rate']:g}` (비교한 후보 내 선택).
- 최종 평가에 사용한 best checkpoint: 총 {report['best_total_epoch']} epoch.
- 마지막 {report['completed_total_epochs']}-epoch checkpoint도 별도 보관.
- Training microbatch {manifest['args']['microbatch']}, generation batch {manifest['args']['generation_batch']}.
- 최대 tensor GPU memory: {report['peak_memory_gib']:.2f} GiB.
- 학습·LR 비교·base/best 전체 평가 시간: {report['seconds'] / 60:.1f}분 (마지막 checkpoint 및 ADMET 추가 평가 시간 제외).

## LR 비교 — validation 34개 전체

| LR | Completion loss | 유효한 SMILES | 변경 + 로컬 제약 통과 |
|---:|---:|---:|---:|
{chr(10).join(trial_lines)}

선택은 변경+로컬 제약 통과 수, 유효한 SMILES 수, 낮은 loss 순서입니다.
Loss만 낮아진 checkpoint를 자동으로 선택하지 않았습니다. Test 결과는 선택에 사용하지 않았습니다.

## 선택 경로의 epoch별 validation

| 총 epoch | LR | Completion loss | 유효한 SMILES | 변경 + 로컬 제약 통과 |
|---:|---:|---:|---:|---:|
{chr(10).join(epoch_lines)}

## 최종 전체 split 평가

`base`는 adapter가 없는 모델, `adapter`는 validation의 **로컬 제약 기준**으로 선택한 best checkpoint,
`last_adapter`는 마지막 5-epoch checkpoint입니다.

| Split | Model | 전체 row | Completion loss | 엄격한 JSON | 유효한 SMILES* | 변경 + 로컬 제약* |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(split_lines)}

별표 항목은 완전한 Markdown fence만 제거한 뒤 검사한 값입니다.
SMILES의 결합/원자/문법 오류는 수정하지 않았습니다. 엄격한 출력 검사도 원본 JSON에 별도 기록했습니다.

![전체 validation과 split 결과](epochs_and_splits.png)

## 해석 범위

모든 train 272 / validation 34 / test 34를 평가했습니다. Train은 학습에 사용한 분자 쌍에 대한
fitting 진단이고, 일반화 평가는 validation/test를 기준으로 해야 합니다. 초기 split은 단백질 서열,
동일 분자, nonempty scaffold를 공유하는 과제의 연결 그룹을 나눈 것입니다.
단백질 homology clustering까지 적용한 원격 상동 단백질 평가로 해석할 수는 없습니다.
표적 이름 기준 train {len(split_targets['train'])} / validation {len(split_targets['validation'])} /
test {len(split_targets['test'])}개 표적을 포함합니다. 각 표적의 과제 수가 같지 않으므로
전체 row 평균은 과제 수가 많은 표적의 영향을 더 받습니다. 전체 표적별 집계는 `per_target.json`에 있습니다.

| Test target | Model | 과제 수 | 유효 SMILES | 변경+로컬 제약 통과 |
|---|---|---:|---:|---:|
{chr(10).join(target_lines)}

`변경 + 로컬 제약`은 유효한 변경 분자가 RDKit 물성·필터·유사도 제약을 모두 만족했다는 뜻입니다.
ADMET-AI와 Boltz 조건을 모두 포함하는 **전체 SMDD 최적화 성공률이 아닙니다**.
추가 ADMET 측정 결과와 held-out Boltz 결합 조건 평가 결과는 아래에 분리했습니다.
Witness exact match는 imitation 지표이며 유일한 성공 기준이 아닙니다.
Graph metric delta는 각 응답에 저장했고 임의의 utility로 합치지 않았습니다.

{interpretation}

{admet_section}
{boltz_section}

## 코드와 산출물

- [전체 학습 안내](../../docs/smdd_full_training.md)
- [학습·평가 코드](../../src/dirmo/training/smdd_full.py)
- [CLI](../../scripts/train_smdd_full.py)
- [ADMET 평가](../../scripts/evaluate_smdd_admet.py)
- [Boltz 평가](../../scripts/evaluate_smdd_boltz.py)
- 실행 폴더: `{args.run}`
- 모델: `best_adapter/`, `last_adapter/`, `checkpoints/`.
- 모든 응답 원문: `final/{{base,adapter,last_adapter}}/{{train,validation,test}}/predictions.jsonl`.
- 집계: `report.json`, `full_results.json`, `split_metrics.csv`.

PLIP 특징은 현재 입력에 추가하지 않았습니다. Reference의 상호작용을 atom map/잔기/거리/각도로
구조화하는 별도 비교 실험을 권합니다. Witness 상호작용은 input에 넣지 않습니다.
PLIP은 결합 친화도나 ADMET 측정을 대신하지 않습니다.
"""
    (args.output / "REPORT.md").write_text(text)
    (args.output / "summary.json").write_text(json.dumps({"selection": {k: report[k] for k in
        ["selected_peak_learning_rate", "best_total_epoch", "completed_total_epochs", "peak_memory_gib", "seconds"]},
        "splits": rows, "split_targets": split_targets, "admet": admet, "oracle_audit": audit_summary, "boltz_heldout": boltz}, indent=2) + "\n")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
