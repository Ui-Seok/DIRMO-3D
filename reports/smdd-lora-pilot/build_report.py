"""Render completed pilot artifacts; run in dirmo (matplotlib available there)."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/smdd_lora_02"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    report = json.loads((args.run / "training/report.json").read_text())
    data = json.loads((args.run / "data/manifest.json").read_text())
    base = json.loads((args.run / "base_diagnostic.json").read_text())
    adapter = json.loads((args.run / "adapter_diagnostic.json").read_text())
    events = [json.loads(line) for line in (args.run / "training/events.jsonl").read_text().splitlines()]
    updates = [event for event in events if event["event"] == "update"]
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot([u["step"] for u in updates], [u["loss"] for u in updates], marker=".", label="Train update mean")
    axes[0].axhline(report["base_validation_loss"], color="tab:orange", linestyle="--", label="Base validation")
    axes[0].axhline(report["best_validation_loss"], color="tab:green", linestyle="--", label="Adapter validation")
    axes[0].set(xlabel="Optimizer update", ylabel="Completion cross entropy", title="LoRA pilot learning")
    axes[0].legend(fontsize=8)
    names = ["json_valid", "smiles_valid", "local_hard_constraints_passed"]
    labels = ["Strict JSON", "SMILES after\nfence removal", "Local constraints\nafter fence removal"]
    n = report["base"]["rows"]
    for shift, source, diagnostic, name, color in [(-0.2, report["base"], base, "Base", "tab:orange"),
                                                   (0.2, report["adapter"], adapter, "Adapter", "tab:green")]:
        values = [source["counts"]["json_valid"]] + [diagnostic["fence_removed_diagnostic"]["counts"][k] for k in names[1:]]
        axes[1].bar([i + shift for i in range(3)], values,
                    width=0.4, label=name, color=color)
    axes[1].set(xticks=range(3), xticklabels=labels, ylim=(0, n + 1), ylabel=f"Passed / {n}",
                title="Same held-out generation prompts")
    axes[1].legend(fontsize=8)
    figure.savefig(args.output / "learning_and_validation.png", dpi=160)
    plt.close(figure)
    table_rows = []
    for field, label in [("json_valid", "엄격한 JSON 형식"), ("smiles_valid", "엄격한 JSON 내 유효한 SMILES"),
                          ("local_hard_constraints_passed", "엄격한 출력의 로컬 제약 통과"),
                          ("changed", "엄격한 출력의 reference 변경"), ("witness_exact_match", "Witness exact match")]:
        table_rows.append(f"| {label} | {report['base']['counts'][field]}/{n} | {report['adapter']['counts'][field]}/{n} |")
    for field, label in [("smiles_valid", "Fence만 제거한 SMILES 유효성"),
                          ("local_hard_constraints_passed", "Fence만 제거한 로컬 제약 통과")]:
        table_rows.append(f"| {label} | {base['fence_removed_diagnostic']['counts'][field]}/{n} | {adapter['fence_removed_diagnostic']['counts'][field]}/{n} |")
    case_rows = []
    for row in adapter["rows"]:
        assessment = row["strict"]
        if assessment["smiles_valid"]:
            failed = [key for key, value in assessment["checks"].items() if not value]
            case_rows.append(f"- `{row['task_id']}`: 실패 제약 {', '.join(failed) or '없음'}; "
                             f"Tanimoto {assessment['tanimoto']:.4f}, SA {assessment['metrics']['sa']:.4f}.")
    text = f"""# SMDD-Bench LoRA pilot 결과

Qwen3.5-9B를 GPU 4에서 1 epoch 학습했습니다. 실행 위치: `{args.run}`.

이번 결과에서는 JSON 형식이 개선됐지만, fence를 제거한 SMILES 유효성은
{base['fence_removed_diagnostic']['counts']['smiles_valid']}/{n}에서
{adapter['fence_removed_diagnostic']['counts']['smiles_valid']}/{n}로 낮아졌습니다.
Adapter의 로컬 제약 전체 통과는 {report['adapter']['counts']['local_hard_constraints_passed']}/{n}입니다.
형식 학습과 분자 편집 능력을 구분해야 하며, 이번 pilot은 최적화 성능 개선을 보여주지 못했습니다.

Train {data['splits']['train']['rows']} / validation {data['splits']['validation']['rows']} / test {data['splits']['test']['rows']}.
단백질 서열·동일 분자·nonempty scaffold의 연결 그룹으로 나눴으며 split 간 키 중복은 0입니다.
Test는 학습기에서 읽지 않았습니다. Validation 생성 비교는 미리 선택한 {n}개에 한정됩니다.

공식 `witness.smi`를 정답으로 사용했습니다. 70개 SDF의 추가 입체화학은 정답으로 전파하지 않았습니다.

- Base validation loss: {report['base_validation_loss']:.6f}
- Adapter validation loss: {report['best_validation_loss']:.6f}
- Epoch 평균 training loss: {report['history'][0]['train_loss']:.6f}
- Trainable parameters: {report['trainable_parameters']:,}
- Peak tensor GPU memory: {report['peak_memory_gib']:.2f} GiB
- 전체 실행 시간: {report['seconds'] / 60:.1f}분

| 검사 | Base | Adapter |
|---|---:|---:|
{chr(10).join(table_rows)}

유효한 adapter 출력의 실패 원인:

{chr(10).join(case_rows)}

Tanimoto는 최소 0.7, SA는 4.5 미만이어야 합니다. 나머지 adapter 출력은 ring closure,
aromatic kekulization, valence 등의 SMILES 문제로 RDKit 검사를 통과하지 못했습니다.

![학습과 생성 검증](learning_and_validation.png)

엄격한 출력 검사는 JSON 형식 검사를 먼저 통과해야 SMILES를 측정합니다.
별도 fence 제거 진단은 형식만 정리하며 분자 자체는 보정하지 않습니다.

Training batch loss와 validation loss는 서로 다른 샘플에서 계산하므로 매 update를 validation 성능으로
해석할 수 없습니다. Validation loss는 witness imitation loss이고, exact match는 유일한 최적화 성공 기준이 아닙니다.

SMILES 유효성·유사도·RDKit 제약 개선은 실제 ADMET/결합 최적화와 구별해야 합니다.
ADMET-AI의 목표/유지 조건 및 Boltz binding probability는 새 출력에서 평가하지 않았습니다.
동일 분자 반환도 일부 로컬 제약을 통과할 수 있습니다. 아직 최종 test 성능이나 전체 SMDD 성공률을 주장할 수 없습니다.

학습 코드와 prompt: [안내 문서](../../docs/smdd_instruction_tuning.md).
완료 결과: `training/report.json`, 응답 원문: `training/{{base,adapter}}_predictions.jsonl`.
"""
    (args.output / "REPORT.md").write_text(text)
    (args.output / "summary.json").write_text(json.dumps({"run": str(args.run), "report": report,
        "base_diagnostic": {k: v for k, v in base.items() if k != "rows"},
        "adapter_diagnostic": {k: v for k, v in adapter.items() if k != "rows"}}, indent=2) + "\n")
    print(args.output / "REPORT.md")


if __name__ == "__main__":
    main()
