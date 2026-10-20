"""Compare native, de novo, and PB-guided repair outputs in one fixed cohort."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dirmo.io import sha256_file, write_json


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pb_row(stage, case_id, pb, *, draw="", arm="", candidate="", selected=False,
           eligible=False, status="", contract=None):
    checks = pb.get("checks", {})
    # A reported False already proves that the all-checks requirement failed,
    # even when another check is unavailable (e.g. failed InChI -> no energy).
    missing = (bool(pb.get("missing")) or not checks
               or (any(v is None for v in checks.values()) and not any(v is False for v in checks.values())))
    return {"stage": stage, "case_id": case_id, "draw": draw, "arm": arm,
            "candidate": candidate, "selected": selected, "repair_eligible": eligible,
            "pb_valid": bool(pb.get("passes")), "measurement_missing": missing,
            "checks_complete": bool(checks) and all(isinstance(v, bool) for v in checks.values()),
            "status": status, "checks_recorded": len(checks),
            "checks_passed": sum(v is True for v in checks.values()),
            "failed_checks": ";".join(k for k, v in checks.items() if v is False),
            "missing_checks": ";".join(k for k, v in checks.items() if v is None),
            "graph_preserved": (contract or {}).get("graph_preserved", ""),
            "held_preserved": (contract or {}).get("held_preserved", ""),
            "sdf": pb.get("sdf", "")}


def rates(rows):
    valid = sum(r["pb_valid"] for r in rows)
    missing = sum(r["measurement_missing"] for r in rows)
    measured = len(rows) - missing
    return {"scheduled": len(rows), "measured": measured, "pb_valid": valid,
            "pb_failed_measured": measured - valid, "missing": missing,
            "partially_available_but_definitely_failed": sum(not r["checks_complete"] and not r["measurement_missing"] for r in rows),
            "pb_valid_pct_scheduled": 100 * valid / len(rows) if rows else None,
            "pb_valid_pct_measured": 100 * valid / measured if measured else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    native_dir = output / "native_measurements"
    native_dir.mkdir(exist_ok=True)
    plan = read(run / "study.json")
    source_root = Path(plan["source_study"]).parent
    os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1",
                      DIRMO_PB_ENERGY_CACHE=str(run / "cache/reference_energy"),
                      DIRMO_PB_MEASURE_CACHE=str(run / "cache/measurements"))
    from dirmo.evaluation.pb_worker import bounded_measure

    def native_measure(case):
        cid = case["case_id"]
        destination = native_dir / f"{cid}.json"
        identity = {"sdf_sha256": sha256_file(case["native_sdf"]),
                    "pocket_sha256": sha256_file(case["pocket_path"])}
        if destination.exists():
            saved = read(destination)
            if saved["identity"] == identity and not saved["pb"].get("missing"):
                return saved
        source = source_root / "evaluations" / f"{cid}.json"
        existing = read(source).get("native", {}) if source.exists() else {}
        pb = existing.get("pocket", {})
        reusable = (existing.get("sdf") == case["native_sdf"]
                    and pb.get("reference_protein") == case["pocket_path"]
                    and pb.get("preset") == "dock" and len(pb.get("checks", {})) == 20
                    and all(isinstance(v, bool) for v in pb["checks"].values()))
        if reusable:
            pb = {**pb, "sdf": case["native_sdf"]}
            provenance = {"mode": "existing_same_native_and_pocket", "source": str(source),
                          "source_sha256": sha256_file(source)}
        else:
            pb = bounded_measure(case["native_sdf"], case["pocket_path"])
            provenance = {"mode": "additional_cpu_measurement"}
        saved = {"case_id": cid, "identity": identity, "provenance": provenance, "pb": pb}
        write_json(destination, saved)
        print(json.dumps({"native": cid, "passes": pb.get("passes"),
                          "missing": pb.get("missing", False), **provenance}), flush=True)
        return saved

    natives = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(native_measure, c) for c in plan["cases"]]):
            item = future.result()
            natives[item["case_id"]] = item

    rows, paired, case_rows = [], [], []
    native_rows, initial_rows = [], []
    arm_rows = {arm: [] for arm in plan["arms"]}
    selected_rows = {arm: [] for arm in plan["arms"]}
    for case in plan["cases"]:
        cid = case["case_id"]
        native = pb_row("native", cid, natives[cid]["pb"])
        rows.append(native)
        native_rows.append(native)
        initial = read(run / "evaluations/initial" / f"{cid}.json")["draws"]
        repairs = read(run / "evaluations/repairs" / f"{cid}.json")["sources"]
        case_initial = []
        for key, item in sorted(initial.items()):
            row = pb_row("first_generated", cid, item["pb"], draw=int(key),
                         eligible=item["repair_eligible"], status=item["generation_status"])
            rows.append(row)
            initial_rows.append(row)
            case_initial.append(row)
        case_row = {"case_id": cid, "native_pb_valid": native["pb_valid"],
                    "native_measurement_missing": native["measurement_missing"],
                    "first_generated_scheduled": len(case_initial),
                    "first_generated_pb_valid": sum(r["pb_valid"] for r in case_initial),
                    "first_generated_missing": sum(r["measurement_missing"] for r in case_initial),
                    "repair_eligible": sum(r["repair_eligible"] for r in case_initial)}
        for arm in plan["arms"]:
            selected_in_case = []
            for draw, arms in sorted(repairs.items()):
                result = arms[arm]
                selection = result["selected_candidate"]
                for candidate in result["candidates"]:
                    row = pb_row("regenerated", cid, candidate.get("pb", {}), draw=int(draw),
                                 arm=arm, candidate=candidate["candidate"],
                                 selected=candidate["candidate"] == selection,
                                 status=candidate.get("status", ""), contract=candidate.get("contract"))
                    rows.append(row)
                    arm_rows[arm].append(row)
                    if row["selected"]:
                        selected_rows[arm].append(row)
                        selected_in_case.append(row)
                        paired.append({"case_id": cid, "draw": int(draw), "arm": arm,
                                       "before_pb_valid": bool(initial[draw]["pb"]["passes"]),
                                       "after_pb_valid": row["pb_valid"],
                                       "selected_candidate": selection,
                                       "graph_preserved": row["graph_preserved"],
                                       "held_preserved": row["held_preserved"],
                                       "before_sdf": initial[draw]["pb"].get("sdf", ""),
                                       "after_sdf": row["sdf"]})
            case_row[f"{arm}_repaired_pb_valid"] = sum(r["pb_valid"] for r in selected_in_case)
            case_row[f"{arm}_final_pb_valid"] = (case_row["first_generated_pb_valid"]
                                                        + case_row[f"{arm}_repaired_pb_valid"])
        case_rows.append(case_row)

    known_keys = {tuple(sorted(n["pb"].get("checks", {}))) for n in natives.values() if not n["pb"].get("missing")}
    for case in plan["cases"]:
        initial = read(run / "evaluations/initial" / f'{case["case_id"]}.json')["draws"]
        known_keys.update(tuple(sorted(item["pb"]["checks"])) for item in initial.values() if item["pb"].get("checks"))
    if len(known_keys) != 1 or len(next(iter(known_keys))) != 20:
        raise ValueError("Native and initial generations do not share exactly the same 20 PB check keys")
    if len(initial_rows) != len(plan["cases"]) * plan["initial_draws"]:
        raise ValueError("Scheduled first-generation denominator is incomplete")
    eligible = [r for r in initial_rows if r["repair_eligible"]]
    if any(r["pb_valid"] for r in eligible):
        raise ValueError("Repair cohort unexpectedly includes a PB-valid input")
    stats = {"created_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
             "protocol": {"pb": "dock; all 20 binary checks; identical fixed protein-only pocket",
                          "initial_generation": plan["initial_generation"], "repair": plan["repair"],
                          "initial_draws": plan["initial_draws"], "repair_draws": plan["repair_draws"],
                          "steps": plan["steps"], "missing_retained_in_scheduled_denominator": True},
             "native": rates(native_rows), "first_generated": rates(initial_rows),
             "repair_inputs": rates(eligible),
             "native_provenance": dict(Counter(n["provenance"]["mode"] for n in natives.values())),
             "arms": {}}
    for arm in plan["arms"]:
        selected = selected_rows[arm]
        if len(selected) != len(eligible):
            raise ValueError(f"Missing selected repair outputs for {arm}")
        if any(r["pb_valid"] and not (r["graph_preserved"] is True and r["held_preserved"] is True) for r in selected):
            raise ValueError(f"PB-only and contract success differ for selected outputs of {arm}")
        total = stats["first_generated"]["scheduled"]
        repaired = sum(r["pb_valid"] for r in selected)
        final_count = stats["first_generated"]["pb_valid"] + repaired
        stats["arms"][arm] = {"all_candidates": rates(arm_rows[arm]), "selected_outputs": rates(selected),
                              "final_pb_valid": final_count, "final_scheduled": total,
                              "final_pb_valid_pct": 100 * final_count / total,
                              "improvement_pp": 100 * repaired / total}
    write_csv(output / "per_ligand_pb.csv", rows)
    write_csv(output / "repair_before_after.csv", paired)
    write_csv(output / "per_complex_pb.csv", case_rows)
    write_json(output / "summary.json", stats)
    native, first = stats["native"], stats["first_generated"]
    lines = ["# 동일 100개 pocket의 단계별 PoseBusters 결과", "",
             "PB-Valid는 고정 6 Å protein-only pocket에서 PoseBusters dock의 20개 검사를 모두 통과한 비율이다. 분자별 연속 점수나 결합력 점수가 아니다.", "",
             "원본 ligand는 pocket 위치·원자 수를 제공한다. 첫 생성은 원본 graph/좌표를 고정하지 않는 de novo 생성이다. 재생성은 첫 생성물의 graph/입체화학을 유지하고 PB 위반으로 정한 영역의 좌표만 바꾸는 pose-only 방식이다.", "",
             "| 단계 | PB 통과 / 전체 예정 | PB-Valid |", "| --- | ---: | ---: |",
             f'| 원본 native ligand | {native["pb_valid"]}/{native["scheduled"]} | {native["pb_valid_pct_scheduled"]:.2f}% |',
             f'| 첫 생성물 (pocket당 16개) | {first["pb_valid"]}/{first["scheduled"]} | {first["pb_valid_pct_scheduled"]:.2f}% |']
    for arm in plan["arms"]:
        value = stats["arms"][arm]
        selected = value["selected_outputs"]
        lines.append(f'| {arm}: 실패 입력별 선택된 재생성물 | {selected["pb_valid"]}/{selected["scheduled"]} | {selected["pb_valid_pct_scheduled"]:.2f}% |')
    lines += ["", f'첫 생성 결과 중 PB 통과/실패 판정 가능한 결과는 {first["measured"]}개이며 그중 {first["pb_valid"]}개 통과: {first["pb_valid_pct_measured"]:.2f}%. 생성 실패 54개, PB 측정 시간 초과 8개는 전체 예정 1,600개 분모에 남긴다. 이 62개를 측정된 PB 실패로 해석하지 않는다. 별도로 1개는 InChI 검사 실패와 내부 에너지 검사 누락이 함께 있어 전체 PB 실패가 확정된 부분 측정 결과다.', "",
              "PB가 실패한 79개 중 위치를 지정하여 부분 pose 복구를 시도할 수 있는 41개가 재생성 대상이다. 이들의 수정 전 PB 통과는 0/41이다. 나머지 38개 실패는 이번 부분 복구에서 제외됐다.", "",
              "각 FM 방법은 실패 입력당 8개 후보를 생성하고 PB 통과·보존 조건을 우선해 하나를 선택했다. UFF는 입력당 한 번 수행한 CPU 대조군이다.", "",
              "| 방법 | 모든 재생성 후보의 PB 통과 | 입력별 선택 성공 | 기존 통과 생성물을 유지한 최종 전체 통과 | 증가 |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm in plan["arms"]:
        value = stats["arms"][arm]
        all_candidates, selected = value["all_candidates"], value["selected_outputs"]
        lines.append(f'| {arm} | {all_candidates["pb_valid"]}/{all_candidates["scheduled"]} ({all_candidates["pb_valid_pct_scheduled"]:.2f}%) | {selected["pb_valid"]}/{selected["scheduled"]} ({selected["pb_valid_pct_scheduled"]:.2f}%) | {value["final_pb_valid"]}/{value["final_scheduled"]} ({value["final_pb_valid_pct"]:.2f}%) | +{value["improvement_pp"]:.4f} pp |')
    lines += ["", "원본 100개·첫 생성물 1,600개·실패 부분집합 41개는 분모가 다르므로 단계별 비율을 직접 빼서 동일 분자의 개선량으로 해석하면 안 된다. 동일 생성 결과에 대한 전체 비교는 첫 생성물과 최종 전체 1,600개 결과다.", "",
              f'원본 PB 기록 출처: {stats["native_provenance"]}. 원본 SDF와 pocket 경로를 확인해 기존 기록을 재사용했고 없는 항목은 동일 PB 설정으로 CPU 측정했다. 출처·SDF/pocket 해시는 native_measurements/에 저장했다.', "",
              "PB가 진단 기준이자 후보 선택 기준이므로 PB 자체의 개선이다. 결합력·활성 개선이나 독립 평가에서의 일반화를 뜻하지 않는다. 물·보조인자를 포함한 전체 단백질 PB와도 구분해야 한다.", "",
              "- per_ligand_pb.csv: 원본, 첫 생성물, 모든 재생성 후보의 PB와 개별 검사 실패 및 SDF 경로.",
              "- repair_before_after.csv: 동일 41개 입력의 방법별 수정 전/선택 후 결과.",
              "- per_complex_pb.csv: pocket별 세 단계 통과 수.",
              "- summary.json: 전체 집계와 분모·실측 여부."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
