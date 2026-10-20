"""One-bond coordinate-mask relaxation; reuse frozen graphs and saved priors."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import benchmark_edited_graph as base
from dirmo.chemistry.graphs import read_sdf
from dirmo.evaluation.edited_graph import graph_from_record, unsupported_tensor_features
from dirmo.io import digest, sha256_file

BASE_SCRIPT = Path(base.__file__).resolve()
SOURCE = ROOT / "runs/fm_edited_graph_gpu3_100_20261016_v1"
ORIGINAL_SUMMARIZE = base.summarize
# The legacy arm label 'core' here means the relaxed mask in this run only.
# Every subprocess enters this wrapper, so the isolated module configuration
# never changes the GPU3 worker's masks or scheduling.
base.ARMS = ("core",)
base.__file__ = __file__


def expand_one_bond(graph, free_indices):
    free = set(free_indices)
    if not free or not free < set(range(graph.GetNumAtoms())):
        raise ValueError("Require valid nonempty free and held sets")
    neighbors = {n.GetIdx() for i in free for n in graph.GetAtomWithIdx(i).GetNeighbors()}
    expanded = free | neighbors
    held = set(range(graph.GetNumAtoms())) - expanded
    if not held:
        raise ValueError("Relaxation would leave no fixed core")
    return sorted(expanded), sorted(held), sorted(expanded - free)


def prepare(args):
    root = args.output
    old = base.read(root / "study.json")
    if old:
        if (old["physical_gpu"], old["steps"], old["draws_per_graph"]) != (
                args.gpu, args.steps, args.draws) or Path(old["source_run"]).resolve() != args.source.resolve():
            raise ValueError("Existing relaxation study differs")
        return old
    source = args.source.resolve()
    plan = base.read(source / "study.json")
    if (len(plan["cases"]), plan["steps"], plan["draws_per_graph"]) != (100, args.steps, args.draws):
        raise ValueError("Require the existing complete 100-case input cohort and paired settings")
    cases, hashes, preflight = [], [], []
    root.mkdir(parents=True, exist_ok=True)
    for original in plan["cases"]:
        case = json.loads(json.dumps(original))
        graph = graph_from_record(case["graphs"][0])
        free, held, released = expand_one_bond(graph, case["free_indices"])
        if len(released) != 1 or len(held) < 3:
            raise ValueError("This protocol expects exactly one released boundary atom and >=3 held atoms")
        case.update(original_free_indices=case["free_indices"], original_held_indices=case["held_indices"],
                    original_held_xyz=case["held_xyz"], released_indices=released,
                    free_indices=free, held_indices=held,
                    held_xyz={str(i): case["held_xyz"][str(i)] for i in held})
        cid = case["case_id"]
        for record in case["graphs"]:
            preflight.append({"case_id": cid, "variant": record["name"],
                              "unsupported_features": unsupported_tensor_features(graph_from_record(record))})
            for draw in range(args.draws):
                marker = Path("cases") / cid / "inputs" / record["name"] / f"{draw:02d}" / "result.json"
                item = base.read(source / marker)
                if item is None:
                    raise ValueError(f"Missing frozen input: {marker}")
                if item["status"] == "ready":
                    actual = sha256_file(item["sdf"])
                    if actual != item["sha256"]:
                        raise ValueError("Source input hash changed")
                    hashes.append({"case_id": cid, "variant": record["name"], "draw": draw,
                                   "sdf": item["sdf"], "sha256": actual, "seed": item["seed"]})
                base.save(root / marker, item)
        base.save(root / "cases" / cid / "case.json", case)
        base.save(root / "cases" / cid / "inputs_ready.json", {"case_id": cid, "status": "reused_frozen_inputs"})
        cases.append(case)
    cache = root / "cache"
    if not cache.exists():
        cache.symlink_to(source / "cache", target_is_directory=True)
    new = {"schema_version": 1, "cases_requested": len(cases), "cases": cases,
           "source_run": str(source), "source_study_sha256": sha256_file(source / "study.json"),
           "physical_gpu": args.gpu, "steps": args.steps, "draws_per_graph": args.draws,
           "checkpoint": plan["checkpoint"], "checkpoint_sha256": plan["checkpoint_sha256"],
           "selection_seed": plan["selection_seed"], "arms": ["core"],
           "arm_semantics": "core in this run = original free atoms plus all one-bond graph neighbors; all other XYZ held",
           "planned_fm_outputs": len(cases)*4*args.draws,
           "planned_measurements": len(cases)*(1+4*args.draws*3),
           "graph_condition": "Identical original/N/O/F target graphs; every discrete graph feature held",
           "prior": "Exactly the same saved input SDF and seed; no new embedding, placement or native-free coordinates",
           "mask_selection": "One-hop expansion from the original free set using original connectivity; no PB/property outcome selection",
           "controls": "Original core/whole FM arms reused from source; relaxed-mask ligand-only UFF200 also measured",
           "primary_endpoint": "PB dock20 + target graph/stereo + requested remaining-core max drift <=0.001 A",
           "movement_endpoints": "Original core and released atom motion measured separately; released motion is not a hard-mask failure",
           "failure_accounting": "All 100 cases/3200 slots retained including original initialization failures and unsupported radicals",
           "training": False, "llm_calls": 0,
           "cache": "Source PB measure/50-conformer energy cache shared with file locks and identity keys",
           "code_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
               Path(__file__), BASE_SCRIPT, ROOT / "src/dirmo/generation/pose_repair.py",
               ROOT / "src/dirmo/evaluation/edited_graph.py")}}
    new["protocol_id"] = digest({k:v for k,v in new.items() if k != "cases"})
    base.save(root / "study.json", new)
    base.save(root / "input_hashes.json", hashes)
    base.save(root / "input_preflight.json", preflight)
    return new


def evaluate(root, cid):
    plan = base.read(root / "study.json")
    source = Path(plan["source_run"])
    baseline = base.read(source / "evaluations" / cid / "result.json", {})
    target = root / "evaluations" / cid
    target.mkdir(parents=True, exist_ok=True)
    # Reuse only a published complete receptor result. Otherwise the same
    # pinned preparation code builds a separate receptor from the same protein.
    if baseline.get("complete") and not (target / "receptor-preparation.json").exists():
        for name in ("receptor-preparation.json", "receptor-preparation.log", "receptor.pdbqt"):
            path = source / "evaluations" / cid / name
            if path.exists():
                shutil.copy2(path, target / name)
        base.save(target / "receptor-reuse.json", {"source": str(source / "evaluations" / cid),
                   "receptor_sha256": sha256_file(target / "receptor.pdbqt") if (target / "receptor.pdbqt").exists() else None})
    base.evaluate_case(root, cid)
    result = base.read(target / "result.json")
    case = base.read(root / "cases" / cid / "case.json")
    original = {int(i):xyz for i,xyz in case["original_held_xyz"].items()}
    for row in result["rows"].values():
        if row.get("sdf") and Path(row["sdf"]).exists():
            xyz = read_sdf(row["sdf"]).GetConformer().GetPositions()
            row["original_core_max_drift"] = max(float(np.linalg.norm(xyz[i]-p)) for i,p in original.items())
            row["released_max_drift"] = max(float(np.linalg.norm(xyz[i]-original[i])) for i in case["released_indices"])
        else:
            row["original_core_max_drift"] = row["released_max_drift"] = None
    local = target / "receptor.pdbqt"
    parent = source / "evaluations" / cid / "receptor.pdbqt"
    result["source_receptor_sha256"] = sha256_file(parent) if parent.exists() else None
    result["receptor_sha256"] = sha256_file(local) if local.exists() else None
    base.save(target / "result.json", result)


def summarize(root, status="running", detailed=False):
    summary = ORIGINAL_SUMMARIZE(root, status, detailed)
    plan = base.read(root / "study.json")
    source = Path(plan["source_run"])
    pairs, common_cases, movements, snapshots = [], [], [], []
    for case in plan["cases"]:
        cid = case["case_id"]
        path = root / "evaluations" / cid / "result.json"
        local = base.read(path, {})
        old_path = source / "evaluations" / cid / "result.json"
        old = base.read(old_path, {})
        if not local.get("complete") or not old.get("complete"):
            continue
        common_cases.append(cid)
        receptor_match = local.get("receptor_sha256") is not None and local.get("receptor_sha256") == sha256_file(source / "evaluations" / cid / "receptor.pdbqt") if (source / "evaluations" / cid / "receptor.pdbqt").exists() else False
        snapshots.append({"case_id":cid, "relaxed_evaluation_sha256":sha256_file(path),
                          "source_evaluation_sha256":sha256_file(old_path), "same_receptor": receptor_match})
        for graph in case["graphs"]:
            variant = graph["name"]
            for draw in range(plan["draws_per_graph"]):
                relaxed = local["rows"][f"{variant}/{draw}/fm_core"]
                for arm, row in (("original_core", old["rows"][f"{variant}/{draw}/fm_core"]),
                                 ("relaxed_core", relaxed),
                                 ("whole", old["rows"][f"{variant}/{draw}/fm_whole"])):
                    pairs.append({"case_id":cid,"variant":variant,"draw":draw,"arm":arm,
                                  "pb_valid":row["pb"]["passes"],"success":row["success"],
                                  "graph_preserved":row["contract"]["graph_preserved"],
                                  "required_core_preserved":None if arm=="whole" else row["contract"]["core_preserved"],
                                  "vina":row["vina"].get("score"), "same_receptor":receptor_match,
                                  "status":row["status"],"released_max_drift":row.get("released_max_drift"),
                                  "original_core_max_drift":row.get("original_core_max_drift")})
                if relaxed.get("released_max_drift") is not None:
                    movements.append(relaxed["released_max_drift"])
    grouped = defaultdict(list)
    for item in pairs:
        grouped[item["arm"]].append(item)
    comparison=[]
    for arm, rows in grouped.items():
        comparison.append({"arm":arm,"slots":len(rows),"pb_valid":sum(x["pb_valid"] for x in rows),
                           "graph_preserved":sum(x["graph_preserved"] for x in rows),
                           "success":sum(x["success"] for x in rows)})
    summary.update(condition="one-bond relaxed core; one released atom per case", source_run=str(source),
                   common_evaluated_cases=len(common_cases),paired_comparison=comparison,
                   unsupported_cases_accounted=sum(base.read(root/"cases"/c["case_id"]/"generation_done.json",{}).get("slot_statuses",{}).get("unsupported_input",0)>0 for c in plan["cases"]),
                   released_drift_median=float(np.median(movements)) if movements else None)
    base.save(root / "summary.json", summary)
    base.save(root / "comparison_snapshot.json", {"cases":common_cases,"sources":snapshots,"groups":comparison})
    lines=["# GPU4: core 고정 경계 한 결합 완화 실험", "",
           f"상태: {status}. 입력 100 pockets, graph 4개, 시작 pose 8개, base 64-step. 추가 FM 예정 3,200개.", "",
           "기존 자유 영역에서 한 결합 이웃까지 확장해 경계 원자 1개를 더 움직인다. 나머지 core 좌표와 전체 chemical graph는 고정한다. 원본·시작 SDF·seed·pocket은 GPU3 실험과 같고 원본 좌표로 다시 초기화하지 않는다.", "",
           f"생성 처리 {summary.get('generation_complete_cases',0)}/100, 결과 슬롯 {summary.get('fm_terminal_slots',0)}/3200, CPU 평가 {summary['evaluation_complete_cases']}/100, 기존 조건과 완결 paired 비교 {len(common_cases)}/100.", "",
           "radical 입력 5edl과 초기화 실패도 예정 분모에 유지한다. 생성 처리 수에는 지원 불가 사례도 포함된다. UFF core는 이 run에서 같은 완화 mask를 쓰는 ligand-only 대조군이다. GPU3 기존 결과에는 쓰지 않는다.", "",
           "종합 성공 = PB dock20 통과 + 요청 graph/입체화학 보존 + 이번에 고정하도록 요청한 나머지 core 최대 이동 ≤0.001 Å. 풀어준 원자의 이동은 별도로 측정하며 고정 위반으로 세지 않는다.", "",
           "## 같은 pocket·graph·시작 pose에서의 비교", "",
           "두 실험의 평가가 모두 완결된 동일 사례만 비교한다. 표의 비율은 초기화·생성 실패도 분모에 포함한다. 실행 중에는 부분 결과다.", "",
           "| 조건 | 예정 슬롯 | PB 통과 | graph/stereo 보존 | 종합 성공 |", "| --- | ---: | ---: | ---: | ---: |"]
    for g in comparison:
        n=g["slots"]
        lines.append(f"| {g['arm']} | {n} | {g['pb_valid']} ({100*g['pb_valid']/n:.1f}%) | {g['graph_preserved']} ({100*g['graph_preserved']/n:.1f}%) | {g['success']} ({100*g['success']/n:.1f}%) |")
    lines += ["",f"완화 원자의 이동 중앙값: {summary['released_drift_median']} Å. 기존 core 전체 이동과 이번 필수 core 이동은 evaluations/*/result.json에 구분해 저장한다.", "",
              "Vina는 동일한 standard-aa receptor score-only 설정이며 수용체 hash가 같은 쌍만 공정한 paired score 비교에 사용한다. 누락 수용체는 제외 사유와 함께 유지한다. 초기 core 위치 정보가 시작 pose에 들어 있으므로 blind docking은 아니다.", "",
              "study.json: 범위·code hash. input_hashes.json: 원본과 동일한 시작 입력 검증. cases/*/case.json: 원래/완화 free·held·released indices. draws: 실제 mask/출력. comparison_snapshot.json: paired 평가 원본 hash. CPU 평가는 저장된 PB/energy cache를 공유한다."]
    (root / "REPORT.md").write_text("\n".join(lines)+"\n")
    if detailed:
        import csv
        if pairs:
            with (root/"paired_comparison.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=["prepare","run","generate","evaluate-case","prepare-case","summarize"])
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--gpu",default="4")
    parser.add_argument("--steps",type=int,default=64)
    parser.add_argument("--draws",type=int,default=8)
    parser.add_argument("--cases",type=int,default=100)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--case")
    args=parser.parse_args();args.output=args.output.resolve()
    base.summarize=summarize
    if args.mode in ("prepare","run"):
        prepare(args)
    if args.mode=="run":
        base.run(args)
        # GPU4 generation can finish before its live GPU3 controls. Publish the
        # final paired cohort after the controls finish, without reserving GPU4.
        while True:
            plan=base.read(args.output/"study.json")
            parent=Path(plan["source_run"])
            ready=sum(base.read(parent/"evaluations"/c["case_id"]/"result.json",{}).get("complete",False) for c in plan["cases"])
            source_status=base.read(parent/"controller.json",{}).get("status")
            state=base.read(args.output/"controller.json")
            if ready==len(plan["cases"]) or source_status in ("failed","complete","complete_with_missing_evaluation"):
                final="complete_with_missing_evaluation" if ready==len(plan["cases"]) else "complete_with_partial_baseline"
                summarize(args.output,final,True)
                state.update(status=final,paired_source_cases=ready,completed_utc=datetime.now(timezone.utc).isoformat())
                base.save(args.output/"controller.json",state)
                break
            state.update(status="waiting_for_source_evaluation",paired_source_cases=ready)
            base.save(args.output/"controller.json",state)
            summarize(args.output,"waiting_for_source_evaluation")
            time.sleep(15)
    elif args.mode=="generate":base.generate(args.output)
    elif args.mode=="evaluate-case":evaluate(args.output,args.case)
    elif args.mode=="prepare-case":base.prepare_case(args.output,args.case)
    elif args.mode=="summarize":summarize(args.output,base.read(args.output/"controller.json",{}).get("status","prepared"),True)


if __name__=="__main__":main()
