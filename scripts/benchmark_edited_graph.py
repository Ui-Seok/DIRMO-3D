"""100-pocket frozen-FLOWR explicit-edit study with paired whole/core pose arms."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from rdkit import Chem
from dirmo.chemistry.graphs import graph_id, read_sdf, write_sdf
from dirmo.evaluation.edited_graph import (anchor_indices, assess_saved_graph, graph_from_record,
    graph_only_conformer, graph_record, terminal_edits, unsupported_tensor_features)
from dirmo.io import append_jsonl, digest, sha256_file

ARMS = ("whole", "core")
VARIANTS = ("original", "C_to_N", "C_to_O", "C_to_F")
PYTHON = "/home/uslee/miniconda3/envs/dirmo/bin/python"
CHECKPOINT = "/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_base_spindr.ckpt"
CHECKSUM = "78ef9ebe4459df1ce2668685785602a37288594db83d2bb5d970c8bc06e1ce9d"


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def emit(**fields):
    print(json.dumps(fields, ensure_ascii=False, allow_nan=False), flush=True)


def case_directory(root, cid):
    return root / "cases" / cid


def output_directory(root, cid, variant, arm, draw):
    return root / "draws" / cid / variant / arm / f"{draw:02d}"


def prepare_study(args):
    root = args.output
    if (root / "study.json").exists():
        plan = read(root / "study.json")
        if (plan["cases_requested"], plan["steps"], plan["physical_gpu"], plan["draws_per_graph"]) != (
                args.cases, args.steps, args.gpu, args.draws):
            raise ValueError("Existing study has a different configuration")
        return plan
    from dirmo.evaluation.pocket import protein_atoms
    source = read(args.source)
    candidates = list(source["cases"])
    np.random.default_rng(args.seed).shuffle(candidates)
    selected, excluded, seen = [], [], set()
    table = Chem.GetPeriodicTable()
    for candidate in candidates:
        cid = candidate["case_id"]
        if candidate["target_id"] in seen:
            excluded.append({"case_id": cid, "reason": "Duplicate exact protein sequence"})
            continue
        try:
            native = read_sdf(candidate["native_sdf"])
            free = candidate["region"]["free_indices"]
            variants = terminal_edits(native, free)
            pocket, elements, _ = protein_atoms(candidate["pocket_path"])
            if len(pocket) > 1000 or native.GetNumAtoms() > 64:
                raise ValueError("Model atom/pocket cap")
            if set(elements) - {"C", "N", "O", "S", "P"}:
                raise ValueError("Nonstandard protein element outside this study's pocket vocabulary")
            distance = np.linalg.norm(native.GetConformer().GetPositions()[:, None] - pocket[None], axis=-1)
            ligand_radii = np.array([table.GetRcovalent(a.GetAtomicNum()) for a in native.GetAtoms()])
            protein_radii = np.array([table.GetRcovalent(table.GetAtomicNumber(e)) for e in elements])
            ratio = distance / (ligand_radii[:, None] + protein_radii[None])
            if float(ratio.min()) < 1.2:
                raise ValueError("Suspected covalent/very-short native protein contact: distance < 1.2 * summed covalent radii; conservative eligibility screen, not a covalent annotation")
            anchors = anchor_indices(native, free, candidate["region"]["anchor_index"])
            held = [i for i in range(native.GetNumAtoms()) if i not in set(free)]
            xyz = native.GetConformer().GetPositions()
            case = {k: candidate[k] for k in ("case_id", "target_id", "scaffold", "source_dataset",
                    "native_sdf", "pocket_path", "protein_path", "atoms", "pocket_atoms")}
            case.update(free_indices=free, held_indices=held, anchors=anchors,
                        held_xyz={str(i): xyz[i].tolist() for i in held},
                        shortest_native_covalent_radius_ratio=float(ratio.min()),
                        source_hashes={"native": sha256_file(case["native_sdf"]),
                                       "pocket": sha256_file(case["pocket_path"]),
                                       "protein": sha256_file(case["protein_path"])},
                        graphs=[graph_record(name, mol, changed, free) for name, mol, changed in variants])
            case["index"] = len(selected)
            save(case_directory(root, cid) / "case.json", case)
            selected.append(case)
            seen.add(case["target_id"])
            emit(status="selected", cases=len(selected), case_id=cid)
            if len(selected) == args.cases:
                break
        except Exception as exc:
            excluded.append({"case_id": cid, "reason": f"{type(exc).__name__}: {exc}"})
    save(root / "eligibility_exclusions.json", excluded)
    if len(selected) != args.cases:
        raise ValueError(f"Only {len(selected)} eligible cases for the frozen edit rule")
    plan = {"schema_version": 1, "cases_requested": args.cases, "cases": selected,
            "source": str(args.source), "source_sha256": sha256_file(args.source),
            "selection_seed": args.seed, "physical_gpu": args.gpu, "checkpoint": CHECKPOINT,
            "checkpoint_sha256": CHECKSUM, "steps": args.steps, "draws_per_graph": args.draws,
            "arms": list(ARMS), "variants": list(VARIANTS),
            "planned_fm_outputs": args.cases * 4 * 2 * args.draws,
            "planned_measurements": args.cases * (1 + 4 * args.draws * 5),
            "training": False, "llm_calls": 0, "pretraining_overlap": "unknown",
            "selection": "Seed-shuffled existing 2000-case pool; unique exact protein sequence; no PB, property or Vina outcome selection",
            "edit_rule": "Same one free terminal C-C carbon replaced by neutral N, O, F; fixed atom count/bonds/maps; no property ranking",
            "native_contact_eligibility": "Reject distance < 1.2 * sum of covalent radii; suspected covalent/short contact rather than verified chemical linkage",
            "pocket": "Same fixed 6 A protein-only pocket in original protein frame; no recutting",
            "initialization": "Graph-only ETKDGv3; fit four local held anchors; overwrite only held XYZ; original free XYZ not used; same 8 starting poses for both arms",
            "whole": "Full target graph fixed; all XYZ free during integration; starts from the same core-placed inputs as core arm",
            "core": "Full target graph fixed; original core XYZ hard-held; terminal region XYZ free",
            "controls": "Same input conformer; ligand-only UFF200 with whole/core coordinate masks; no protein energy term",
            "evaluation": "Raw PB dock 20 checks + graph/stereo + held drift<=0.001 A; score-only Vina if strict receptor preparation succeeds; no native free RMSD endpoint",
            "failure_accounting": "All scheduled slots retained, including initialization/generation/PB/Vina failures; no replacement cases after freezing",
            "batch": "Up to 8 independent priors of one target graph and coordinate mask; OOM halves batch, minimum 1",
            "code_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                Path(__file__), ROOT / "src/dirmo/evaluation/edited_graph.py", ROOT / "src/dirmo/generation/pose_repair.py")}}
    plan["protocol_id"] = digest({k: v for k, v in plan.items() if k != "cases"})
    save(root / "study.json", plan)
    return plan


def prepare_case(root, cid):
    plan = read(root / "study.json")
    case = read(case_directory(root, cid) / "case.json")
    held_xyz = {int(i): xyz for i, xyz in case["held_xyz"].items()}
    for vi, record in enumerate(case["graphs"]):
        graph = graph_from_record(record)
        hashes = []
        for draw in range(plan["draws_per_graph"]):
            directory = case_directory(root, cid) / "inputs" / record["name"] / f"{draw:02d}"
            marker = directory / "result.json"
            if not marker.exists():
                seed = plan["selection_seed"] + case["index"] * 10000 + vi * 100 + draw
                try:
                    mol = graph_only_conformer(graph, held_xyz, case["anchors"], seed)
                    path = directory / "input.sdf"
                    write_sdf(path, mol)
                    contract = assess_saved_graph(graph, path, held_xyz)
                    if not contract["graph_preserved"] or not contract["core_preserved"]:
                        raise ValueError("Persisted ETKDG graph/core contract failed")
                    save(marker, {"status": "ready", "seed": seed, "sdf": str(path),
                                  "sha256": sha256_file(path), "contract": contract,
                                  "native_free_xyz_used": False})
                except Exception as exc:
                    save(marker, {"status": "initialization_failed", "seed": seed,
                                  "reason": f"{type(exc).__name__}: {exc}"})
            item = read(marker)
            if item["status"] == "ready":
                xyz = read_sdf(item["sdf"]).GetConformer().GetPositions()
                hashes.append(digest(xyz[case["free_indices"]].tolist()))
        if len(set(hashes)) != len(hashes):
            raise RuntimeError("Duplicate free-coordinate priors are not independent draws")
    save(case_directory(root, cid) / "inputs_ready.json", {"case_id": cid, "status": "ready"})
    emit(status="inputs_ready", case_id=cid)


def generate(root):
    from dirmo.config import FMConfig
    from dirmo.gpu import inventory, resolve_device, worker_environment
    from dirmo.generation.pose_repair import PoseRepairBackend
    from dirmo.generation.flowr import CapabilityError
    plan = read(root / "study.json")
    device = resolve_device(plan["physical_gpu"], inventory())
    os.environ.update(worker_environment(device, str(ROOT / "src")))
    state = {"status": "loading", "pid": os.getpid(), "physical_gpu": plan["physical_gpu"], "gpu_uuid": device["uuid"]}
    save(root / "gpu_worker.json", state)
    config = FMConfig(gpu=plan["physical_gpu"], port=8877, python=sys.executable,
                      repo_path="/home/uslee/code/flowr_root", checkpoint_path=CHECKPOINT,
                      checkpoint_sha256=CHECKSUM)
    backend = PoseRepairBackend(config, allow_whole_pose=True)
    save(root / "kernel.json", backend.identity)
    pending = [c for c in plan["cases"] if not (case_directory(root, c["case_id"]) / "generation_done.json").exists()]
    consecutive_errors = 0
    try:
        while pending:
            ready = [c for c in pending if (case_directory(root, c["case_id"]) / "inputs_ready.json").exists()]
            if not ready:
                state.update(status="waiting_for_input_preparation", remaining_cases=len(pending))
                save(root / "gpu_worker.json", state)
                time.sleep(3)
                continue
            case = ready[0]
            cid = case["case_id"]
            for record in case["graphs"]:
                name = record["name"]
                unsupported = unsupported_tensor_features(graph_from_record(record))
                for arm in ARMS:
                    items = []
                    free_maps = [record["atom_maps"][i] for i in (range(case["atoms"]) if arm == "whole" else case["free_indices"])]
                    for draw in range(plan["draws_per_graph"]):
                        directory = output_directory(root, cid, name, arm, draw)
                        if (directory / "result.json").exists():
                            continue
                        if unsupported:
                            save(directory / "result.json", {
                                "status": "unsupported_input", "reason": "Input graph has features absent from the frozen FLOWR tensor representation",
                                "unsupported_features": unsupported, "graph_id": record["graph_id"],
                                "source_graph_modified": False})
                            continue
                        source = read(case_directory(root, cid) / "inputs" / name / f"{draw:02d}" / "result.json")
                        if source["status"] != "ready":
                            save(directory / "result.json", {"status": "initialization_failed", "reason": source["reason"]})
                            continue
                        items.append({"case_id": f"{cid}/{name}/{arm}", "draw": draw,
                                      "seed": source["seed"], "free_maps": free_maps,
                                      "pocket_path": case["pocket_path"], "input_sdf": source["sdf"],
                                      "output_dir": str(directory)})
                    capacity = min(8, len(items))
                    while items:
                        batch = items[:capacity]
                        state.update(status="generating", case_id=cid, variant=name, arm=arm,
                                     batch_size=len(batch), remaining_cases=len(pending))
                        save(root / "gpu_worker.json", state)
                        backend.torch.cuda.empty_cache()
                        try:
                            _, stats = backend.generate(batch, steps=plan["steps"])
                            stats.update(case_id=cid, variant=name, arm=arm, status="complete")
                            append_jsonl(root / "batches.jsonl", stats)
                            consecutive_errors = 0
                            items = items[len(batch):]
                        except backend.torch.OutOfMemoryError:
                            append_jsonl(root / "batches.jsonl", {"status": "oom", "case_id": cid,
                                         "variant": name, "arm": arm, "batch_size": len(batch)})
                            backend.torch.cuda.empty_cache()
                            if len(batch) > 1:
                                capacity = max(1, len(batch) // 2)
                                continue
                            save(Path(batch[0]["output_dir"]) / "result.json", {"status": "oom"})
                            items = items[1:]
                        except CapabilityError as exc:
                            # Input-specific round-trip failures must not stop the
                            # cohort. Check single inputs before discarding a batch.
                            if str(exc) != "Tensorization changed the graph or atom order":
                                raise
                            append_jsonl(root / "input_validation_errors.jsonl", {
                                "case_id": cid, "variant": name, "arm": arm,
                                "batch_size": len(batch), "reason": str(exc)})
                            if len(batch) > 1:
                                capacity = 1
                                continue
                            save(Path(batch[0]["output_dir"]) / "result.json", {
                                "status": "unsupported_input", "reason": f"{type(exc).__name__}: {exc}",
                                "source_graph_modified": False})
                            items = items[1:]
                        except Exception as exc:
                            traceback.print_exc()
                            for item in batch:
                                save(Path(item["output_dir"]) / "result.json", {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
                            consecutive_errors += 1
                            items = items[len(batch):]
                            if consecutive_errors >= 8:
                                raise RuntimeError("Eight consecutive FM batch errors; stopped to avoid consuming the cohort on a broken input path") from exc
            statuses = Counter(read(output_directory(root, cid, rec["name"], arm, draw) / "result.json")["status"]
                               for rec in case["graphs"] for arm in ARMS for draw in range(plan["draws_per_graph"]))
            save(case_directory(root, cid) / "generation_done.json", {
                "case_id": cid, "scheduled": len(case["graphs"]) * len(ARMS) * plan["draws_per_graph"],
                "status": "accounted_with_missing" if any(k not in ("ok", "contract_failed") for k in statuses) else "complete",
                "slot_statuses": dict(statuses)})
            pending.remove(case)
            emit(status="generation_case_complete", case_id=cid, remaining=len(pending))
        state.update(status="complete", remaining_cases=0)
        save(root / "gpu_worker.json", state)
    except BaseException as exc:
        state.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
        save(root / "gpu_worker.json", state)
        raise


def evaluate_case(root, cid):
    from Bio.PDB import PDBParser
    from dirmo.evaluation.fm_quality import prepare_receptor, vina_for_mol
    from dirmo.evaluation.pb_guided import uff_control
    from dirmo.evaluation.pb_worker import bounded_measure
    plan = read(root / "study.json")
    case = read(case_directory(root, cid) / "case.json")
    os.environ.update(CUDA_VISIBLE_DEVICES="", DIRMO_PB_ENERGY_CACHE=str(root / "cache/reference_energy"),
                      DIRMO_PB_MEASURE_CACHE=str(root / "cache/measurements"))
    directory = root / "evaluations" / cid
    directory.mkdir(parents=True, exist_ok=True)
    receptor_meta = directory / "receptor-preparation.json"
    if not receptor_meta.exists():
        protein = PDBParser(QUIET=True).get_structure(cid, case["protein_path"])
        receptor, receptor_status = prepare_receptor(protein, directory)
    else:
        receptor_status = read(receptor_meta)["status"]
        receptor = str(directory / "receptor.pdbqt") if receptor_status == 0 else None
    native = read_sdf(case["native_sdf"])
    score_case = {"receptor_path": receptor, "box_center": native.GetConformer().GetPositions().mean(axis=0).tolist(),
                  "box_size": [40., 40., 40.]}
    held = {int(i): xyz for i, xyz in case["held_xyz"].items()}
    result_path = directory / "result.json"
    result = read(result_path, {"case_id": cid, "rows": {}, "receptor_status": receptor_status,
                               "vina_protocol": "Vina score-only; original standard-aa heavy XYZ; Meeko HIE/Gasteiger; no water/cofactors; no docking/minimization; same 40 A box", "complete": False})

    def measure(key, path, graph, kind, variant, draw, status):
        if key in result["rows"]:
            return
        row = {"kind": kind, "variant": variant, "draw": draw, "status": status, "sdf": path}
        if not path or not Path(path).exists():
            row.update(pb={"passes": False, "missing": True}, vina={"status": "missing", "reason": status},
                       contract={"graph_preserved": False, "core_preserved": False, "core_max_drift": None}, success=False)
        else:
            row["contract"] = assess_saved_graph(graph, path, held)
            row["pb"] = bounded_measure(path, case["pocket_path"])
            try:
                row["vina"] = vina_for_mol(read_sdf(path), score_case)
            except Exception as exc:
                row["vina"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            require_core = kind in ("fm_core", "uff_core", "prior", "native")
            row["success"] = bool(row["contract"]["graph_preserved"] and row["pb"]["passes"]
                                  and (not require_core or row["contract"]["core_preserved"]))
        result["rows"][key] = row
        save(result_path, result)

    measure("native", case["native_sdf"], native, "native", "original", -1, "reference")
    for record in case["graphs"]:
        name, graph = record["name"], graph_from_record(record)
        for draw in range(plan["draws_per_graph"]):
            source_dir = case_directory(root, cid) / "inputs" / name / f"{draw:02d}"
            source = read(source_dir / "result.json")
            path = source.get("sdf")
            measure(f"{name}/{draw}/prior", path, graph, "prior", name, draw, source["status"])
            for arm in ARMS:
                generated = read(output_directory(root, cid, name, arm, draw) / "result.json", {})
                measure(f"{name}/{draw}/fm_{arm}", generated.get("sdf_path"), graph, f"fm_{arm}", name, draw,
                        generated.get("status", "missing"))
                uff_dir = source_dir / f"uff_{arm}"
                uff_result = uff_dir / "result.json"
                if not uff_result.exists():
                    try:
                        if not path:
                            raise ValueError("No graph-only input conformer")
                        free = list(range(case["atoms"])) if arm == "whole" else case["free_indices"]
                        uff_path = uff_dir / "raw.sdf"
                        details = uff_control(path, free, uff_path)
                        save(uff_result, {"status": "ok", "sdf": str(uff_path), **details})
                    except Exception as exc:
                        save(uff_result, {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
                uff = read(uff_result)
                measure(f"{name}/{draw}/uff_{arm}", uff.get("sdf"), graph, f"uff_{arm}", name, draw, uff["status"])
    result["complete"] = True
    save(result_path, result)
    emit(status="evaluation_case_complete", case_id=cid, rows=len(result["rows"]))


def summarize(root, status="running", detailed=False):
    plan = read(root / "study.json")
    counts, groups, eval_cases, errors = Counter(), defaultdict(list), 0, []
    for case in plan["cases"]:
        cid = case["case_id"]
        counts["prepared_cases"] += (case_directory(root, cid) / "inputs_ready.json").exists()
        counts["generation_complete_cases"] += (case_directory(root, cid) / "generation_done.json").exists()
        for record in case["graphs"]:
            for arm in ARMS:
                for draw in range(plan["draws_per_graph"]):
                    entry = read(output_directory(root, cid, record["name"], arm, draw) / "result.json")
                    if entry:
                        counts["fm_terminal_slots"] += 1
                        counts[f"fm_{entry['status']}"] += 1
        evaluated = read(root / "evaluations" / cid / "result.json", {})
        if evaluated.get("complete"):
            eval_cases += 1
        for row in evaluated.get("rows", {}).values():
            groups[row["variant"], row["kind"]].append({"case_id": cid, **row})
        failure = read(root / "evaluations" / cid / "worker_failed.json")
        if failure:
            errors.append({"case_id": cid, **failure})
    batches = []
    history = root / "batches.jsonl"
    if history.exists():
        for line in history.read_text().splitlines():
            try:
                batches.append(json.loads(line))
            except ValueError:
                pass
    completed = [b for b in batches if b["status"] == "complete"]
    actual_samples = sum(b["batch_size"] for b in completed)
    seconds = sum(b["inference_seconds"] for b in completed)
    entries = []
    for (variant, kind), rows in sorted(groups.items()):
        expected = plan["cases_requested"] * (1 if kind == "native" else plan["draws_per_graph"])
        by_case = defaultdict(list)
        for row in rows:
            by_case[row["case_id"]].append(row)
        vina = [r["vina"]["score"] for r in rows if r["vina"].get("status") == "ok"]
        entries.append({"variant": variant, "kind": kind, "evaluated": len(rows), "scheduled": expected,
                        "pb_valid": sum(r["pb"]["passes"] for r in rows),
                        "pb_missing": sum(bool(r["pb"].get("missing")) for r in rows),
                        "graph_preserved": sum(r["contract"]["graph_preserved"] for r in rows),
                        "core_preserved": sum(r["contract"]["core_preserved"] for r in rows),
                        "success": sum(r["success"] for r in rows),
                        "any_success_cases": sum(any(r["success"] for r in rr) for rr in by_case.values()),
                        "vina_measured": len(vina), "vina_median": float(np.median(vina)) if vina else None})
    summary = {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
               "cases": plan["cases_requested"], "planned_fm_outputs": plan["planned_fm_outputs"],
               **counts, "evaluation_complete_cases": eval_cases, "gpu": read(root / "gpu_worker.json", {}),
               "actual_fm_samples": actual_samples, "gpu_inference_seconds": seconds,
               "gpu_eta_seconds_from_completed_batches": ((plan["planned_fm_outputs"] - counts["fm_terminal_slots"]) * seconds / actual_samples) if actual_samples else None,
               "groups": entries, "evaluation_worker_failures": errors}
    save(root / "summary.json", summary)
    lines = [f"# 화학 graph 편집 후 FLOWR.root 생성: {plan['cases_requested']} pockets", "",
             f"상태: {status}. GPU {plan['physical_gpu']}; base checkpoint; {plan['steps']}-step.", "",
             f"입력 준비 {counts['prepared_cases']}/{plan['cases_requested']}, GPU 완료 pocket {counts['generation_complete_cases']}/{plan['cases_requested']}, FM 결과 슬롯 {counts['fm_terminal_slots']}/{plan['planned_fm_outputs']}, CPU 평가 완료 {eval_cases}/{plan['cases_requested']}.", "",
             "원본 graph와 terminal C→N/O/F 세 편집 graph를 비교한다. 모든 편집은 원자 수·결합·atom map을 유지하며 지표 순위로 고르지 않았다. 두 FM 조건 모두 편집 후 전체 graph를 고정한다. whole은 전체 좌표 자유, core는 원본 core 좌표 고정이다.", "",
             "초기 좌표는 graph-only ETKDG 후 고정 core의 국소 anchor에 배치했다. 원본 자유 영역 좌표를 초기화에 사용하지 않는다. 두 FM 조건은 같은 시작 SDF를 공유하므로, whole도 초기 상태의 core 위치 정보를 받는다. 완전한 blind docking 실험이 아니다.", "",
             "PB는 같은 고정 protein-only pocket의 dock 전체 20 검사다. 성공은 PB+graph/stereo 보존이며 core 조건은 보호 좌표 최대 변위 ≤0.001 Å도 요구한다. 새 편집 분자에는 실험 정답 pose가 없어 native free RMSD를 정확도로 사용하지 않는다.", "",
             "Vina는 별도 standard-aa receptor의 score-only 대리 점수다. 수용체 준비·PB 실패/누락은 별도로 유지하고 전체 예정 분모에서 제거하지 않는다. 각 graph의 입력/whole-UFF/core-UFF 대조군도 평가한다.", "",
             "| Graph | 방법 | 평가/예정 | PB 통과 | Graph 보존 | 조건 포함 성공 | 8개 중 성공 pocket | Vina 실측 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for e in entries:
        lines.append(f"| {e['variant']} | {e['kind']} | {e['evaluated']}/{e['scheduled']} | {e['pb_valid']} | {e['graph_preserved']} | {e['success']} | {e['any_success_cases']} | {e['vina_measured']} |")
    lines += ["", "실행 중 집계는 부분 결과다. 입력 초기화·생성 실패도 전체 예정 FM 슬롯에 남긴다. 사전 contact 기준은 공유결합 확인 자료가 아니라 매우 가까운 native contact를 제외하는 보수적 적격성 기준이다. 원본 PB 통과 여부로 cohort를 바꾸지 않았다. 사전학습 중복은 미확인이다.", "",
              "코드·조건: study.json, cases/*/case.json. 초기 SDF: cases/*/inputs. FM raw: draws. PB/Vina: evaluations/*/result.json. 로그: gpu.log, preparation/*.log, evaluations/*/worker.log. 실행 재개는 같은 study.json과 저장된 결과를 유지한다."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    if detailed:
        import csv
        rows = []
        for rr in groups.values():
            for r in rr:
                rows.append({"case_id": r["case_id"], "variant": r["variant"], "kind": r["kind"],
                             "draw": r["draw"], "status": r["status"], "pb_valid": r["pb"]["passes"],
                             "pb_missing": bool(r["pb"].get("missing")), "success": r["success"],
                             **r["contract"], "vina_status": r["vina"]["status"], "vina": r["vina"].get("score"),
                             "failed_checks": ";".join(r["pb"].get("failed_checks", [])), "sdf": r["sdf"]})
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (root / "metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        chemical = [{"case_id": c["case_id"], "variant": g["name"], **g["properties"]}
                    for c in plan["cases"] for g in c["graphs"]]
        with (root / "chemical_properties.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(chemical[0]))
            writer.writeheader()
            writer.writerows(chemical)
    return summary


def cpu_command(root, mode, cid, timeout):
    directory = root / ("preparation" if mode == "prepare-case" else "evaluations") / cid
    directory.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    command = [sys.executable, str(Path(__file__).resolve()), mode, "--output", str(root), "--case", cid]
    for attempt in range(2):
        with (directory / "worker.log").open("a") as log:
            try:
                process = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log, timeout=timeout)
                if process.returncode == 0:
                    return {"case_id": cid, "status": "complete"}
                reason = f"CPU worker exited {process.returncode}"
            except subprocess.TimeoutExpired:
                reason = f"CPU case exceeded {timeout} seconds"
        append_jsonl(root / "cpu_retries.jsonl", {"case_id": cid, "mode": mode, "attempt": attempt + 1, "reason": reason})
    save(directory / "worker_failed.json", {"status": "failed", "reason": reason})
    return {"case_id": cid, "status": "failed", "reason": reason}


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"status": "preparing_cohort", "pid": os.getpid(), "physical_gpu": args.gpu,
                 "started_utc": datetime.now(timezone.utc).isoformat()}
        save(root / "controller.json", state)
        plan = prepare_study(args)
        prep_pool, eval_pool = ThreadPoolExecutor(max_workers=4), ThreadPoolExecutor(max_workers=args.workers)
        gpu_log = (root / "gpu.log").open("a")
        gpu = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "generate", "--output", str(root)],
                               cwd=ROOT, stdout=gpu_log, stderr=gpu_log,
                               env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        state.update(status="running", gpu_pid=gpu.pid)
        save(root / "controller.json", state)
        prep = {c["case_id"]: prep_pool.submit(cpu_command, root, "prepare-case", c["case_id"], 3600)
                for c in plan["cases"] if not (case_directory(root, c["case_id"]) / "inputs_ready.json").exists()}
        evaluations = {}
        try:
            while True:
                for cid, future in prep.items():
                    if future.done() and future.result()["status"] == "failed":
                        # Preserve all scheduled slots when CPU initialization fails.
                        case = next(c for c in plan["cases"] if c["case_id"] == cid)
                        for g in case["graphs"]:
                            for draw in range(plan["draws_per_graph"]):
                                marker = case_directory(root, cid) / "inputs" / g["name"] / f"{draw:02d}" / "result.json"
                                if not marker.exists():
                                    save(marker, {"status": "initialization_failed", "reason": future.result()["reason"]})
                        save(case_directory(root, cid) / "inputs_ready.json", {"status": "ready_with_missing", "case_id": cid})
                for c in plan["cases"]:
                    cid = c["case_id"]
                    if cid not in evaluations and (case_directory(root, cid) / "generation_done.json").exists():
                        if not read(root / "evaluations" / cid / "result.json", {}).get("complete"):
                            evaluations[cid] = eval_pool.submit(cpu_command, root, "evaluate-case", cid, 21600)
                        else:
                            evaluations[cid] = None
                summary = summarize(root)
                if gpu.poll() is not None and gpu.returncode != 0:
                    raise RuntimeError(f"GPU worker exited {gpu.returncode}; see gpu.log")
                if summary["generation_complete_cases"] == len(plan["cases"]) and len(evaluations) == len(plan["cases"]) and all(f is None or f.done() for f in evaluations.values()):
                    missing = any(e["pb_missing"] for e in summary["groups"]) or summary["evaluation_worker_failures"]
                    final = "complete_with_missing_evaluation" if missing else "complete"
                    summary = summarize(root, final, detailed=True)
                    state.update(status=final, completed_utc=datetime.now(timezone.utc).isoformat())
                    save(root / "controller.json", state)
                    emit(status=final, output=str(root), fm_slots=summary["fm_terminal_slots"])
                    break
                time.sleep(15)
        except BaseException as exc:
            state.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            save(root / "controller.json", state)
            if gpu.poll() is None:
                gpu.terminate()
            summarize(root, "failed", detailed=True)
            raise
        finally:
            prep_pool.shutdown(wait=False)
            eval_pool.shutdown(wait=False)
            gpu_log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "prepare", "prepare-case", "generate", "evaluate-case", "summarize"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case")
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--source", type=Path, default=ROOT / "runs/fm_pose_gpu4_2000_batch32_v1/steps_64/study.json")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.mode == "run":
        run(args)
    elif args.mode == "prepare":
        prepare_study(args)
    elif args.mode == "prepare-case":
        prepare_case(args.output, args.case)
    elif args.mode == "generate":
        generate(args.output)
    elif args.mode == "evaluate-case":
        evaluate_case(args.output, args.case)
    else:
        summarize(args.output, detailed=True)


if __name__ == "__main__":
    main()
