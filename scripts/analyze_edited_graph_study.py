"""Reproduce the 2026-10-18 partial-study report, without generation or scoring.

The narrative contains findings for this frozen 43-generated/30-evaluated
snapshot. Refuse a changed cohort rather than reuse those findings silently.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
from rdkit import Chem
from dirmo.chemistry.graphs import read_sdf, smiles, stereo_from_coordinates
from dirmo.evaluation.edited_graph import graph_from_record
from dirmo.io import sha256_file

VARIANTS = ["original", "C_to_N", "C_to_O", "C_to_F"]
KINDS = ["prior", "uff_whole", "uff_core", "fm_whole", "fm_core"]
LABELS = {"prior": "Starting pose", "uff_whole": "UFF whole", "uff_core": "UFF core",
          "fm_whole": "FM whole", "fm_core": "FM core", "native": "Native"}
COLORS = ["#9aa4b2", "#8b6caf", "#ca9b44", "#3877be", "#179989"]


def load(p):
    return json.loads(p.read_text())


def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def boot_ci(values, statistic="mean"):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return [None, None]
    rng = np.random.default_rng(20261018)
    samples = x[rng.integers(0, len(x), size=(5000, len(x)))]
    res = samples.mean(axis=1) if statistic == "mean" else np.median(samples, axis=1)
    return np.quantile(res, [.025, .975]).tolist()


def ratio(a, b):
    return f"{a}/{b} ({100*a/b:.1f}%)" if b else "—"


def markdown(frame):
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    rows.extend("| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def index_signature(mol):
    atoms = tuple((a.GetAtomicNum(), a.GetFormalCharge(), a.GetIsotope(), a.GetIsAromatic(),
                   a.GetNumRadicalElectrons(), a.GetTotalNumHs()) for a in mol.GetAtoms())
    bonds = tuple(sorted((min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                          max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()), str(b.GetBondType()))
                         for b in mol.GetBonds()))
    return atoms, bonds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.run.resolve(), args.output.resolve()
    plan = load(root / "study.json")
    generated = sum((root / "cases" / c["case_id"] / "generation_done.json").exists() for c in plan["cases"])
    evaluated = sum(load(p).get("complete", False) for p in (root / "evaluations").glob("*/result.json"))
    if (len(plan["cases"]), generated, evaluated, plan["steps"], plan["draws_per_graph"]) != (100, 43, 30, 64, 8):
        raise ValueError("The saved cohort changed; update the dated report narrative before analyzing a resumed/completed experiment")
    out.mkdir(parents=True, exist_ok=True)
    cases = {c["case_id"]: c for c in plan["cases"]}
    snapshot = out / "controller_exit_snapshot"
    if not snapshot.exists():
        snapshot.mkdir()
        for name in ["summary.json", "REPORT.md", "metrics.csv", "controller.json", "gpu_worker.json"]:
            if (root / name).exists():
                shutil.copy2(root / name, snapshot / name)
    # Refresh counters after CPU workers that outlived the GPU/controller finished.
    spec = importlib.util.spec_from_file_location("edited_benchmark", ROOT / "scripts/benchmark_edited_graph.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    summary = bench.summarize(root, load(root / "controller.json")["status"], detailed=True)

    rows, completed_cases, receptor = [], [], []
    for cid, case in cases.items():
        p = root / "evaluations" / cid / "result.json"
        if not p.exists():
            continue
        d = load(p)
        if not d.get("complete"):
            continue
        assert len(d["rows"]) == 1 + len(VARIANTS) * plan["draws_per_graph"] * len(KINDS)
        completed_cases.append(cid)
        receptor.append({"case_id": cid, "status": d["receptor_status"],
                         "log": str(root / "evaluations" / cid / "receptor-preparation.log")})
        for row in d["rows"].values():
            rows.append({"case_id": cid, "variant": row["variant"], "kind": row["kind"],
                         "draw": row["draw"], "status": row["status"],
                         "sdf_present": bool(row["sdf"] and Path(row["sdf"]).exists()),
                         "sdf": row["sdf"], "pb_valid": bool(row["pb"]["passes"]),
                         "pb_missing": bool(row["pb"].get("missing")), "success": bool(row["success"]),
                         **row["contract"], "vina_status": row["vina"]["status"],
                         "vina": row["vina"].get("score"),
                         "failed_checks": ";".join(row["pb"].get("failed_checks", []))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "evaluated_rows.csv", index=False)
    pd.DataFrame(receptor).to_csv(out / "receptor_availability.csv", index=False)
    complete = set(completed_cases)
    n = len(complete)
    assert n and len(df) == n * 161
    metrics = []
    for (variant, kind), g in df.groupby(["variant", "kind"]):
        valid = g[g.sdf_present]
        bycase = g.groupby("case_id").success.mean()
        metrics.append({"variant": variant, "kind": kind, "scheduled": len(g),
                        "sdf_present": int(g.sdf_present.sum()), "pb_valid": int(g.pb_valid.sum()),
                        "pb_missing": int(g.pb_missing.sum()), "success": int(g.success.sum()),
                        "success_rate": float(g.success.mean()), "success_ci": boot_ci(bycase),
                        "graph_preserved": int(g.graph_preserved.sum()),
                        "core_preserved": int(g.core_preserved.sum()),
                        "max_core_drift": float(valid.core_max_drift.max()) if len(valid) else None,
                        "any_success_cases": int(g.groupby("case_id").success.any().sum()),
                        "vina_measured": int((g.vina_status == "ok").sum())})
    save(out / "group_metrics.json", metrics)
    pd.DataFrame([{k: v for k, v in x.items() if k != "success_ci"} for x in metrics]).to_csv(out / "group_metrics.csv", index=False)
    pooled = []
    for kind in ["native"] + KINDS:
        g = df[df.kind == kind]
        pooled.append({"kind": kind, "scheduled": len(g), "sdf_present": int(g.sdf_present.sum()),
                       "pb_valid": int(g.pb_valid.sum()), "success": int(g.success.sum()),
                       "graph_preserved": int(g.graph_preserved.sum()),
                       "success_rate": float(g.success.mean()),
                       "success_ci": boot_ci(g.groupby("case_id").success.mean()),
                       "vina_measured": int((g.vina_status == "ok").sum())})

    # Independent audits use saved target graphs and saved final SDFs, never
    # setting output stereo from the requested graph to make a result match.
    audit_rows, conditions, radical_records = [], [], []
    input_status = Counter()
    for cid, case in cases.items():
        held = case["held_indices"]
        held_xyz = np.array([case["held_xyz"][str(i)] for i in held])
        for record in case["graphs"]:
            graph = graph_from_record(record)
            signature = index_signature(graph)
            target = smiles(graph)
            radicals = sum(a.GetNumRadicalElectrons() for a in graph.GetAtoms())
            if radicals:
                radical_records.append({"case_id": cid, "variant": record["name"], "radical_electrons": radicals,
                                        "smiles": target})
            for draw in range(plan["draws_per_graph"]):
                src = root / "cases" / cid / "inputs" / record["name"] / f"{draw:02d}" / "result.json"
                source = load(src)
                input_status[source["status"]] += 1
                source_hash_valid = source.get("sha256") == sha256_file(source["sdf"]) if source["status"] == "ready" else False
                for arm in ("whole", "core"):
                    d = root / "draws" / cid / record["name"] / arm / f"{draw:02d}"
                    p = d / "result.json"
                    if not p.exists():
                        continue
                    item = load(p)
                    if item.get("sdf_path"):
                        mol = read_sdf(item["sdf_path"])
                        actual = smiles(stereo_from_coordinates(mol))
                        drift = float(np.linalg.norm(mol.GetConformer().GetPositions()[held] - held_xyz, axis=1).max())
                        audit_rows.append({"case_id": cid, "variant": record["name"], "draw": draw, "arm": arm,
                                           "status": item["status"], "in_complete_evaluation": cid in complete,
                                           "index_graph_preserved": index_signature(mol) == signature,
                                           "strict_stereo_preserved": actual == target,
                                           "core_max_drift": drift, "core_preserved": drift <= .001,
                                           "expected_smiles": target, "actual_smiles": actual})
                    if (d / "condition.json").exists():
                        condition = load(d / "condition.json")
                        intended = [False] * case["atoms"] if arm == "whole" else [i in held for i in range(case["atoms"])]
                        actual_mask = condition["coordinate_fixed_mask"]
                        conditions.append({"case_id": cid, "variant": record["name"], "draw": draw, "arm": arm,
                                           "coordinate_mask_correct": actual_mask[:case["atoms"]] == intended and not any(actual_mask[case["atoms"]:]),
                                           "graph_mask_correct": all(condition["graph_fixed_mask"][:case["atoms"]]) and not any(condition["graph_fixed_mask"][case["atoms"]:]),
                                           "input_hash_correct": source_hash_valid and condition["input_sha256"] == source["sha256"],
                                           "input_hash": condition["input_sha256"]})
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out / "independent_graph_coordinate_audit.csv", index=False)
    cond = pd.DataFrame(conditions)
    cond.to_csv(out / "condition_audit.csv", index=False)
    pairs = cond.pivot(index=["case_id", "variant", "draw"], columns="arm", values="input_hash").dropna()
    audits = {"saved_sdf_count": len(audit), "by_arm": {}, "condition_count": len(cond),
              "coordinate_masks_correct": int(cond.coordinate_mask_correct.sum()),
              "graph_masks_correct": int(cond.graph_mask_correct.sum()),
              "input_hashes_correct": int(cond.input_hash_correct.sum()),
              "whole_core_input_pairs": len(pairs), "same_input_pairs": int((pairs.whole == pairs.core).sum()),
              "all_prepared_input_status": dict(input_status), "radical_graphs": radical_records}
    for arm, g in audit.groupby("arm"):
        audits["by_arm"][arm] = {"saved_sdf_count": len(g), "index_graph_preserved": int(g.index_graph_preserved.sum()),
                                 "strict_stereo_preserved": int(g.strict_stereo_preserved.sum()),
                                 "core_preserved": int(g.core_preserved.sum()), "max_core_drift": float(g.core_max_drift.max())}
    save(out / "preservation_audit.json", audits)
    assert len(audit) == summary["fm_ok"] + summary["fm_contract_failed"]
    assert int(audit.strict_stereo_preserved.sum()) == summary["fm_ok"]
    assert audit.index_graph_preserved.all()
    assert cond.coordinate_mask_correct.all() and cond.graph_mask_correct.all() and cond.input_hash_correct.all()
    assert (pairs.whole == pairs.core).all()
    assert sum(summary.get(f"fm_{status}", 0) for status in (
        "ok", "contract_failed", "initialization_failed", "failed", "build_failed", "oom")) == summary["fm_terminal_slots"]

    # Pair each output to exactly the same graph, draw and saved starting pose.
    pairing = ["case_id", "variant", "draw"]
    prior = df[df.kind == "prior"][pairing + ["vina", "success", "graph_preserved", "pb_valid"]]
    vina_stats, paired_records = [], []
    for kind in KINDS[1:]:
        g = df[df.kind == kind].merge(prior, on=pairing, suffixes=("", "_prior"))
        g = g[g.vina.notna() & g.vina_prior.notna()].copy()
        g["delta_vina"] = g.vina - g.vina_prior
        g.to_csv(out / f"paired_vina_{kind}_vs_prior.csv", index=False)
        for subset in ("all_scored", "both_graph_preserved", "both_strict_success"):
            sub = g if subset == "all_scored" else g[g.graph_preserved & g.graph_preserved_prior] if subset == "both_graph_preserved" else g[g.success & g.success_prior]
            pocket_medians = sub.groupby("case_id").delta_vina.median()
            vina_stats.append({"kind": kind, "subset": subset, "pairs": len(sub), "pockets": len(pocket_medians),
                               "median_delta": float(sub.delta_vina.median()) if len(sub) else None,
                               "improved": int((sub.delta_vina < 0).sum()),
                               "pocket_median_delta": float(pocket_medians.median()) if len(sub) else None,
                               "pocket_median_ci": boot_ci(pocket_medians, "median")})
            for cid, value in pocket_medians.items():
                paired_records.append({"case_id": cid, "kind": kind, "subset": subset, "delta_vina": value})
    save(out / "paired_vina_summary.json", vina_stats)
    pd.DataFrame(paired_records).to_csv(out / "paired_vina_pocket_medians.csv", index=False)

    # Edit-vs-original score comparisons use one mean per pocket/graph among
    # strict-success poses. Seeds differ between graphs: not common-noise pairs.
    edit_pairs, native_pairs = [], []
    for kind in ("fm_whole", "fm_core"):
        strict = df[(df.kind == kind) & df.success & df.vina.notna()]
        means = strict.groupby(["case_id", "variant"]).vina.mean().unstack()
        for variant in VARIANTS[1:]:
            for cid, values in means[["original", variant]].dropna().iterrows():
                edit_pairs.append({"case_id": cid, "kind": kind, "variant": variant,
                                   "original_vina_mean": values["original"], "edited_vina_mean": values[variant],
                                   "delta_vina": values[variant] - values["original"]})
        original = means["original"].dropna()
        native = df[(df.kind == "native") & df.pb_valid & df.vina.notna()].set_index("case_id").vina
        for cid in original.index.intersection(native.index):
            native_pairs.append({"case_id": cid, "kind": kind, "native_vina": native[cid],
                                 "generated_original_mean": original[cid], "delta_vina": original[cid]-native[cid]})
    edits = pd.DataFrame(edit_pairs)
    edits.to_csv(out / "edited_vs_original_successful_pose_vina.csv", index=False)
    natives = pd.DataFrame(native_pairs)
    natives.to_csv(out / "generated_original_vs_native_vina.csv", index=False)
    edit_summary = []
    for (kind, variant), g in edits.groupby(["kind", "variant"]):
        edit_summary.append({"kind": kind, "variant": variant, "pockets": len(g),
                             "median_delta": float(g.delta_vina.median()), "improved_pockets": int((g.delta_vina < 0).sum()),
                             "median_ci": boot_ci(g.delta_vina, "median")})
    save(out / "edited_vina_summary.json", edit_summary)

    chemical = pd.DataFrame([{"case_id": cid, "variant": g["name"], "in_evaluated_cohort": cid in complete,
                              **g["properties"]} for cid, c in cases.items() for g in c["graphs"]])
    chemical.to_csv(out / "chemical_properties_all_100.csv", index=False)
    chemical_summary, chemical_delta = [], []
    for cohort, data in (("evaluated_30", chemical[chemical.in_evaluated_cohort]), ("selected_100", chemical)):
        for prop in ["qed", "sa", "logp", "mw", "hbd", "hba", "tpsa"]:
            wide = data.pivot(index="case_id", columns="variant", values=prop)
            for variant in VARIANTS[1:]:
                change = wide[variant] - wide.original
                chemical_summary.append({"cohort": cohort, "variant": variant, "property": prop, "pockets": len(change),
                                         "mean_delta": float(change.mean()), "median_delta": float(change.median()),
                                         "mean_ci": boot_ci(change), "increased": int((change > 1e-9).sum()),
                                         "decreased": int((change < -1e-9).sum())})
                for cid, value in change.items():
                    chemical_delta.append({"cohort": cohort, "case_id": cid, "variant": variant, "property": prop, "delta": value})
    save(out / "chemical_change_summary.json", chemical_summary)
    pd.DataFrame(chemical_delta).to_csv(out / "chemical_changes.csv", index=False)

    failures = []
    for kind in KINDS:
        g = df[(df.kind == kind) & df.sdf_present]
        counts = Counter(check for checks in g.failed_checks for check in checks.split(";") if check)
        failures += [{"kind": kind, "check": check, "failed": count, "sdf_denominator": len(g)} for check, count in counts.items()]
    pd.DataFrame(failures).to_csv(out / "pb_failure_breakdown.csv", index=False)

    paired_binary = []
    for comparison in ["prior", "uff_core", "fm_core"]:
        a = df[df.kind == "fm_whole"][pairing+["success"]]
        b = df[df.kind == comparison][pairing+["success"]]
        m = a.merge(b, on=pairing, suffixes=("_whole", "_other"))
        delta = m.success_whole.astype(float)-m.success_other.astype(float)
        pocket_delta = pd.DataFrame({"case_id": m.case_id, "delta": delta}).groupby("case_id").delta.mean()
        paired_binary.append({"comparison": f"fm_whole_minus_{comparison}", "pairs": len(m),
                              "mean_difference": float(delta.mean()), "ci": boot_ci(pocket_delta),
                              "whole_only_success": int((m.success_whole & ~m.success_other).sum()),
                              "other_only_success": int((~m.success_whole & m.success_other).sum())})

    overview = {"analysis_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), "run": str(root),
                "status": summary["status"], "selected_cases": len(cases),
                "generation_complete_cases": summary["generation_complete_cases"],
                "evaluated_complete_cases": n, "evaluated_case_ids": completed_cases,
                "fm_counts": {k: v for k, v in summary.items() if k.startswith("fm_")},
                "planned_fm_outputs": plan["planned_fm_outputs"],
                "not_attempted_slots": plan["planned_fm_outputs"]-summary["fm_terminal_slots"],
                "pooled": pooled, "paired_binary": paired_binary,
                "vina_receptor_success_cases": sum(x["status"] == 0 for x in receptor),
                "vina_receptor_failed_cases": sum(x["status"] != 0 for x in receptor),
                "generation_only_case_ids": [cid for cid in cases if (root / "cases" / cid / "generation_done.json").exists() and cid not in complete],
                "analysis_code_sha256": sha256_file(__file__)}
    save(out / "analysis.json", overview)
    plots(df, metrics, pooled, pd.DataFrame(paired_records), edits, chemical_summary, out)
    report(plan, overview, metrics, audits, vina_stats, edit_summary, chemical_summary, natives, out)
    print(json.dumps({"output": str(out), "status": overview["status"], "generation_complete_cases": summary["generation_complete_cases"],
                      "evaluated_complete_cases": n, "saved_sdf_count": len(audit), "rows": len(df)}, ensure_ascii=False))


def plots(df, metrics, pooled, pocket_vina, edits, chemicals, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 170, "figure.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), layout="constrained")
    ax = axes[0]
    x = np.arange(len(KINDS))
    entries = {g["kind"]: g for g in pooled}
    ax.bar(x-.18, [100*entries[k]["pb_valid"]/entries[k]["scheduled"] for k in KINDS], .36, color="#b9cbe1", label="PB only")
    bars = ax.bar(x+.18, [100*entries[k]["success_rate"] for k in KINDS], .36, color=COLORS, label="PB + graph/stereo (+ core if required)")
    for b, k in zip(bars, KINDS):
        e = entries[k]
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.3, f"{100*e['success_rate']:.1f}%", ha="center", fontsize=9)
    ax.axhline(90, ls="--", color="#b44e4e", lw=1.1, label="Native: 27/30 PB valid")
    ax.set(xticks=x, xticklabels=[LABELS[k].replace(" ", "\n", 1) for k in KINDS], ylim=(0, 106), ylabel="Rate (% of scheduled poses)",
           title="A. Same 30 pockets; 960 slots per method")
    ax.legend(fontsize=7, loc="upper left")
    ax = axes[1]
    for i, (kind, color) in enumerate(zip(KINDS, COLORS)):
        es = [next(e for e in metrics if e["kind"] == kind and e["variant"] == v) for v in VARIANTS]
        xx = np.arange(4)+(i-2)*.15
        ax.bar(xx, [100*e["success_rate"] for e in es], .145, color=color, label=LABELS[kind])
    ax.set(xticks=range(4), xticklabels=["Original", "C to N", "C to O", "C to F"], ylim=(0, 100),
           ylabel="Strict success (%)", title="B. Graph variants; 240 slots per bar")
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Frozen FLOWR.root | 64 steps | partial cohort: 30 of 100 evaluated", fontsize=13)
    fig.savefig(out / "generation_quality.png")
    fig.savefig(out / "generation_quality.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    ax = axes[0]
    data = pocket_vina[pocket_vina.subset == "all_scored"]
    for i, (kind, color) in enumerate(zip(KINDS[1:], COLORS[1:])):
        vals = data[data.kind == kind].delta_vina.to_numpy()
        ax.boxplot([vals], positions=[i], widths=.5, patch_artist=True, showfliers=False,
                   boxprops={"facecolor": color, "alpha": .55}, medianprops={"color": "black"})
        jitter = np.random.default_rng(i).uniform(-.12, .12, len(vals))
        ax.scatter(i+jitter, vals, color=color, s=19, alpha=.8)
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_yscale("symlog", linthresh=5)
    ax.set(xticks=range(4), xticklabels=[LABELS[k].replace(" ", "\n") for k in KINDS[1:]],
           ylabel="Delta Vina vs starting pose (kcal/mol; symlog)", title="A. One median per pocket; 19 receptors\nAll scored poses; linear axis within +/-5")
    ax = axes[1]
    for i, variant in enumerate(VARIANTS[1:]):
        for j, (kind, color) in enumerate(zip(["fm_whole", "fm_core"], COLORS[-2:])):
            values = edits[(edits.variant == variant) & (edits.kind == kind)].delta_vina.to_numpy()
            xx = i+(j-.5)*.32
            ax.scatter(xx+np.random.default_rng(i+j).uniform(-.045, .045, len(values)), values, color=color, alpha=.6, s=19,
                       label=LABELS[kind] if i == 0 else None)
            ax.plot([xx-.09, xx+.09], [np.median(values)]*2, color=color, lw=3)
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set(xticks=range(3), xticklabels=["C to N", "C to O", "C to F"],
           ylabel="Edited minus original graph Vina (kcal/mol)", title="B. Within-pocket means of strict-success poses\nGraph-specific seeds; available-pose subset")
    ax.legend(fontsize=8)
    fig.suptitle("Vina score-only comparisons | below zero = lower score", fontsize=13)
    fig.savefig(out / "paired_vina.png")
    fig.savefig(out / "paired_vina.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, prop, title in zip(axes, ["qed", "sa", "logp"], ["QED", "SA (lower = easier)", "RDKit LogP"]):
        es = [next(e for e in chemicals if e["cohort"] == "evaluated_30" and e["property"] == prop and e["variant"] == v) for v in VARIANTS[1:]]
        means = np.array([e["mean_delta"] for e in es]);cis = np.array([e["mean_ci"] for e in es])
        ax.bar(range(3), means, color=["#7365a7", "#239a9c", "#ca9b44"], alpha=.85)
        ax.errorbar(range(3), means, yerr=np.maximum(0, np.array([means-cis[:, 0], cis[:, 1]-means])), fmt="none", color="#333", capsize=3)
        ax.axhline(0, color="gray", lw=.8)
        ax.set(xticks=range(3), xticklabels=["C to N", "C to O", "C to F"], ylabel="Mean edited minus original", title=title)
    fig.suptitle("Chemical changes come from graph edits, not FM pose generation | 30 pockets; 95% bootstrap intervals", fontsize=11)
    fig.savefig(out / "chemical_changes.png")
    fig.savefig(out / "chemical_changes.pdf")
    plt.close(fig)


def report(plan, a, metrics, audit, vina, edits, chemistry, natives, out):
    pooled = {x["kind"]: x for x in a["pooled"]}
    table = [{"방법": LABELS[k], "PB 통과": ratio(pooled[k]["pb_valid"], pooled[k]["scheduled"]),
              "종합 성공": ratio(pooled[k]["success"], pooled[k]["scheduled"]),
              "SDF 생성/준비": ratio(pooled[k]["sdf_present"], pooled[k]["scheduled"])} for k in ["native"]+KINDS]
    variant = []
    for v in VARIANTS:
        entry = {"Graph": v}
        for k in KINDS:
            e = next(e for e in metrics if e["variant"] == v and e["kind"] == k)
            entry[LABELS[k]] = ratio(e["success"], e["scheduled"])
        variant.append(entry)
    any_table = []
    for v in VARIANTS:
        item = {"Graph": v}
        for k in ["fm_whole", "fm_core"]:
            e = next(e for e in metrics if e["variant"] == v and e["kind"] == k)
            item[LABELS[k]] = ratio(e["any_success_cases"], a["evaluated_complete_cases"])
        any_table.append(item)
    vina_table = []
    for e in vina:
        ci = e["pocket_median_ci"]
        vina_table.append({"방법": LABELS[e["kind"]], "subset": e["subset"], "pairs/pockets": f"{e['pairs']}/{e['pockets']}",
                           "전체 pair Δ 중앙값": f"{e['median_delta']:+.3f}" if e["median_delta"] is not None else "—",
                           "낮아진 pairs": ratio(e["improved"], e["pairs"]),
                           "pocket 중앙값 [95% CI]": f"{e['pocket_median_delta']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci[0] is not None else "—"})
    edit_table = [{"조건": LABELS[e["kind"]], "편집": e["variant"], "pockets": e["pockets"],
                   "Δ Vina 중앙값": f"{e['median_delta']:+.3f}", "낮아진 pocket": f"{e['improved_pockets']}/{e['pockets']}",
                   "95% CI": f"[{e['median_ci'][0]:+.3f}, {e['median_ci'][1]:+.3f}]"} for e in edits]
    chemicals = []
    for v in VARIANTS[1:]:
        e = {"편집": v}
        for prop in ["qed", "sa", "logp", "hbd", "hba", "tpsa"]:
            x = next(c for c in chemistry if c["cohort"] == "evaluated_30" and c["variant"] == v and c["property"] == prop)
            e[f"평균 Δ{prop}"] = f"{x['mean_delta']:+.3f}"
        chemicals.append(e)
    lines = ["# 편집 graph 조건 FLOWR.root 실험 분석 — 2026-10-18", "",
             "**100-pocket 실험은 완주하지 못했다. 43개 pocket의 생성이 완료됐고, 44번째 5edl에서 입력 표현 오류로 중단됐다. 평가가 완결된 30개 pocket만 아래 품질 비교에 사용한다.**", "",
             f"분석 시각: {a['analysis_kst']} (KST). 원본: `{a['run']}`.", "",
             "## 실행 상태와 중단 원인", "",
             f"- 준비 100/100, 생성 완료 {a['generation_complete_cases']}/100, 완결 평가 {a['evaluated_complete_cases']}/100.",
             "- 생성 완료 43개: 2,752 예정 슬롯 중 SDF 2,574개, 초기화 실패 178개. SDF 중 엄격한 graph/stereo 보존 2,302개, 계약 실패 272개.",
             "- 44번째 5edl: 64슬롯이 `CapabilityError: Tensorization changed the graph or atom order`로 실패. 8개 graph/조건 batch가 연속 실패하자 안전 중단했다. 후속 56개 pocket의 3,584슬롯은 실행되지 않았다. 이 중단은 OOM 기록이 아니라 graph round-trip 거부다.",
             "- 저장된 5edl 입력의 RDKit 표현에는 carbon radical 전자 1개(`[CH]`)가 있다. CPU에서 FLOWR `GeometricMol.from_rdkit → to_rdkit`만 실행해도 `[CH] → C`로 변해 radical 1→0, canonical identity 불일치가 재현됐다. 기존 tensor 표현에 radical/수소 지정 보존 채널이 없다는 설명과 일치한다. 이는 이번 입력 표현의 진단이며 실제 실험 ligand가 radical이라는 주장은 아니다. 실제 로그의 검사 전체를 모델과 함께 재실행한 것은 아니다.",
             "- 무작위 시드와 기존 cohort는 유지했다. 이 입력을 자동 중성화·교체하거나 이후 결과를 채워 넣지 않았다. 새 GPU 실행은 하지 않았다.",
             "- CPU 평가 30개는 2026-10-16 22:23 KST까지 끝났다. 원래 summary는 controller 종료 시각 22:13 KST의 24개/일부 행만 반영했다. 이번에 원본 exit snapshot을 보관하고 summary/metrics를 30개 완결 결과로 갱신했다.",
             f"- 생성은 끝났지만 평가가 시작되지 않은 13개: {', '.join(a['generation_only_case_ids'])}.", "",
             "## 실험이 검증한 것", "",
             "동일 pocket에서 원본 graph와 자유 말단의 같은 C를 N/O/F로 바꾼 세 graph를 미리 확정했다. graph당 graph-only ETKDG 시작 pose 8개를 만들고, 보호 core에 배치했다. 원본 자유 영역 좌표는 사용하지 않았다. 같은 시작 SDF를 전체 좌표 자유(FM whole)와 core XYZ 고정(FM core)에 넣었다. **둘 다 편집 후 전체 graph를 고정**한다. FM이 자유 영역의 graph를 직접 설계한 실험은 아니다. Pocket은 원본 frame의 고정 6 Å protein-only 영역이며, whole도 초기 core 위치 정보를 받으므로 blind docking이 아니다. 원본 initial ligand에서 어느 편집이 더 좋은지 LLM이 고른 것도 아니다.", "",
             "PB는 dock 20-check 설정이다. pocket이 protein-only이므로 water/cofactor 검사는 실제 보조분자가 없는 입력에 대한 결과다. 종합 성공은 PB 전체 통과 + 요청 graph의 입체화학 보존이며, core 조건은 보호 heavy XYZ 최대 이동 ≤0.001 Å도 요구한다. 초기화 실패는 성공률 분모에 남겼다. 새 편집 분자의 실험 pose가 없으므로 native 전체 RMSD를 정확도로 쓰지 않았다.", "",
             "## 물리 타당성과 종합 성공", "", markdown(pd.DataFrame(table)), "",
             "Native를 제외한 각 방법은 같은 30 pockets × 4 graphs × 8 draws = 960슬롯이다. 시작 pose 초기화는 900/960 성공했다. PB 측정 timeout은 없으며 PB 누락 60개/방법은 초기화 SDF 부재다. Native는 30개 기준으로 별도 해석한다. Native PB 실패는 내부 에너지 2개, 내부 입체 충돌 1개이며 단백질 접촉 실패는 없었다.", "",
             f"![Generation quality]({out / 'generation_quality.png'})", "", markdown(pd.DataFrame(variant)), "",
             "8개 후보 중 종합 성공이 한 개 이상 있는 pocket 비율:", "", markdown(pd.DataFrame(any_table)), "",
             "모든 비교는 동일 30개 pocket에서 한다. 8개 중 성공률은 1회 생성 성공률과 구분한다. 여러 후보를 생성하면 성공 후보를 찾을 가능성이 커진다. 위 결과는 최종 100개 성능이나 사전학습과 분리된 test benchmark 성능이 아니다. 중단과 CPU 평가 완료 순서에 의한 부분 cohort이므로 과도한 일반화를 피한다.", "",
             "## 조건을 실제로 지켰는가", "",
             f"전체 저장 SDF {audit['saved_sdf_count']}개를 독립 재검사했다. 모든 SDF는 원자 순서별 원소·전하·동위원소·radical·수소 수·결합 연결/차수를 보존했다. 즉 272개 계약 실패는 비입체 graph가 바뀐 것이 아니라 좌표에서 재추론한 입체화학이 요청과 달랐기 때문이다. 분자 identity 보존에는 이 입체화학도 필요하므로 성공으로 세지 않았다.", "",
             f"- Core: {audit['by_arm']['core']['core_preserved']}/{audit['by_arm']['core']['saved_sdf_count']} 출력의 보호 좌표 보존, 저장 SDF 최대 이동 {audit['by_arm']['core']['max_core_drift']:.6f} Å.",
             f"- Whole: 전체 좌표가 자유라 원본 core 이동은 허용한다. 원본 core 최대 이동 {audit['by_arm']['whole']['max_core_drift']:.3f} Å.",
             f"- 엄격한 stereo 보존: whole {ratio(audit['by_arm']['whole']['strict_stereo_preserved'], audit['by_arm']['whole']['saved_sdf_count'])}, core {ratio(audit['by_arm']['core']['strict_stereo_preserved'], audit['by_arm']['core']['saved_sdf_count'])}.",
             f"- 저장 condition {audit['condition_count']}개에서 graph/좌표 mask와 입력 hash 확인. 전체/부분 조건 사이 시작 SDF는 {audit['same_input_pairs']}/{audit['whole_core_input_pairs']}쌍 동일했다.", "",
             "core 고정 후 물리 성공률이 낮아진 이유는 보호 XYZ 무시보다, 고정 영역과 자유 영역의 경계에서 결합 기하를 동시에 만족시키기 어려운 현상에 가깝다. 평가된 900 SDF/방법에서 core는 결합 길이 208개·각도 179개, whole은 각각 63개·76개가 실패했다. 반면 단백질 최소거리 실패는 시작 pose 450개 → whole 30개/core 36개로 줄었다. 단일 출력이 여러 검사를 실패할 수 있으므로 검사 실패 수는 합산하지 않는다. 경계가 직접 원인인지는 추가 공간별 진단이 필요하다.", "",
             "UFF whole은 ligand-only energy를 줄여도 단백질을 고려하지 않아 protein 최소거리 실패가 638/900개로 늘었다. 이 결과는 단백질 문맥 없이 ligand만 최적화하는 대조군의 한계를 보여준다.", "",
             "## Vina: 같은 시작 pose와 비교", "",
             f"Vina receptor 준비 성공 {a['vina_receptor_success_cases']}/30, 실패 {a['vina_receptor_failed_cases']}/30이다. 잔기 template 불일치·결손 등 로그는 receptor_availability.csv의 파일 경로를 따른다. 실패 사례를 임의 residue 삭제로 통과시키지 않았다. 각 방법의 점수는 567개 입력/출력 쌍에서 비교 가능했다. Vina는 score-only 대리 점수이며 실험 affinity가 아니다. Δ=출력−시작 pose, 음수가 낮은 점수다.", "",
             markdown(pd.DataFrame(vina_table)), "",
             "all_scored는 PB 또는 stereo 실패도 포함한다. both_graph_preserved는 양쪽 graph/stereo가 맞는 쌍, both_strict_success는 양쪽 모두 물리/조건 검사까지 통과한 더 작은 조건부 집합이다. 이 필터 결과는 별도의 전체 성능이 아니다. 한 pocket 안의 8개 pose를 독립 표본으로 가정하지 않도록 pocket 중앙값과 pocket 재표집 5,000회 CI를 함께 기록했다. 현재 부분 cohort의 탐색적 불확실성 범위다.", "",
             "## 화학 편집이 원본 graph보다 나았는가", "",
             "아래는 동일 pocket·FM 조건에서 graph별 종합 성공 pose들의 Vina 평균을 만든 후, 편집 graph 평균−원본 graph 평균을 비교한 것이다. 두 graph에 유효 후보가 있는 pocket만 사용한다. graph 간 초기 seed가 다르고 성공 subset 크기도 다르므로 단일 원자 편집의 인과 효과나 공정한 고정 성공률 비교로 해석하지 않는다.", "", markdown(pd.DataFrame(edit_table)), "",
             f"![Paired Vina]({out / 'paired_vina.png'})", "",
             "원본 실험 ligand와의 score 비교는 generated_original_vs_native_vina.csv에 있다. 원본과 같은 graph의 성공 pose만 사용하며, 비교 가능한 native도 PB 통과한 사례로 제한했다.", ""]
    for kind, g in natives.groupby("kind"):
        lines += [f"- {LABELS[kind]} 원본 graph vs 실험 native: {len(g)} pockets, pocket별 평균 Δ의 중앙값 {g.delta_vina.median():+.3f} kcal/mol, 낮아진 pocket {(g.delta_vina<0).sum()}/{len(g)}."]
    lines += ["", "## 2D 화학 특성", "", markdown(pd.DataFrame(chemicals)), "",
              "평가 완료 30개 pocket에서 graph 편집 직후의 평균 변화다. QED·SA·RDKit LogP 변화는 C→N/O/F 편집으로 결정되며 FM 재생성이 만들어낸 변화가 아니다. 동일 graph/stereo가 보존되면 이들 descriptor는 재생성 전후 동일하다. LogP를 낮추는 방향과 QED/SA를 개선하는 방향은 항상 같지 않다. 목표 범위를 미리 주거나 후보를 점수로 골라 만든 편집이 아니어서 평균 편집 이득을 보장할 수 없다. pKa·용해도·실험 affinity는 이번에 측정하지 않았다. 전체 선정 100개 graph의 특성 변화도 chemical_change_summary.json에 별도로 기록했다.", "",
              f"![Chemical changes]({out / 'chemical_changes.png'})", "",
              "## 해석과 다음 단계", "",
              "1. 이 부분 결과는 **명시적 편집 graph를 유지하면서 pocket 안 좌표를 개선할 수 있음**을 지지한다. 시작 pose 대비 FM의 protein clash와 Vina score가 개선됐다.",
              "2. core 좌표 hard hold는 실제 작동했다. 그러나 stereo와 경계 기하 품질은 별개이며, 전체 자유 pose보다 종합 성공률은 낮았다. hard condition 준수만으로 좋은 생성이 보장되지 않는다.",
              "3. 어떤 화학 편집이 목표를 개선하는지는 아직 증명하지 못했다. N/O/F 모두 임의의 고정 규칙이며, LLM 목표 설정이나 후보 랭킹 효과를 평가한 것이 아니다. 편집 후 물성·Vina의 상충을 함께 봐야 한다.",
              "4. 먼저 입력 적격성에 radical/수소·graph round-trip 사전 검사를 추가하고, 개별 pocket의 입력 실패가 cohort 전체를 중단시키지 않도록 복구 경로를 설계해야 한다. 5edl을 원본대로 보존/명시적 unsupported 처리할지 결정하고, 같은 선정 100개와 seed를 유지해 미완료 부분을 재개하는 것이 다음 우선순위다. 이번 분석에서는 재생성이나 누락 평가를 새로 돌리지 않았다.",
              "5. 100개 완료 후 전체 비교와 pocket별 paired 분석을 다시 해야 한다. core의 실패 원자가 고정/자유 경계에 몰리는지도 점검하고, 결과가 유지되면 그 뒤 목표 기반 graph 후보 선택 실험으로 진행한다.", "",
              "## 파일", "",
              "- analysis.json: 완료/실패 상태와 pooled 통계, paired binary 차이 및 pocket bootstrap CI.",
              "- evaluated_rows.csv, group_metrics.csv: 완결 30개 기준 전체 행과 조건별 통계.",
              "- independent_graph_coordinate_audit.csv, condition_audit.csv: 전체 저장 SDF와 condition 독립 검증.",
              "- paired_vina_*_vs_prior.csv, paired_vina_summary.json: 동일 시작 pose와 paired score 비교.",
              "- edited_vs_original_successful_pose_vina.csv, edited_vina_summary.json: 같은 pocket에서 graph 편집 비교.",
              "- chemical_properties_all_100.csv, chemical_changes.csv: graph별 2D descriptor와 변화.",
              "- pb_failure_breakdown.csv, receptor_availability.csv: 물리 검사 실패 및 receptor 누락.",
              "- controller_exit_snapshot: 중단 당시의 옛 summary/REPORT/metrics. 원본 결과와 오류는 실험 run에 유지."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
