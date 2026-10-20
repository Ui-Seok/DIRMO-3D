"""Analyze saved observations only; no LLM/FLOWR weights or live workers are used.

Run with the dirmo environment's Python. The input feature names are taken from
the archived dataset, so the old 93-feature experiments remain reproducible after
the production schema changed to 95 features. Writes only beside this script.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
CASE_CACHE = {}


def records(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def case_metadata(directory):
    directory = Path(directory)
    if directory not in CASE_CACHE:
        manifest = json.loads((directory / "manifest.json").read_text())
        config = json.loads((directory / "resolved_config.json").read_text())
        CASE_CACHE[directory] = (manifest["case"], config)
    return CASE_CACHE[directory]


def compact(row):
    after = row["resulting_work_item"].get("observation") or {}
    before = row["observation_before"]
    generation = row["generation_result"]
    case_dir = Path(row["snapshot_path"]).parent
    case, config = case_metadata(case_dir)
    action = row["action"]
    source = row.get("candidate_source")
    if source is None and action["kind"] == "CHEMICAL_EDIT":
        source = "oracle" if action["candidate"]["candidate_id"].startswith("oracle-") else "llm"
    delta = row.get("utility_change")
    minimum = case["objective"]["minimum_improvement"]
    feasible = row.get("feasible") is True
    target_met = after.get("target_met") is True
    return {
        "target": row["target_id"], "group": row.get("group_id", row["target_id"]),
        "seed": row["seed"], "kind": action["kind"], "steps": row["integration_steps"],
        "source": source, "candidate": (action.get("candidate") or {}).get("candidate_id"),
        "product_graph": row["resulting_work_item"]["graph_id"],
        "before_graph": before["graph_id"], "kernel": row.get("kernel_id"),
        "generation_kernel": row.get("generation_kernel_sha256"),
        "gen_status": generation["status"], "gen_reason": generation.get("reason"),
        "raw_draws": generation["raw_draws"], "nfe": generation["nfe"],
        "fm_seconds": generation["latency_seconds"],
        "cost_unknown": "cost unknown" in (generation.get("reason") or ""),
        "feasible": feasible, "target_met": target_met, "delta": delta,
        "minimum_improvement": minimum,
        "improved": feasible and delta is not None and delta >= minimum,
        "eligible": feasible and target_met and delta is not None and delta >= minimum,
        "before_feasible": before["feasible"], "before_target_met": before["target_met"],
        "before_vina": before["metrics"].get("vina"), "before_utility": before["utility"],
        "failures": after.get("failures", []),
        "metric_delta": {name: after.get("metrics", {}).get(name, 0) - before["metrics"].get(name, 0)
                         for name in ("qed", "sa", "vina") if name in after.get("metrics", {})
                         and name in before["metrics"]},
        "feature_names": row["feature_names"], "x": row["pre_action_features"],
        "pre_state_id": row["pre_state_id"], "case_dir": str(case_dir),
        "sdf": row["resulting_work_item"]["sdf_path"],
        "prior": action["generation"]["prior"],
        "noise_std": action["generation"].get("prior_noise_std", 0),
        "region": action.get("region", []),
        "checkpoint": Path(config["fm"]["checkpoint_path"]).name,
        "mode": config["fm"].get("inpaint_mode", "graph"),
        "objective": case["objective"],
    }


def stats(rows):
    values = [r["delta"] for r in rows if r["delta"] is not None]
    return {
        "records": len(rows), "targets": len({r["target"] for r in rows}),
        "generation_status": dict(Counter(r["gen_status"] for r in rows)),
        "measured": len(values), "feasible": sum(r["feasible"] for r in rows),
        "improved_feasible": sum(r["improved"] for r in rows),
        "success_eligible": sum(r["eligible"] for r in rows),
        "targets_with_improvement": len({r["target"] for r in rows if r["improved"]}),
        "targets_with_eligible_candidate": len({r["target"] for r in rows if r["eligible"]}),
        "delta_median_measured": float(np.median(values)) if values else None,
        "delta_median_feasible": (float(np.median([r["delta"] for r in rows if r["feasible"]
                                                 and r["delta"] is not None]))
                                  if any(r["feasible"] and r["delta"] is not None for r in rows) else None),
        "known_raw_draws": sum(r["raw_draws"] for r in rows if not r["cost_unknown"]),
        "unknown_cost_records": sum(r["cost_unknown"] for r in rows),
        "failures": dict(Counter(f for r in rows for f in r["failures"])),
    }


def bootstrap_mean(values, repeats=4000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(0)
    means = rng.choice(values, (repeats, len(values)), replace=True).mean(1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def clean(rows):
    broken = {r["target"] for r in rows if not r["before_feasible"]
              or r["before_utility"] is None
              or (r["before_vina"] is not None and r["before_vina"] > 0)}
    return [r for r in rows if r["target"] not in broken], sorted(broken)


def goal_transitions(rows):
    before = {r["target"]: r["before_target_met"] for r in rows}
    qualified = {r["target"] for r in rows if r["eligible"]}
    newly_reached = sorted(target for target in qualified if not before[target])
    return {"targets": len(before), "initial_goal_met_targets": sum(before.values()),
            "initial_goal_unmet_targets": sum(not value for value in before.values()),
            "qualified_improved_targets": len(qualified),
            "initially_unmet_targets_reaching_goals": newly_reached}


def step_comparison(rows):
    paired = defaultdict(dict)
    for r in rows:
        if r["source"] is None:
            paired[(r["target"], r["before_graph"], r["seed"])][r["steps"]] = r
    complete = [pair for pair in paired.values() if 64 in pair and 128 in pair]
    by_target = defaultdict(list)
    measured = []
    for pair in complete:
        by_target[pair[64]["target"]].append(float(pair[128]["feasible"]) - float(pair[64]["feasible"]))
        if pair[64]["delta"] is not None and pair[128]["delta"] is not None:
            measured.append(pair[128]["delta"] - pair[64]["delta"])
    return {
        "paired_records": len(complete), "targets": len(by_target),
        "feasible_64": sum(p[64]["feasible"] for p in complete),
        "feasible_128": sum(p[128]["feasible"] for p in complete),
        "feasibility_difference_128_minus_64": float(np.mean([
            np.mean(v) for v in by_target.values()])),
        "target_bootstrap_interval": bootstrap_mean([np.mean(v) for v in by_target.values()]),
        "both_measured_pairs": len(measured),
        "mean_delta_difference_both_measured": float(np.mean(measured)),
        "median_delta_difference_both_measured": float(np.median(measured)),
        "median_fm_seconds_by_steps": {
            str(step): float(np.median([p[step]["fm_seconds"] for p in complete
                                       if p[step]["gen_status"] == "ok"])) for step in (64, 128)},
    }


def r2(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    denominator = np.square(y - y.mean()).sum()
    return float(1 - np.square(y - prediction).sum() / denominator) if denominator else None


def matched_probe(rows):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    grouped = defaultdict(dict)
    for r in rows:
        if r["gen_status"] == "ok" and r["delta"] is not None:
            arm = "edit" if r["source"] else "pose"
            grouped[(r["target"], r["product_graph"], r["kernel"], arm)][r["seed"]] = r
    output = {}
    for arm in ("pose", "edit"):
        cells = [by_seed for key, by_seed in grouped.items() if key[-1] == arm and len(by_seed) == 3]
        draws = [[cell[seed] for seed in sorted(cell)] for cell in cells]
        x = np.asarray([d[0]["x"] for d in draws])
        y = np.asarray([np.mean([d[1]["delta"], d[2]["delta"]]) for d in draws])
        probe = np.asarray([d[0]["delta"] for d in draws])
        groups = np.asarray([d[0]["target"] for d in draws])
        models = {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                  "boosted": GradientBoostingRegressor(random_state=0)}
        predictions = {name: np.empty(len(y)) for name in models}
        calibrated_probe = np.empty(len(y))
        for train, test in GroupKFold(n_splits=5).split(x, y, groups):
            for name, model in models.items():
                model.fit(x[train], y[train])
                predictions[name][test] = model.predict(x[test])
            calibrator = LinearRegression().fit(probe[train, None], y[train])
            calibrated_probe[test] = calibrator.predict(probe[test, None])
        output[arm] = {
            "cells": len(cells), "targets": len(set(groups)), "feature_count": x.shape[1],
            "first_draw_predicts_next_two_r2": r2(y, probe),
            "first_draw_pearson_r_squared": float(np.corrcoef(y, probe)[0, 1] ** 2),
            "grouped_cv_calibrated_first_draw_r2": r2(y, calibrated_probe),
            "grouped_cv_pre_action_r2": {name: r2(y, p) for name, p in predictions.items()},
            "protocol": "Seed 42 predicts mean delta-U of seeds 43 and 44; models use the same cells "
                        "and outcome, with 5-fold target-grouped CV. Conditional on all three draws "
                        "having measured utility. First-draw features represent each cell.",
        }
        output[arm]["probe_seed_sensitivity"] = {
            str(draws[0][role]["seed"]): r2(
                [np.mean([d[j]["delta"] for j in range(3) if j != role]) for d in draws],
                [d[role]["delta"] for d in draws]) for role in range(3)}
        # Average over the choice of probe by including each seed once as the probe.
        # These are dependent roles of the SAME cell, kept together in target CV.
        pooled_x = np.repeat(x, 3, axis=0)
        pooled_groups = np.repeat(groups, 3)
        pooled_y = np.asarray([np.mean([d[j]["delta"] for j in range(3) if j != role])
                               for d in draws for role in range(3)])
        pooled_probe = np.asarray([d[role]["delta"] for d in draws for role in range(3)])
        pooled_predictions = {name: np.empty(len(pooled_y)) for name in models}
        pooled_calibrated = np.empty(len(pooled_y))
        for train, test in GroupKFold(n_splits=5).split(pooled_x, pooled_y, pooled_groups):
            for name, model in models.items():
                model.fit(pooled_x[train], pooled_y[train])
                pooled_predictions[name][test] = model.predict(pooled_x[test])
            calibrator = LinearRegression().fit(pooled_probe[train, None], pooled_y[train])
            pooled_calibrated[test] = calibrator.predict(pooled_probe[test, None])
        output[arm]["all_probe_roles_pooled"] = {
            "probe_r2": r2(pooled_y, pooled_probe),
            "grouped_cv_calibrated_probe_r2": r2(pooled_y, pooled_calibrated),
            "pre_action_r2": {name: r2(pooled_y, p) for name, p in pooled_predictions.items()},
            "note": "All three choices of probe per cell are pooled, not three independent cells. "
                    "Each cell repeats its seed-42 feature vector in target-grouped CV."}
    return output


def allocation(rows, measured_only):
    """Replay equal upper draw budgets on the same measured candidate pool.

    The measured-only variant matches the legacy report's inclusion rule. The
    all-draws variant also includes graph-mismatch outcomes as zero retained gain,
    but excludes raw_draws=0 capability rejections because those are not FM draws.
    This is an offline replay of initial-snapshot interventions, not a closed-loop
    policy evaluation.
    """
    matrix = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["source"] != "oracle":
            continue
        if measured_only and r["delta"] is None:
            continue
        if not measured_only and (r["raw_draws"] != 1 or r["cost_unknown"]):
            continue
        rank = int(r["candidate"].split("-")[1])
        matrix[r["target"]][rank][r["seed"]] = r

    def achieved(draws):
        return max([0.0] + [r["delta"] for r in draws if r["feasible"] and r["delta"] is not None])

    def policy_values(cell, budget):
        ranks = sorted(cell)[:budget]
        if len(ranks) < budget or 1 not in cell or len(cell[1]) < budget:
            return None
        breadth = np.mean([achieved([cell[rank][seed] for rank, seed in zip(ranks, pick)])
                           for pick in itertools.product(*(sorted(cell[rank]) for rank in ranks))])
        depth = np.mean([achieved([cell[1][seed] for seed in pick])
                         for pick in itertools.combinations(sorted(cell[1]), budget)])
        result = {"breadth": float(breadth), "depth_rank1": float(depth)}
        if budget == 3:
            values = []
            for pick in itertools.product(*(sorted(cell[rank]) for rank in ranks[:2])):
                opened = {rank: cell[rank][seed] for rank, seed in zip(ranks[:2], pick)}
                feasible = {rank: r["delta"] for rank, r in opened.items()
                            if r["feasible"] and r["delta"] is not None}
                if not feasible:
                    values.append(achieved(list(opened.values())))
                    continue
                best = max(feasible, key=feasible.get)
                unused = [seed for seed in sorted(cell[best]) if seed != opened[best]["seed"]]
                if not unused:
                    return None
                values.append(np.mean([achieved(list(opened.values()) + [cell[best][seed]])
                                       for seed in unused]))
            result["probe2_commit"] = float(np.mean(values))
        return result

    results = {}
    for budget in (2, 3):
        scored = {target: values for target, cell in matrix.items()
                  if (values := policy_values(cell, budget)) is not None}
        names = ("breadth", "depth_rank1") if budget == 2 else ("breadth", "depth_rank1", "probe2_commit")
        entry = {"pockets": len(scored), "mean": {}, "median": {}, "paired": {}}
        for name in names:
            values = [v[name] for v in scored.values()]
            entry["mean"][name], entry["median"][name] = float(np.mean(values)), float(np.median(values))
        for a, b in itertools.combinations(names, 2):
            differences = [v[a] - v[b] for v in scored.values()]
            entry["paired"][a + " - " + b] = {"mean": float(np.mean(differences)),
                                             "interval": bootstrap_mean(differences)}
        results[str(budget)] = entry
    return results


def reference_repeats(rows):
    from rdkit import Chem
    output = []
    for step in (64, 128):
        subset = sorted([r for r in rows if r["steps"] == step and r["source"] is None],
                        key=lambda r: r["seed"])
        if len(subset) < 2:
            continue
        coords = []
        for row in subset[:2]:
            mol = Chem.SDMolSupplier(row["sdf"], removeHs=False, sanitize=False)[0]
            indices = sorted(range(mol.GetNumAtoms()), key=lambda i: mol.GetAtomWithIdx(i).GetAtomMapNum())
            coords.append(mol.GetConformer().GetPositions()[indices])
        output.append({"steps": step, "seeds": [r["seed"] for r in subset[:2]],
                       "pocket_frame_rmsd_angstrom": float(np.sqrt(np.square(coords[0] - coords[1]).sum(1).mean())),
                       "same_utility": subset[0]["delta"] == subset[1]["delta"],
                       "alignment": "None; atom-map correspondence in the fixed pocket frame"})
    return output


def ledger(root):
    files = list(root.rglob("costs.json"))
    resources = Counter()
    elapsed = 0.0
    for path in files:
        data = json.loads(path.read_text())
        resources.update(data["used"])
        elapsed += data["wall_seconds"]
    attempts = valid_cases = cached_cases = 0
    token_total = llm_seconds = 0.0
    valid_candidates = []
    for path in root.rglob("llm.jsonl"):
        events = list(records(path))
        direct = [event for event in events if not event.get("cache_hit")]
        attempts += len(direct)
        token_total += sum(event.get("total_tokens", 0) or 0 for event in direct)
        llm_seconds += sum(event.get("latency_seconds", 0) or 0 for event in direct)
        valid = max([event.get("chemical_valid_count", 0) or 0 for event in events] + [0])
        valid_cases += valid > 0
        cached_cases += any(event.get("cache_hit") for event in events)
        valid_candidates.append(valid)
    return {"case_runs": len(files), "resources": dict(resources), "summed_case_seconds": elapsed,
            "recorded_llm_seconds": llm_seconds, "recorded_llm_attempts": attempts,
            "reported_llm_tokens": token_total, "cases_with_recorded_valid_candidate": valid_cases,
            "cases_with_cache_hit": cached_cases,
            "valid_candidate_count_distribution": dict(Counter(valid_candidates)),
            "note": "Sums saved cost ledgers/LLM events; case seconds include orchestration and CPU work. "
                    "Some older LLM event versions did not record chemical_valid_count."}


def main():
    datasets, output = {}, {"datasets": {}, "inputs": {},
                           "scope": "Saved observations; CPU statistical fitting only; no LLM/FLOWR inference."}
    for path in sorted((PROJECT / "runs").glob("*/branches.jsonl")):
        rows = [compact(row) for row in records(path)]
        name = path.parent.name
        datasets[name] = rows
        by_arm = defaultdict(list)
        for row in rows:
            by_arm[f"{row['kind']}:{row['source'] or '-'}:{row['steps']}"].append(row)
        output["inputs"][str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        output["datasets"][name] = {
            "total": stats(rows), "arms": {key: stats(value) for key, value in by_arm.items()},
            "conditions": sorted({f"{r['checkpoint']}|{r['mode']}|{r['prior']}|noise={r['noise_std']}"
                                  for r in rows}),
        }
    sweep, broken = clean(datasets["sweep"])
    output["sweep"] = {"excluded_baseline_targets": broken, "clean_total": stats(sweep),
                       "steps": step_comparison(sweep), "matched_probe": matched_probe(sweep),
                       "allocation_measured_only_corrected": allocation(sweep, True),
                       "allocation_all_actual_draws_corrected": allocation(sweep, False)}
    output["reference_without_noise"] = reference_repeats(datasets["gpu_05"])
    output["reference_with_noise"] = reference_repeats(datasets["gpu_06"])
    region_drift = []
    for path in (PROJECT / "runs/regions").rglob("redesign.json"):
        report = json.loads(path.read_text())
        if report.get("held_max_drift_angstrom") is not None:
            region_drift.append(report["held_max_drift_angstrom"])
    output["region_held_max_drift"] = {
        "records": len(region_drift), "median": float(np.median(region_drift)),
        "p90": float(np.quantile(region_drift, 0.9)), "max": max(region_drift)}
    output["region_clean"] = stats(clean(datasets["regions"])[0])
    output["goal_transitions"] = {name: goal_transitions(datasets[name]) for name in ("sweep", "regions")}
    output["goal_transitions_clean"] = {"sweep": goal_transitions(sweep),
                                        "regions": goal_transitions(clean(datasets["regions"])[0])}
    output["costs"] = {name: ledger(PROJECT / "runs" / name) for name in ("sweep", "regions")}
    output["clean_sweep_arms"] = {}
    for kind, source, steps in sorted({(r["kind"], r["source"] or "-", r["steps"]) for r in sweep}):
        subset = [r for r in sweep if (r["kind"], r["source"] or "-", r["steps"]) == (kind, source, steps)]
        output["clean_sweep_arms"][f"{kind}:{source}:{steps}"] = stats(subset)
    output["single_target_results"] = []
    for path in sorted((PROJECT / "runs").rglob("result.json")):
        result = json.loads(path.read_text())
        state = json.loads((path.parent / "checkpoint.json").read_text())["state"]
        selected = result["selected"]["observation"]
        baseline = state["original_ligand"]["observation"]
        output["single_target_results"].append({
            "path": str(path.relative_to(PROJECT)), "status": result["status"],
            "selected_original": result["selected"]["item_id"] == "original",
            "delta": selected["utility"] - baseline["utility"],
            "cost": result["cost"], "routing": result.get("routing")})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: output[key] for key in ("sweep", "reference_without_noise",
                                                   "reference_with_noise", "region_held_max_drift")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
