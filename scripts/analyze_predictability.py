"""Preregistered predictability test for the DIRMO router.

Question: can a pre-action feature vector predict what an intervention will do, well
enough that a learned router could beat measuring? Two targets per arm -- whether the
action survives evaluation (classification) and how much utility it moves (regression) --
against a trivial baseline and a zero-parameter rule, under grouped cross-validation on
the protein so no target appears in both train and test.

Stop rule, fixed before looking: if every feature family's out-of-fold score is inside
the group-bootstrap 95% interval of the trivial baseline, or AUC <= 0.60 and R2 <= 0.10,
the quantity is declared unpredictable and no router is built on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dirmo.routing.features import FEATURE_NAMES
from dirmo.training.router import group_components

AUC_BAR, R2_BAR = 0.60, 0.10


def auc(y, score):
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # Mid-ranks for ties, so a constant score scores exactly 0.5.
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    positive = ranks[y].sum()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((positive - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def r2(y, pred):
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    denominator = ((y - y.mean()) ** 2).sum()
    return float("nan") if denominator == 0 else float(1 - ((y - pred) ** 2).sum() / denominator)


def out_of_fold(x, y, groups, kind):
    """Pooled out-of-fold predictions under GroupKFold on the protein."""
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    unique = np.unique(groups)
    if len(unique) < 3:
        return None
    models = ({"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)),
               "boosted": GradientBoostingClassifier(random_state=0)} if kind == "classify"
              else {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                    "boosted": GradientBoostingRegressor(random_state=0)})
    predictions = {name: np.full(len(y), np.nan) for name in models}
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    for train, test in splitter.split(x, y, groups=groups):
        if kind == "classify" and len(np.unique(y[train])) < 2:
            continue
        for name, model in models.items():
            model.fit(x[train], y[train])
            predictions[name][test] = (model.predict_proba(x[test])[:, 1] if kind == "classify"
                                       else model.predict(x[test]))
    return predictions


def group_bootstrap(y, pred, groups, kind, repeats=2000, seed=0):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    scores = []
    for _ in range(repeats):
        chosen = rng.choice(unique, len(unique), replace=True)
        index = np.concatenate([np.where(groups == g)[0] for g in chosen])
        value = auc(y[index], pred[index]) if kind == "classify" else r2(y[index], pred[index])
        if np.isfinite(value):
            scores.append(value)
    if len(scores) < repeats // 10:
        return None
    return [float(v) for v in np.quantile(scores, [0.025, 0.975])]


def within_group_auc(y, pred, pockets, usable):
    """Rank accuracy among rows of the SAME pocket.

    Pooled AUC mixes two questions: which pocket is promising, and which action to take
    inside it. A router only ever faces the second, and between-pocket variance dilutes
    it, so this is the metric that decides whether a learned router can choose at all.
    """
    pairs = hits = 0.0
    mixed = 0
    for pocket in set(pockets):
        rows_here = usable & (pockets == pocket)
        positive, negative = pred[rows_here & (y == 1)], pred[rows_here & (y == 0)]
        if len(positive) and len(negative):
            mixed += 1
        for a in positive:
            for b in negative:
                pairs += 1
                hits += 1.0 if a > b else (0.5 if a == b else 0.0)
    return (float(hits / pairs) if pairs else None), int(pairs), mixed


def evaluate(rows, label, target, kind, families, zero_rule, pockets=None):
    y = np.asarray([target(r) for r in rows], dtype=float)
    groups = np.asarray(group_components(rows, "target"))
    full = np.asarray([r["pre_action_features"] for r in rows], dtype=float)
    out = {"arm": label, "target": kind, "rows": len(rows), "groups": int(len(np.unique(groups))),
           "positive_rate" if kind == "classify" else "target_median":
               float(y.mean()) if kind == "classify" else float(np.median(y)), "models": {}}

    trivial = np.full(len(y), y.mean())
    out["models"]["trivial_constant"] = {
        "score": 0.5 if kind == "classify" else r2(y, trivial),
        "interval": group_bootstrap(y, trivial, groups, kind)}
    rule = np.asarray([zero_rule(r) for r in rows], dtype=float)
    out["models"]["zero_parameter_rule"] = {
        "score": auc(y, rule) if kind == "classify" else r2(y, rule),
        "interval": group_bootstrap(y, rule, groups, kind), "definition": "proposed_delta_qed"}

    for family, columns in families.items():
        predictions = out_of_fold(full[:, columns], y, groups, kind)
        if predictions is None:
            out["models"][family] = {"score": None, "reason": "fewer than three groups"}
            continue
        for name, pred in predictions.items():
            usable = np.isfinite(pred)
            if usable.sum() < len(y) // 2:
                out["models"][f"{family}:{name}"] = {"score": None, "reason": "folds unusable"}
                continue
            score = auc(y[usable], pred[usable]) if kind == "classify" else r2(y[usable], pred[usable])
            entry = {"score": score, "features": len(columns),
                     "interval": group_bootstrap(y[usable], pred[usable], groups[usable], kind)}
            if kind == "classify" and pockets is not None:
                entry["within_pocket_auc"], entry["within_pocket_pairs"], entry["mixed_pockets"] = \
                    within_group_auc(y, pred, pockets, usable)
            out["models"][f"{family}:{name}"] = entry

    bar = AUC_BAR if kind == "classify" else R2_BAR
    learned = {k: v["score"] for k, v in out["models"].items()
               if v.get("score") is not None and k not in ("trivial_constant", "zero_parameter_rule")}
    best = max(learned.values()) if learned else float("nan")
    trivial_hi = (out["models"]["trivial_constant"]["interval"] or [0, bar])[1]
    # One model clearing the bar while another sits below chance on the same data is
    # instability, not signal, so require the model families to agree on the sign.
    full = [v for k, v in learned.items() if k.startswith("all_93")]
    agree = bool(full) and all(v > 0.5 if kind == "classify" else v > 0 for v in full)
    within = [v.get("within_pocket_auc") for v in out["models"].values()
              if v.get("within_pocket_auc") is not None]
    pairs = max((v.get("within_pocket_pairs", 0) for v in out["models"].values()), default=0)
    mixed = max((v.get("mixed_pockets", 0) for v in out["models"].values()), default=0)
    out["verdict"] = {
        "best_learned_score": best, "bar": bar, "trivial_upper_95": trivial_hi,
        "zero_parameter_score": out["models"]["zero_parameter_rule"]["score"],
        "models_agree_on_sign": agree,
        "best_within_pocket_auc": max(within) if within else None,
        "within_pocket_pairs": pairs, "mixed_pockets": mixed,
        "rule_ties_learned": bool(np.isfinite(best)
                                 and out["models"]["zero_parameter_rule"]["score"] is not None
                                 and out["models"]["zero_parameter_rule"]["score"] >= best - 0.02)}
    if kind == "classify":
        # A verdict needs enough pockets where outcomes actually differ; otherwise the
        # pooled number is decided by a handful of negatives.
        enough = mixed >= 10 and pairs >= 30
        out["verdict"]["predictable"] = bool(
            enough and agree and within and max(within) > bar)
        out["verdict"]["blocked_by"] = (None if enough else
                                        f"only {mixed} pockets with mixed outcomes ({pairs} pairs); "
                                        f"need 10 pockets and 30 pairs")
    else:
        out["verdict"]["predictable"] = bool(
            np.isfinite(best) and best > bar and best > trivial_hi and agree)
        out["verdict"]["blocked_by"] = None if agree else "model families disagree on sign"
    return out


def cells(rows):
    """Group draws by (pocket, resulting graph, kernel, arm) across seed passes.

    A cell is one intervention repeated under different seeds. Cell-level targets are
    what a router actually needs: whether redrawing this graph can succeed at all, and
    how widely its outcome varies -- the quantity that decides if resampling is worth a
    draw. Pre-action features are identical across a cell's seeds by construction
    (collect computes them from one shared snapshot), so any member's vector serves.
    """
    from collections import defaultdict
    grouped = defaultdict(dict)
    for r in rows:
        key = (r["group_id"], r["resulting_work_item"]["graph_id"], r["kernel_id"],
               "edit" if r["candidate_source"] else "pose")
        grouped[key][r["seed"]] = r
    out = []
    for (pocket, _, _, arm), by_seed in grouped.items():
        draws = list(by_seed.values())
        measured = [d["utility_change"] for d in draws if d["utility_change"] is not None]
        out.append({"pocket": pocket, "arm": arm, "seeds": len(draws),
                    "feasible_rate": float(np.mean([d["feasible"] for d in draws])),
                    "any_feasible": float(any(d["feasible"] for d in draws)),
                    "spread": (max(measured) - min(measured)) if len(measured) >= 2 else None,
                    "best": max(measured) if measured else None,
                    "group_id": pocket, "target_id": draws[0]["target_id"],
                    "pre_state_id": draws[0]["pre_state_id"],
                    "feature_names": draws[0]["feature_names"],
                    "pre_action_features": draws[0]["pre_action_features"]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = [json.loads(l) for l in args.dataset.read_text().splitlines() if l.strip()]
    if any(r["feature_names"] != FEATURE_NAMES for r in rows):
        print(json.dumps({"status": "blocked", "reason": "feature schema mismatch"}), file=sys.stderr)
        return 2

    # One objective, and drop pockets whose supplied pose is already unusable: their
    # baseline utility is so low that every action looks like a large gain.
    baselines = {}
    for r in rows:
        baselines.setdefault(r["group_id"], r["observation_before"])
    broken = {t for t, b in baselines.items()
              if not b["feasible"] or (b["metrics"].get("vina") or -99) > 0 or b["utility"] is None}
    clean = [r for r in rows if r["group_id"] not in broken]

    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    families = {
        "all_93": list(range(len(FEATURE_NAMES))),
        "proposed_delta_only": [i for n, i in index.items() if n.startswith("proposed_delta_")],
        "state_metrics_only": [i for n, i in index.items() if n.startswith("state_")],
        "uncertainty_and_history": [index[n] for n in
                                    ("round", "plateau", "attempts", "observed_feasible",
                                     "wilson_width", "prior_failures")],
    }
    delta_qed = index["proposed_delta_qed"]

    report = {"dataset": str(args.dataset.resolve()), "rows_total": len(rows),
              "objectives": sorted({r["objective_sha256"] for r in rows}),
              "generation_kernels": sorted({r["generation_kernel_sha256"] for r in rows}),
              "targets_total": len(baselines), "targets_dropped_broken_baseline": sorted(broken),
              "rows_after_filter": len(clean), "stop_rule": {"auc_bar": AUC_BAR, "r2_bar": R2_BAR},
              "questions": []}

    arms = {"edit_llm_and_oracle": lambda r: r["candidate_source"] in ("llm", "oracle"),
            "edit_oracle_only": lambda r: r["candidate_source"] == "oracle",
            "pose_and_steps": lambda r: r["candidate_source"] is None}
    for label, keep in arms.items():
        subset = [r for r in clean if keep(r)]
        executed = [r for r in subset if r["generation_result"]["status"] == "ok"]
        if len(executed) >= 20:
            report["questions"].append(evaluate(
                executed, label, lambda r: r["feasible"], "classify", families,
                lambda r: r["pre_action_features"][delta_qed],
                pockets=np.asarray([r["group_id"] for r in executed])))
        measured = [r for r in subset if r["utility_change"] is not None]
        if len(measured) >= 20:
            report["questions"].append(evaluate(
                measured, label, lambda r: r["utility_change"], "regress", families,
                lambda r: r["pre_action_features"][delta_qed]))

    # Q1: cell-level questions, only answerable once a graph has several seeds.
    report["cell_level"] = []
    all_cells = cells([r for r in clean if r["generation_result"]["status"] == "ok"])
    report["cells_total"] = len(all_cells)
    report["cells_with_three_seeds"] = sum(1 for c in all_cells if c["seeds"] >= 3)
    for arm in ("pose", "edit"):
        repeated = [c for c in all_cells if c["arm"] == arm and c["seeds"] >= 2]
        if len(repeated) >= 20:
            report["cell_level"].append(evaluate(
                repeated, f"{arm}_cells", lambda c: c["any_feasible"], "classify", families,
                lambda c: c["pre_action_features"][delta_qed],
                pockets=np.asarray([c["pocket"] for c in repeated])))
        spread = [c for c in repeated if c["spread"] is not None]
        if len(spread) >= 20:
            report["cell_level"].append(evaluate(
                spread, f"{arm}_cells_spread", lambda c: c["spread"], "regress", families,
                lambda c: c["pre_action_features"][delta_qed]))
            report["cell_level"][-1]["spread_summary"] = {
                "median": float(np.median([c["spread"] for c in spread])),
                "p90": float(np.quantile([c["spread"] for c in spread], 0.9)),
                "max": float(max(c["spread"] for c in spread)), "cells": len(spread)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "analyzed", "output": str(args.output),
                      "rows": report["rows_after_filter"], "cells": report["cells_total"],
                      "cells_with_three_seeds": report["cells_with_three_seeds"],
                      "verdicts": {f"{q['arm']}:{q['target']}": q["verdict"]["predictable"]
                                   for q in report["questions"] + report["cell_level"]}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
