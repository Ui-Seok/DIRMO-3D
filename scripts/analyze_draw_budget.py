"""Preregistered: does probing then committing beat simply spreading draws?

A pocket offers several edit candidates; each costs one FM draw plus one evaluation, and
the pipeline keeps the best FEASIBLE result, so an infeasible draw is worth nothing. Given
a fixed draw budget the question is an allocation one: sample many candidates once
(breadth), or sample one candidate repeatedly after a probe identifies it (depth)?

Hypothesis H: probe-then-commit beats breadth at equal budget.

Prediction recorded BEFORE computing: H fails. The variance decomposition on this dataset
put 5% of edit utility variance within a candidate (across seeds) and 95% between
candidates, so a draw spent on a new candidate buys far more than a repeat of a known one.

Decision rule, fixed in advance: a policy wins only if its median achieved value across
pockets is higher AND the paired difference's 95% bootstrap interval over pockets excludes
zero. Ties are reported as ties.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RANK = re.compile(r"oracle-(\d+)-")


def matrix(rows):
    """pocket -> {candidate rank -> {seed -> (delta_u, feasible)}} for the oracle arm."""
    baselines = {}
    for row in rows:
        baselines.setdefault(row["group_id"], row["observation_before"])
    broken = {t for t, b in baselines.items()
              if not b["feasible"] or (b["metrics"].get("vina") or -99) > 0 or b["utility"] is None}
    out = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if row["group_id"] in broken or row.get("candidate_source") != "oracle":
            continue
        if row["utility_change"] is None:
            continue
        found = RANK.match(row["action"]["candidate"]["candidate_id"])
        if not found:
            continue
        out[row["group_id"]][int(found.group(1))][row["seed"]] = (
            row["utility_change"], bool(row["feasible"]))
    return out, sorted(broken)


def achieved(draws):
    """What the pipeline would keep: the best feasible result, or nothing."""
    usable = [value for value, feasible in draws if feasible]
    return max(usable) if usable else 0.0


def breadth(cell, budget):
    """One draw on each of `budget` candidates, averaged over which seed each one lands on."""
    ranks = sorted(cell)[:budget]
    if len(ranks) < budget:
        return None
    seeds = [sorted(cell[r]) for r in ranks]
    totals = [achieved([cell[r][s] for r, s in zip(ranks, pick)])
              for pick in itertools.product(*seeds)]
    return st.mean(totals)


def depth(cell, budget, rank=1):
    """All draws on one candidate."""
    if rank not in cell or len(cell[rank]) < budget:
        return None
    seeds = sorted(cell[rank])
    return st.mean(achieved([cell[rank][s] for s in pick])
                   for pick in itertools.combinations(seeds, budget))


def probe_then_commit(cell, probes, budget):
    """One draw on each of `probes` candidates, then spend the rest on the best observed."""
    ranks = sorted(cell)[:probes]
    if len(ranks) < probes or budget < probes:
        return None
    remaining = budget - probes
    seeds = [sorted(cell[r]) for r in ranks]
    totals = []
    for pick in itertools.product(*seeds):
        chosen = dict(zip(ranks, pick))
        opened = {rank: cell[rank][seed] for rank, seed in chosen.items()}
        draws = list(opened.values())
        # Commit to the best feasible probe; with none feasible there is nothing to commit to.
        feasible = {rank: value for rank, (value, ok) in opened.items() if ok}
        best = max(feasible, key=feasible.get) if feasible else None
        if best is None or not remaining:
            totals.append(achieved(draws))
            continue
        unused = [seed for seed in sorted(cell[best]) if seed != chosen[best]]
        if len(unused) < remaining:
            return None
        totals.append(st.mean(achieved(draws + [cell[best][seed] for seed in more])
                              for more in itertools.combinations(unused, remaining)))
    return st.mean(totals)


def paired(a, b, seed=0):
    difference = np.asarray([x - y for x, y in zip(a, b)])
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(difference, len(difference), replace=True).mean()) for _ in range(4000)]
    low, high = np.quantile(means, [0.025, 0.975])
    return {"median_difference": float(np.median(difference)), "mean_difference": float(difference.mean()),
            "interval": [float(low), float(high)], "excludes_zero": bool(low > 0 or high < 0)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    cells, broken = matrix(rows)

    report = {"dataset": str(args.dataset.resolve()), "pockets_seen": len(cells),
              "pockets_dropped_broken_baseline": broken,
              "hypothesis": "probe-then-commit beats breadth at equal budget",
              "prediction_before_computing": "H fails; breadth wins, because 95% of edit "
                                             "utility variance is between candidates",
              "decision_rule": "higher median AND paired 95% bootstrap interval excludes zero",
              "budgets": {}}
    for budget in (2, 3):
        policies = {"breadth": lambda c, b=budget: breadth(c, b),
                    "depth_rank1": lambda c, b=budget: depth(c, b),
                    **({"probe2_commit": lambda c, b=budget: probe_then_commit(c, 2, b)}
                       if budget >= 3 else {})}
        scored = {name: {} for name in policies}
        for pocket, cell in cells.items():
            values = {name: fn(cell) for name, fn in policies.items()}
            if any(v is None for v in values.values()):
                continue
            for name, value in values.items():
                scored[name][pocket] = value
        shared = sorted(set.intersection(*(set(v) for v in scored.values()))) if scored else []
        entry = {"pockets_compared": len(shared), "median": {}, "mean": {}, "comparisons": {}}
        for name in policies:
            series = [scored[name][p] for p in shared]
            entry["median"][name] = float(np.median(series)) if series else None
            entry["mean"][name] = float(np.mean(series)) if series else None
        for left, right in itertools.combinations(policies, 2):
            if len(shared) >= 10:
                entry["comparisons"][f"{left} - {right}"] = paired(
                    [scored[left][p] for p in shared], [scored[right][p] for p in shared])
        report["budgets"][str(budget)] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "analyzed", "output": str(args.output),
                      "pockets": report["pockets_seen"],
                      "medians": {b: e["median"] for b, e in report["budgets"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
