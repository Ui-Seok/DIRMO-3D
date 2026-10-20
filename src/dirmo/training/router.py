from __future__ import annotations

from collections import Counter

import numpy as np

from ..io import digest, read_jsonl, write_json
from ..routing.features import FEATURE_NAMES


def group_components(rows: list[dict], unit: str = "target") -> list[str]:
    """Union all rows sharing target/series OR state; a state can never cross folds."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        group = row["target_id"] if unit == "target" else row.get("series_id")
        if not group:
            raise ValueError(f"Missing grouping identifier for unit={unit}")
        for key in ((unit, group), ("state", row["pre_state_id"])):
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return [str(find(i)) for i in range(len(rows))]


def grouped_folds(rows, unit="target", folds=3, seed=42):
    groups = np.array(group_components(rows, unit))
    unique = np.unique(groups)
    if len(unique) < 2:
        return [], {"status": "insufficient_groups", "groups": len(unique), "rows": len(rows),
                    "reason": "Cannot estimate held-out generalization from a single target/series"}
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=min(folds, len(unique)))
    split = list(splitter.split(np.zeros(len(rows)), groups=groups))
    for train, test in split:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Grouped split leakage")
    return split, {"status": "grouped_cv", "groups": len(unique), "folds": len(split), "unit": unit}


def cluster_bootstrap(values, clusters, repeats=500, seed=42):
    values, clusters = np.asarray(values), np.asarray(clusters)
    unique = np.unique(clusters)
    if len(unique) < 2:
        return {"interval": None, "reason": "At least two independent clusters are required"}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        chosen = rng.choice(unique, len(unique), replace=True)
        means.append(np.mean(np.concatenate([values[clusters == c] for c in chosen])))
    return {"interval": np.quantile(means, [0.025, 0.975]).tolist(), "method": "cluster bootstrap", "repeats": repeats}


def fit_model(x, y, train_indices, epochs, seed, device="cpu"):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    train_x, train_y = x[train_indices], y[train_indices]
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale < 1e-6] = 1
    label_mean, label_scale = train_y.mean(0), train_y.std(0)
    label_scale[label_scale < 1e-6] = 1
    model = nn.Sequential(nn.Linear(x.shape[1], 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    inputs = torch.tensor((train_x - mean) / scale, dtype=torch.float32, device=device)
    labels = torch.tensor((train_y - label_mean) / label_scale, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = nn.functional.smooth_l1_loss(model(inputs), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    layers = [{"weight": layer.weight.detach().cpu().numpy().tolist(), "bias": layer.bias.detach().cpu().numpy().tolist()}
              for layer in model if isinstance(layer, nn.Linear)]
    return model, {"mean": mean.tolist(), "scale": scale.tolist(), "label_mean": label_mean.tolist(),
                   "label_scale": label_scale.tolist(), "layers": layers}, float(loss.item())


def train_router(dataset: str, output: str, kernel_id: str | None = None, unit: str = "target",
                 epochs: int = 100, seed: int = 42, device: str = "cpu") -> dict:
    if not 1 <= epochs <= 1000:
        raise ValueError("Epochs must be bounded to 1..1000")
    raw = read_jsonl(dataset)
    kernels = sorted({row["kernel_id"] for row in raw if row.get("kernel_id")})
    if kernel_id is None:
        if len(kernels) != 1:
            raise ValueError("Select --kernel-id: multiple or zero kernels must not share reward labels")
        kernel_id = kernels[0]
    rows = [row for row in raw if row.get("kernel_id") == kernel_id and row.get("utility_change") is not None]
    if len(rows) < 2:
        raise ValueError("Need at least two genuinely executed, measured action outcomes in this kernel")
    if any(row["feature_names"] != FEATURE_NAMES for row in rows):
        raise ValueError("Dataset feature schema mismatch")
    # utility_change is only defined relative to one objective. Rows measured under a
    # different objective are a different experiment, exactly as for the kernel.
    objectives = {row.get("objective_sha256") for row in rows}
    if None in objectives:
        raise ValueError("Dataset rows lack objective_sha256; recollect so the reward "
                         "labels carry the objective they were measured under")
    if len(objectives) != 1:
        raise ValueError(f"Rows span {len(objectives)} objectives; utility changes are not "
                         f"comparable across them: {sorted(objectives)}")
    x = np.asarray([row["pre_action_features"] for row in rows], dtype=np.float64)
    y = np.asarray([[row["utility_change"], row["compute_cost"]["wall_seconds"]] for row in rows], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("Non-finite or malformed training data")
    splits, split_report = grouped_folds(rows, unit=unit, seed=seed)
    fold_results, errors, error_clusters = [], [], []
    import torch
    for fold, (train, test) in enumerate(splits):
        model, scaling, _ = fit_model(x, y, train, epochs, seed + fold, device)
        with torch.inference_mode():
            pred = model(torch.tensor((x[test] - scaling["mean"]) / scaling["scale"],
                                     dtype=torch.float32, device=device)).cpu().numpy()
        pred = pred * scaling["label_scale"] + scaling["label_mean"]
        absolute = np.abs(pred - y[test])
        fold_results.append({"fold": fold, "train_rows": len(train), "test_rows": len(test),
                             "mae_improvement": float(absolute[:, 0].mean()), "mae_cost_seconds": float(absolute[:, 1].mean()),
                             "train_state_ids": sorted({rows[i]["pre_state_id"] for i in train}),
                             "test_state_ids": sorted({rows[i]["pre_state_id"] for i in test})})
        errors.extend(absolute[:, 0].tolist())
        groups = group_components(rows, unit)
        error_clusters.extend(groups[i] for i in test)
    _, artifact, train_loss = fit_model(x, y, np.arange(len(rows)), epochs, seed, device)
    artifact.update(feature_names=FEATURE_NAMES, kernel_id=kernel_id, training_rows=len(rows),
                    objective_sha256=objectives.pop(),
                    integration_steps=rows[0]["integration_steps"], dataset_hash=digest(raw),
                    prediction_targets=["utility_change", "wall_seconds"], seed=seed,
                    supported_action_kinds=sorted({row["action"]["kind"] for row in rows}))
    write_json(output, artifact)
    report = {"status": "trained", "device": device, "router": output, "rows": len(rows),
              "objective_sha256": artifact["objective_sha256"],
              "excluded_unmeasured_or_other_kernel": len(raw) - len(rows), "kernel_id": kernel_id,
              "observed_feasible_rows": sum(row.get("feasible") is True for row in rows),
              "coverage": dict(Counter(row["action"]["kind"] for row in rows)),
              "distinct_states": len({row["pre_state_id"] for row in rows}), "split": split_report,
              "folds": fold_results, "bootstrap_mae": cluster_bootstrap(errors, error_clusters) if errors else None,
              "training_loss": train_loss,
              "limitations": ["Only measured utility differences are regression labels; infeasibility is stored separately.",
                              "Predictions are metric utility and cost, not success probabilities or causal labels.",
                              "Only covered actions in this generation kernel are supported.",
                              "Single-target training is a CLI smoke check; no generalization claim."]}
    write_json(output + ".report.json", report)
    return report
