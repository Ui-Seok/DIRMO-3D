"""Audit exported OOF scorers and summarize the GPU Router development study."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from train_router_ablation import predict_artifact


def read(path):
    return json.loads(path.read_text())


def cluster_interval(first, second, field):
    """Resample independent groups, preserving the state-weighted estimand."""
    other = {r['state_id']: r for r in second}
    groups = defaultdict(list)
    for row in first:
        value = other[row['state_id']][field]
        if row[field] is not None and value is not None:
            groups[row['group_id']].append(float(row[field]) - float(value))
    sums = np.array([sum(v) for _, v in sorted(groups.items())])
    counts = np.array([len(v) for _, v in sorted(groups.items())])
    rng = np.random.default_rng(20261026)
    draws = rng.integers(len(sums), size=(5000, len(sums)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {'state_mean_difference': float(sums.sum() / counts.sum()),
            'ci95': np.quantile(samples, [.025, .975]).tolist(),
            'states': int(counts.sum()), 'groups': len(sums), 'resamples': 5000,
            'note': 'Exploratory grouped bootstrap; model fitting is not repeated.'}


def analyze(run, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = read(run / 'prepared/rows.json')
    audit = read(run / 'prepared/audit.json')
    summary = read(run / 'summary.json')
    study = read(run / 'study.json')
    arrays = np.load(run / 'prepared/arrays.npz')
    x, y = arrays['x'], arrays['y']
    oof = read(run / 'oof_predictions.json')
    assert len(rows) == len(x) == len(oof)
    predictions = {m: np.array([[r['models'][m]['utility'], r['models'][m]['wall_seconds']]
                                for r in oof]) for m in 'ABC'}
    probability = np.array([r['models']['C']['feasibility'] for r in oof])
    constant_probability = np.full(len(rows), np.nan)
    coverage = np.zeros(len(rows), dtype=int)
    export_error = 0.
    # Recompute from fold-specific JSON weights; no final refit is used here.
    for fold in range(5):
        split = read(run / f'fold_{fold}/split.json')
        sets = [set(split[name]['group_id']) for name in ['train', 'validation', 'test']]
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        for key in ['target_id', 'series_id', 'parent_graph_id', 'state_id']:
            a, b, c = [set(split[name][key]) for name in ['train', 'validation', 'test']]
            assert not (a & b or a & c or b & c), key
        test = np.array([i for i, r in enumerate(rows) if r['group_id'] in sets[2]])
        train = np.array([i for i, r in enumerate(rows) if r['group_id'] in sets[0]])
        coverage[test] += 1
        constant_probability[test] = y[train, 2].mean()
        for variant in 'ABC':
            model = read(run / f'fold_{fold}/{variant}.json')
            assert set(model['training_groups']) == sets[0]
            assert set(model['test_groups']) == sets[2]
            features = x[test, :len(model['feature_names'])]
            values = [predict_artifact(member, features) for member in model['members']]
            mean = np.mean([r[0] for r in values], axis=0)
            export_error = max(export_error, float(np.abs(mean - predictions[variant][test]).max()))
            assert np.allclose(mean, predictions[variant][test], atol=2e-5, rtol=1e-4)
            if variant == 'C':
                assert np.allclose(np.mean([r[1] for r in values], axis=0), probability[test], atol=2e-5)
    assert np.all(coverage == 1)
    regression = {}
    for variant, pred in predictions.items():
        measured = np.isfinite(y[:, 0])
        feasible = measured & (y[:, 2] > .5)
        regression[variant] = {
            'utility_mae_same_211_measured_rows': float(np.abs(pred[measured, 0] - y[measured, 0]).mean()),
            'utility_mae_same_121_feasible_rows': float(np.abs(pred[feasible, 0] - y[feasible, 0]).mean()),
            'cost_mae_same_278_rows_seconds': float(np.abs(pred[:, 1] - y[:, 1]).mean())}
    decisions = {m: read(run / f'decisions_{m}_forced.json') for m in 'ABC'}
    retained_parent_metrics = {}
    for variant, choices in decisions.items():
        eligible = [r for r in choices if r['parent_feasible']]
        accepted = [r for r in eligible if r['selected_feasible'] and r['best_feasible_gain'] > 0]
        retained_parent_metrics[variant] = {
            'states': len(eligible), 'accepted_positive_utility_children': len(accepted),
            'mean_final_metric_deltas_over_all_93_states': {
                key: sum(r['selected_metric_deltas'].get(key, 0.) for r in accepted) / len(eligible)
                for key in ['vina', 'qed', 'sa']}}
    menus = defaultdict(list)
    for r in rows:
        menus[r['state_id']].append(r)
    random = []
    for state, menu in sorted(menus.items()):
        gains = [max(0., r['utility_delta']) if r['feasible'] and r['utility_delta'] is not None else 0.
                 for r in menu]
        random.append({'state_id': state, 'group_id': menu[0]['group_id'],
                       'selected_feasible': float(np.mean([r['feasible'] for r in menu])),
                       'best_feasible_gain': float(np.mean(gains)) if menu[0]['parent_feasible'] else None})
    comparisons = {}
    for current, control in [('B', 'A'), ('C', 'B'), ('A', 'Random'), ('B', 'Random'), ('C', 'Random')]:
        comparisons[f'{current}-{control}'] = {
            k: cluster_interval(decisions[current], random if control == 'Random' else decisions[control], k)
            for k in ['selected_feasible', 'best_feasible_gain']}
    calibration = []
    for low, high in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        mask = (probability >= low) & ((probability < high) if high < 1 else (probability <= high))
        if mask.any():
            calibration.append({'low': float(low), 'high': float(high), 'rows': int(mask.sum()),
                                'predicted': float(probability[mask].mean()),
                                'observed': float(y[mask, 2].mean())})
    analysis = {'created_at_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds'),
                'source_run': str(run), 'scope_warning': summary['source_warning'],
                'oof_export_audit': {'rows': len(rows), 'folds': 5, 'one_test_fold_per_row': True,
                                     'max_regression_absolute_error': export_error},
                'common_row_regression_metrics': regression,
                'final_metrics_with_incumbent_retention': retained_parent_metrics,
                'feasibility_baseline': {'constant_from_each_fold_training_rows_brier':
                                        float(np.mean((constant_probability - y[:, 2]) ** 2)),
                                        'C_brier': summary['models']['C']['feasibility_brier'],
                                        'C_auroc': summary['models']['C']['feasibility_auroc'],
                                        'calibration_bins': calibration},
                'state_weighted_group_bootstrap_95': comparisons,
                'interpretation': 'All intervals are exploratory. Historical test-derived data were already inspected.'}
    (output / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#8b929c', '#7aa2dc', '#337655', '#cb9d3d', '#c8c9ce']
    names = ['Random', 'A: legacy inputs', 'B: + spatial', 'C: + feasibility', 'Observed oracle']
    feasible = [summary['reference_baselines']['random_expected_forced_feasible_rate']]
    gains = [summary['reference_baselines']['random_expected_best_feasible_gain']]
    feasible += [summary['models'][m]['forced_one_action']['feasible_yield'] for m in 'ABC']
    gains += [summary['models'][m]['forced_one_action']['mean_best_feasible_gain'] for m in 'ABC']
    feasible += [summary['reference_baselines']['observed_oracle_any_feasible_rate']]
    gains += [summary['reference_baselines']['observed_oracle_best_feasible_gain']]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for ax, values, title, label in [
            (axes[0], np.array(feasible) * 100, 'Usable selected candidate (97 states)', 'Feasible yield (%)'),
            (axes[1], gains, 'Retain parent if candidate is unusable or worse (93 states)', 'Mean best-feasible utility gain')]:
        bars = ax.bar(np.arange(5), values, color=colors)
        ax.set_xticks(np.arange(5), names, rotation=17, ha='right')
        ax.set_ylabel(label); ax.set_title(title, fontsize=10); ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, max(values) * 1.19)
        ax.bar_label(bars, fmt='%.1f' if ax is axes[0] else '%.4f', padding=4, fontsize=9)
    fig.suptitle('GPU Router A/B/C: exploratory grouped out-of-fold replay\nHistorical test-derived cohort; one forced choice, cost weight = 0', fontsize=11)
    fig.savefig(output / 'forced_comparison.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for model, color in zip('ABC', colors[1:4]):
        policies = summary['models'][model]['stop_policies']
        keys = list(policies)
        axes[0].plot(range(len(keys)), [100 * policies[k]['stop_rate'] for k in keys], 'o-', label=model, color=color)
        axes[1].plot(range(len(keys)), [policies[k]['mean_best_feasible_gain'] for k in keys], 'o-', label=model, color=color)
    for ax in axes:
        ax.set_xticks(range(4), ['0', '0.0002', '0.001', '0.01'])
        ax.set_xlabel('Predeclared cost weight (utility / second)'); ax.legend(title='Model')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('STOP (%)'); axes[1].set_ylabel('Mean best-feasible utility gain')
    fig.suptitle('STOP sensitivity: replay of recorded single-action costs\nWeights were not selected by these held-out outcomes', fontsize=11)
    fig.savefig(output / 'stop_sensitivity.png', dpi=180)
    plt.close(fig)
    return analysis, study, audit


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result, _, _ = analyze(args.run, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
