from ..chemistry.graphs import execute_edit, read_sdf, smiles
from ..chemistry.properties import graph_properties
from ..diagnosis import wilson
from ..schemas import ActionKind


METRICS = ["qed", "sa", "mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "vina"]
RESOURCES = ["fm_draws", "fm_nfe", "llm_calls", "llm_tokens", "evaluation_calls", "wall_seconds"]
KINDS = [ActionKind.CHEMICAL_EDIT, ActionKind.POSE_RESAMPLE, ActionKind.CHANGE_GENERATION_CONDITION, ActionKind.STOP]
GOAL_FIELDS = ["target", "weight", "direction", "max_loss", "present"]
FEATURE_NAMES = ([f"state_{k}" for k in METRICS] + [f"state_{k}_missing" for k in METRICS] +
                 ["round", "plateau", "attempts", "observed_feasible", "wilson_width", "prior_failures"] +
                 [f"remaining_{k}" for k in RESOURCES] +
                 [f"objective_{metric}_{field}" for metric in METRICS for field in GOAL_FIELDS] +
                 ["protected_atom_count", "protected_feature_count", "protected_interaction_count"] +
                 [f"action_{k.value}" for k in KINDS] +
                 ["action_steps", "action_seeds", "changed_atoms"] + [f"proposed_delta_{k}" for k in METRICS[:-1]])


def features(state, action, case) -> list[float]:
    """Use only pre-action state and proposed action, never its observed result."""
    obs = state.work_item.observation
    metrics = obs.metrics if obs else {}
    attempts = state.graph_attempts.get(state.work_item.graph_id, 0)
    successes = sum(1 for item in state.history if item.graph_id == state.work_item.graph_id and
                    item.source == "fm_raw" and item.observation and item.observation.feasible)
    interval = wilson(min(successes, attempts), attempts)["interval"]
    values = [metrics.get(k, 0) for k in METRICS] + [float(k not in metrics) for k in METRICS]
    values += [state.round_index, state.plateau, attempts, float(bool(obs and obs.feasible)),
               interval[1] - interval[0] if interval else 1.0, len(obs.failures) if obs else 0]
    values += [state.remaining_budget.get(k, 0) for k in RESOURCES]
    for key in METRICS:
        goal = case.objective.metrics.get(key)
        values += [goal.target, goal.weight, 1 if goal.direction == "maximize" else -1,
                   goal.max_loss if goal.max_loss is not None else -1, 1] if goal else [0, 0, 0, -1, 0]
    values += [len(case.protected.atom_maps), len(case.protected.features), len(case.protected.interactions)]
    values += [float(action.kind == k) for k in KINDS]
    values += [action.generation.integration_steps, len(action.seeds),
               len(action.candidate.changed_atom_maps) if action.candidate else 0]
    delta = {k: 0 for k in METRICS[:-1]}
    if action.candidate:
        parent = read_sdf(state.work_item.sdf_path)
        product, _ = execute_edit(parent, action.candidate, read_sdf(state.original_ligand.sdf_path), case.protected)
        before, after = graph_properties(smiles(parent)), graph_properties(smiles(product))
        delta = {k: after[k] - before[k] for k in delta}
    return [float(v) for v in values] + [delta[k] for k in METRICS[:-1]]
