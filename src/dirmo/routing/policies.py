from __future__ import annotations

from ..chemistry.graphs import candidate_regions, execute_edit, graph_id, read_sdf
from ..schemas import Action, ActionKind, GenerationParameters


def available_actions(state, case, generation: GenerationParameters, capabilities: dict,
                      nfe_factor: int = 1, seed_base: int = 42, extra_candidates=(),
                      regions: int = 0) -> list[Action]:
    evidence = [e.evidence_id for e in state.evidence]
    stop = Action(action_id="stop", kind=ActionKind.STOP, reason="Return best feasible history entry")
    if not capabilities.get("fixed_graph") or state.remaining_budget.get("wall_seconds", 0) <= 0:
        return [stop]
    remaining = state.remaining_budget
    if remaining.get("fm_draws", 0) < 1 or remaining.get("evaluation_calls", 0) < 1:
        return [stop]
    actions = []
    # The two conditioning modes are mutually exclusive in one worker. Whole-graph actions
    # cannot preserve the graph without graph_inpainting, and a redesign cannot hold its
    # boundary without inpainting_mode, both measured. Offering an action the loaded mode
    # cannot execute would spend a draw on a certain failure, so mask it here -- before any
    # policy sees it -- rather than discovering it per draw.
    mode = capabilities.get("graph_mode", "graph")
    used = state.graph_attempts.get(state.work_item.graph_id, 0)
    seed = seed_base + state.round_index * 1000 + used
    if mode == "graph" and used < case.budgets.seeds_per_graph:
        actions.append(Action(action_id="pose_resample", kind=ActionKind.POSE_RESAMPLE,
                   generation=generation, seeds=[seed], evidence_ids=evidence,
                   reason="Test recoverability of the current work-item graph"))
        # Euler/random were inspected in upstream. Only NFE changes are exposed.
        if "integration_steps" in capabilities.get("condition_changes", []) and generation.integration_steps < 128:
            alternate = generation.model_copy(update={"integration_steps": min(128, generation.integration_steps * 2)})
            actions.append(Action(action_id="change_steps", kind=ActionKind.CHANGE_GENERATION_CONDITION,
                   generation=alternate, seeds=[seed], evidence_ids=evidence,
                   reason="Test whether more Euler steps help the same graph; new kernel is recorded"))
    planned = (list(state.proposal.edit_candidates[:case.budgets.candidate_cap])
               if state.proposal and mode == "graph" else [])
    # Collection candidates are appended AFTER the planner's cap, so the planner's own
    # condition is identical whether or not an oracle arm is present.
    pending = [*planned, *(extra_candidates if mode == "graph" else ())]
    if pending:
        parent = read_sdf(state.work_item.sdf_path)
        original = read_sdf(state.original_ligand.sdf_path)
        offered = set()
        for candidate in pending:
            try:
                product, _ = execute_edit(parent, candidate, original, case.protected)
                gid = graph_id(product)
                if gid in state.candidate_history or gid in offered or (
                        state.graph_attempts.get(gid, 0) >= case.budgets.seeds_per_graph):
                    continue
                offered.add(gid)
                actions.append(Action(action_id=f"edit:{candidate.candidate_id}", kind=ActionKind.CHEMICAL_EDIT,
                        candidate=candidate, generation=generation, seeds=[seed],
                        evidence_ids=candidate.evidence_ids, reason=candidate.protection_explanation))
            except ValueError:
                continue
    if regions and mode == "fragment" and "fragment" in capabilities.get("priors", []):
        parent = read_sdf(state.work_item.sdf_path)
        for region in candidate_regions(parent, case.protected, regions):
            condition = generation.model_copy(update={"prior": "fragment", "prior_noise_std": 0.0,
                                                      "prior_noise_atom_maps": [],
                                                      "prior_noise_focus_std": 0.0,
                                                      "prior_free_atom_maps": list(region)})
            actions.append(Action(action_id="redesign:" + "-".join(str(v) for v in region),
                    kind=ActionKind.REDESIGN_REGION, region=list(region), generation=condition,
                    seeds=[seed], evidence_ids=evidence,
                    reason=f"Free atom maps {region} and let the generator choose what goes there; "
                           "the product graph is measured, not requested"))
    # Mask infeasible actions before ANY policy, including learned and baselines.
    return [a for a in actions if (a.generation.integration_steps + 1) * nfe_factor <= remaining.get("fm_nfe", 0)] + [stop]


class RuleRouter:
    name = "rule"

    def select(self, state, actions: list[Action]) -> Action:
        lookup = {kind: [a for a in actions if a.kind == kind] for kind in ActionKind}
        observation = state.work_item.observation
        stop = lookup[ActionKind.STOP][0]
        if observation and any(f.endswith((":missing", ":disabled", ":failed")) for f in observation.failures):
            return stop.model_copy(update={"reason": "Required external evaluation is unavailable"})
        if observation and observation.feasible and observation.target_met and state.best_feasible:
            return stop.model_copy(update={"reason": "Feasible work item meets fixed targets"})
        pose_failed = observation is None or any(f in ("geometry:invalid", "posebusters:invalid", "fm_failed")
                                                for f in observation.failures)
        if pose_failed:
            if state.graph_attempts.get(state.work_item.graph_id, 0) >= 1 and lookup[ActionKind.CHANGE_GENERATION_CONDITION]:
                return lookup[ActionKind.CHANGE_GENERATION_CONDITION][0]
            if lookup[ActionKind.POSE_RESAMPLE]:
                return lookup[ActionKind.POSE_RESAMPLE][0]
            if lookup[ActionKind.CHANGE_GENERATION_CONDITION]:
                return lookup[ActionKind.CHANGE_GENERATION_CONDITION][0]
        if lookup[ActionKind.CHEMICAL_EDIT]:
            return lookup[ActionKind.CHEMICAL_EDIT][0]
        return stop.model_copy(update={"reason": "No executable improving/recovery action within the caps"})


class FixedActionRouter:
    def __init__(self, kind: ActionKind):
        self.kind, self.name = kind, f"fixed:{kind.value}"

    def select(self, state, actions):
        return next((a for a in actions if a.kind == self.kind), next(a for a in actions if a.kind == ActionKind.STOP))


class FixedSequenceRouter:
    name = "fixed_sequence"

    def __init__(self, sequence=None):
        self.sequence = sequence or [ActionKind.CHEMICAL_EDIT, ActionKind.POSE_RESAMPLE]

    def select(self, state, actions):
        return FixedActionRouter(self.sequence[state.round_index % len(self.sequence)]).select(state, actions)


class LLMRouterBaseline:
    name = "llm_baseline"

    def select(self, state, actions):
        preferred = state.proposal.preferred_action if state.proposal else None
        return next((a for a in actions if a.action_id == preferred), next(a for a in actions if a.kind == ActionKind.STOP))
