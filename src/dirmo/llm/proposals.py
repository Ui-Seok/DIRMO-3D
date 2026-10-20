from __future__ import annotations

import json
from pathlib import Path

from ..chemistry.graphs import ChemicalError, executable_edits, execute_edit, read_sdf, smiles
from ..io import append_jsonl, digest, write_json
from ..schemas import Proposal, State


def parse_proposal(raw: str, evidence_ids: set[str], parent_graph: str, candidate_cap: int) -> Proposal:
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    proposal = Proposal.model_validate_json(text)
    if len(proposal.edit_candidates) > candidate_cap:
        raise ValueError("Too many edit candidates")
    ids = [candidate.candidate_id for candidate in proposal.edit_candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate IDs")
    refs = set(proposal.evidence_ids)
    for h in proposal.hypotheses:
        refs.update(h.evidence_ids)
        if h.status != "untested":
            raise ValueError("LLM hypotheses must remain untested until action outcomes are measured")
    for c in proposal.edit_candidates:
        refs.update(c.evidence_ids)
        if c.parent_graph_id != parent_graph:
            raise ValueError("Candidate belongs to a different work-item graph")
    if refs - evidence_ids:
        raise ValueError(f"Unobserved evidence IDs: {sorted(refs - evidence_ids)}")
    return proposal


def messages(state: State, case, cap: int) -> list[dict]:
    mol = read_sdf(state.work_item.sdf_path)
    # The action space, measured by the same validator that will execute the choice.
    # Without it a planner guesses at the allowlist and most candidates are rejected.
    space = executable_edits(mol, read_sdf(state.original_ligand.sdf_path), case.protected)
    prompt = {
        "task": "Propose small chemical edits and testable hypotheses from measured facts. Return only JSON.",
        "parent_graph_id": state.work_item.graph_id,
        "canonical_smiles": smiles(mol),
        "mapped_smiles": state.work_item.mapped_smiles,
        "atom_table": [{"atom_map": atom.GetAtomMapNum(), "element": atom.GetSymbol(),
                        "neighbor_maps": [neighbor.GetAtomMapNum() for neighbor in atom.GetNeighbors()]}
                       for atom in mol.GetAtoms()],
        "measured_evidence": [e.model_dump(mode="json") for e in state.evidence
                              if e.source not in ("preservation.features", "posebusters.continuous")][:48],
        "observation": {"metrics": state.work_item.observation.metrics,
                        "failures": state.work_item.observation.failures,
                        "feasible": state.work_item.observation.feasible,
                        "graph_matches": state.work_item.observation.graph_matches},
        "executable_edits": space,
        "objective": case.objective.model_dump(mode="json"),
        "protected": case.protected.model_dump(mode="json"),
        "remaining_budget": state.remaining_budget,
        "maximum_candidates": cap,
        "rules": ["Use only the provided evidence IDs. Hypotheses are untested, not causal proofs.",
                  "executable_edits lists every operation that can run on this graph. Choose from it. "
                  "It is not a ranking; decide from the measured evidence which edit serves the objective.",
                  "Return compact JSON without markdown or indentation; use only fields in the schema.",
                  "Use at most two hypotheses and two tradeoffs. Keep each explanation or test under 30 words.",
                  "Do not calculate numerical properties or claim efficacy/toxicity/success probabilities.",
                  "Use complete unique atom maps; retain parent map IDs on retained atoms.",
                  "Prefer operation edits and set mapped_product_smiles to null. Do not supply both.",
                  "changed_atom_maps belongs to the candidate, never inside operation.",
                  "For replace_atom specify atom_map and element. For change_bond specify both atom_map and other_map.",
                  "One operation performs one local change. replace_atom changes only that atom's element; bonds and neighbors remain.",
                  "For replace_atom, changed_atom_maps contains only the replaced atom, and the new element must differ.",
                  "An atom replacement cannot remove an entire group or convert an aromatic ring to a saturated ring.",
                  "remove_terminal_atom requires exactly one neighbor; changed_atom_maps includes it and that neighbor.",
                  "If using mapped_product_smiles, every atom must use [element:map] notation, never unmapped SMILES.",
                  "The product must actually differ from the parent graph. Returning the parent mapped_smiles is invalid.",
                  "New atoms use IDs greater than the largest parent map ID.",
                  "changed_atom_maps must include every added/removed/changed atom AND affected bond endpoints.",
                  "Use mapped_product_smiles OR one allowlisted operation, never executable code.",
                  "preferred_action may be pose_resample, change_steps, stop, or edit:<candidate_id>.",
                  "Keep protection conditions. A failed pose does not prove the graph is impossible."],
        "json_schema": Proposal.model_json_schema(),
    }
    return [{"role": "system", "content": "You are a chemistry planning assistant. Tools supply all measurements."},
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))}]


class Planner:
    def __init__(self, client, config, run_dir: Path, budget, persist=None):
        self.client, self.config, self.run_dir, self.budget = client, config, run_dir, budget
        self.persist = persist

    def propose(self, state, case) -> Proposal | None:
        cfg = self.config
        prompt = messages(state, case, case.budgets.candidate_cap)
        health = self.client.health()
        model_record = self.run_dir / "llm_model.json"
        if model_record.exists() and digest(json.loads(model_record.read_text())) != digest(health["model_identity"]):
            raise ValueError("Resident LLM weights/environment changed; start a new run instead of reusing outcomes")
        write_json(model_record, health["model_identity"])
        key = digest({"model": health["model_identity"], "generation": cfg.model_dump(mode="json"),
                      "prompt": prompt})
        cache = self.run_dir / "cache" / "llm" / f"{key}.json"
        evidence = {e.evidence_id for e in state.evidence}
        if cache.exists():
            result = json.loads(cache.read_text())
            proposal = parse_proposal(result["raw"], evidence, state.work_item.graph_id, case.budgets.candidate_cap)
            append_jsonl(self.run_dir / "llm.jsonl", {"cache_hit": True, "cache_key": key, "actual_cost": 0})
            return proposal
        for retry in range(cfg.parsing_retries + 1):
            # Reserve input + output upper bound before each attempt. Refund only reported unused tokens.
            token_cap = cfg.context_cap
            self.budget.charge("llm_attempt", llm_calls=1, llm_tokens=token_cap)
            if self.persist:
                self.persist({"stage": "llm", "cache_key": key, "retry": retry})
            try:
                response = self.client.request("/propose", {"messages": prompt},
                                               timeout=min(600, self.budget.remaining()["wall_seconds"]))
            except Exception as exc:
                append_jsonl(self.run_dir / "llm.jsonl", {"retry": retry, "error": str(exc),
                            "prompt": prompt, "cost_accounting": "full reservation retained; remote cost unknown"})
                return None
            tokens = response.get("total_tokens")
            if tokens is not None and 0 <= tokens <= token_cap:
                self.budget.refund("llm_actual_tokens", llm_tokens=token_cap - tokens)
            event = {"retry": retry, "cache_key": key, "cache_hit": False, "prompt": prompt, **response}
            try:
                if response.get("status") != "ok":
                    raise ValueError(response.get("reason", "LLM failed"))
                proposal = parse_proposal(response["raw"], evidence, state.work_item.graph_id,
                                          case.budgets.candidate_cap)
                valid, rejects, seen = [], [], set(state.candidate_history)
                for candidate in proposal.edit_candidates:
                    try:
                        mol, _ = execute_edit(read_sdf(state.work_item.sdf_path), candidate,
                                              read_sdf(state.original_ligand.sdf_path), case.protected)
                        from ..chemistry.graphs import graph_id
                        gid = graph_id(mol)
                        if gid in seen:
                            raise ChemicalError("duplicate_candidate_graph")
                        valid.append(candidate)
                        seen.add(gid)
                    except ChemicalError as exc:
                        rejects.append({"candidate_id": candidate.candidate_id, "reason": str(exc)})
                event.update(parsing_valid=True, chemical_valid_count=len(valid), rejected=rejects)
                proposal.edit_candidates = valid
                if rejects and not valid:
                    raise ChemicalError("All proposed edits were rejected: " + json.dumps(rejects))
                write_json(cache, {"raw": proposal.model_dump_json(), "model": health["model_identity"]})
                append_jsonl(self.run_dir / "llm.jsonl", event)
                return proposal
            except (ValueError, KeyError) as exc:
                if isinstance(exc, ChemicalError):
                    event.update(chemical_valid=False, chemical_error=str(exc))
                else:
                    event.update(parsing_valid=False, parsing_error=str(exc))
                append_jsonl(self.run_dir / "llm.jsonl", event)
                prompt = [*prompt, {"role": "assistant", "content": response.get("raw", "")},
                          {"role": "user", "content": f"Return valid JSON only. Validation error: {exc}"}]
        return None
