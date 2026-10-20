from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class MetricGoal(Schema):
    direction: Literal["maximize", "minimize"]
    target: float
    weight: float = Field(default=1.0, ge=0)
    max_loss: float | None = Field(default=None, ge=0)


class Objective(Schema):
    metrics: dict[str, MetricGoal] = Field(default_factory=lambda: {
        "qed": MetricGoal(direction="maximize", target=0.6),
        "sa": MetricGoal(direction="minimize", target=4.0, weight=0.1, max_loss=0.5),
        "vina": MetricGoal(direction="minimize", target=-6.0, weight=0.1, max_loss=0.5),
    })
    minimum_improvement: float = Field(default=0.01, gt=0)
    required_evaluations: list[Literal["properties", "geometry", "posebusters", "vina", "interactions"]] = (
        Field(default_factory=lambda: ["properties", "geometry", "posebusters", "vina"])
    )

    @model_validator(mode="after")
    def known_metrics(self):
        known = {"qed", "sa", "mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "vina"}
        if not self.metrics or set(self.metrics) - known:
            raise ValueError("Objective must use supported measured metrics")
        return self


class BudgetLimits(Schema):
    fm_draws: int = Field(default=4, ge=0)
    fm_nfe: int = Field(default=512, ge=0)
    llm_calls: int = Field(default=3, ge=0)
    llm_tokens: int = Field(default=12000, ge=0)
    evaluation_calls: int = Field(default=8, ge=1)
    wall_seconds: float = Field(default=1800, gt=0)
    max_rounds: int = Field(default=2, ge=0)
    candidate_cap: int = Field(default=2, ge=1, le=16)
    seeds_per_graph: int = Field(default=2, ge=1, le=16)
    plateau_rounds: int = Field(default=2, ge=1)


class ProtectedFeature(Schema):
    family: str
    atom_maps: list[int] = Field(min_length=1)
    reference_position: tuple[float, float, float]
    max_distance: float = Field(default=1.5, gt=0)
    hard: bool = True


class ProtectedInteraction(Schema):
    residue_id: str
    role: str
    hard: bool = True


class Protection(Schema):
    atom_maps: list[int] = Field(default_factory=list)
    smarts: list[str] = Field(default_factory=list)
    features: list[ProtectedFeature] = Field(default_factory=list)
    interactions: list[ProtectedInteraction] = Field(default_factory=list)
    max_atom_displacement: float | None = Field(default=None, gt=0)


class Case(Schema):
    case_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    target_id: str
    series_id: str | None = None
    protein_path: str | None = None
    pocket_path: str | None = None
    initial_ligand_path: str
    receptor_path: str | None = None
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] = (24.0, 24.0, 24.0)
    coordinate_units: Literal["angstrom"] = "angstrom"
    preparation_notes: str = "As supplied; no protonation or receptor repair performed."
    objective: Objective = Field(default_factory=Objective)
    protected: Protection = Field(default_factory=Protection)
    budgets: BudgetLimits = Field(default_factory=BudgetLimits)

    @model_validator(mode="after")
    def validate_input(self):
        if not (self.protein_path or self.pocket_path):
            raise ValueError("protein_path or pocket_path is required")
        if any(v <= 0 for v in self.box_size):
            raise ValueError("box_size must be positive")
        return self


class EditOperation(Schema):
    kind: Literal["replace_atom", "add_terminal_atom", "remove_terminal_atom", "change_bond"]
    atom_map: int = Field(gt=0)
    other_map: int | None = Field(default=None, gt=0)
    element: str | None = None
    bond_order: Literal[1, 2, 3] = 1
    formal_charge: int = Field(default=0, ge=-2, le=2)


class EditCandidate(Schema):
    candidate_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    parent_graph_id: str
    changed_atom_maps: list[int] = Field(min_length=1)
    mapped_product_smiles: str | None = None
    operation: EditOperation | None = None
    atom_correspondence: dict[int, int] = Field(default_factory=dict)
    protection_explanation: str
    evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def one_edit(self):
        if (self.mapped_product_smiles is None) == (self.operation is None):
            raise ValueError("Supply exactly one of mapped_product_smiles or operation")
        return self


class Hypothesis(Schema):
    hypothesis_id: str
    statement: str
    evidence_ids: list[str] = Field(min_length=1)
    test: str
    status: Literal["untested", "supported", "contradicted", "uncertain"] = "untested"


class Proposal(Schema):
    hypotheses: list[Hypothesis]
    evidence_ids: list[str]
    edit_candidates: list[EditCandidate]
    expected_tradeoffs: list[str]
    preferred_action: str | None = None


class ActionKind(str, Enum):
    CHEMICAL_EDIT = "CHEMICAL_EDIT"
    # Free a region and let the generator decide what goes there. The product graph is an
    # OUTPUT of this action, unlike CHEMICAL_EDIT where the caller specifies it exactly.
    REDESIGN_REGION = "REDESIGN_REGION"
    POSE_RESAMPLE = "POSE_RESAMPLE"
    CHANGE_GENERATION_CONDITION = "CHANGE_GENERATION_CONDITION"
    STOP = "STOP"
    EXTRA_VALIDATION = "EXTRA_VALIDATION"


class GenerationParameters(Schema):
    # "reference" starts integration from the DIRMO reference conformer instead of a
    # Gaussian draw. Upstream FLOWR's own "conformer" mode is a different mechanism
    # (its own cached ETKDG generator) and is not implemented here.
    prior: Literal["random", "reference", "fragment"] = "random"
    # Seeded isotropic displacement of the reference prior, in angstrom. The reference
    # conformer is deterministic and this checkpoint integrates coordinates with
    # integration-coord-noise-std=0, so without this two seeds are the same draw.
    # 0.3 matches the checkpoint's train-ligand-coord-noise-std.
    prior_noise_std: float = Field(default=0.0, ge=0, le=2.0)
    # Concentrate extra displacement on atoms a measurement flagged, instead of shaking the
    # whole ligand: the atoms left alone anchor the molecule while the model re-places the
    # few that clash or strain. Empty list means uniform prior_noise_std everywhere.
    prior_noise_atom_maps: list[int] = Field(default_factory=list)
    prior_noise_focus_std: float = Field(default=0.0, ge=0, le=4.0)
    # "fragment": hold the listed atoms' measured coordinates AND graph, and let the model
    # redesign the rest -- elements, charges and the bonds that touch them. Unlike the
    # whole-graph priors this is a mode the checkpoint was actually trained on, but it
    # means the product graph is an OUTPUT, not something the caller specifies.
    prior_free_atom_maps: list[int] = Field(default_factory=list)
    solver: Literal["euler"] = "euler"

    @model_validator(mode="after")
    def focused_noise_is_consistent(self):
        if self.prior_noise_atom_maps:
            if any(v <= 0 for v in self.prior_noise_atom_maps):
                raise ValueError("prior_noise_atom_maps must be positive atom map IDs")
            if len(set(self.prior_noise_atom_maps)) != len(self.prior_noise_atom_maps):
                raise ValueError("Repeated atom maps in prior_noise_atom_maps")
            if not self.prior_noise_focus_std:
                raise ValueError("prior_noise_atom_maps needs a nonzero prior_noise_focus_std")
        elif self.prior_noise_focus_std:
            raise ValueError("prior_noise_focus_std needs prior_noise_atom_maps")
        if (self.prior == "fragment") != bool(self.prior_free_atom_maps):
            raise ValueError("The fragment prior needs prior_free_atom_maps, and only it may carry them")
        if self.prior_free_atom_maps and (
                any(v <= 0 for v in self.prior_free_atom_maps)
                or len(set(self.prior_free_atom_maps)) != len(self.prior_free_atom_maps)):
            raise ValueError("prior_free_atom_maps must be distinct positive atom map IDs")
        return self
    integration_steps: int = Field(default=64, ge=1, le=256)
    # Unsupported fields (e.g. QED guidance) are rejected by extra='forbid'.


def kernel_identity(kernel_base: dict, parameters: GenerationParameters, pocket_sha256: str) -> dict:
    """The complete generation kernel.

    The FM worker records this with every draw and the orchestrator recomputes it to
    gate the learned router and to deduplicate requests. Both sides must agree, so the
    composition lives here rather than in either of them. Anything that changes what a
    draw means belongs in this dict: outcomes under different kernels are different
    experiments and must never share reward labels.
    """
    return {**kernel_base, "integration_steps": parameters.integration_steps,
            "prior": parameters.prior, "prior_noise_std": parameters.prior_noise_std,
            "pocket_sha256": pocket_sha256}


def generation_kernel(kernel: dict) -> dict:
    """The kernel without the pocket: how a draw is made, not which task it was made for.

    Reward labels from different generation conditions must never mix, but labels from
    different pockets must mix, because generalising across pockets is the point of
    grouped cross-validation. ``kernel_identity`` keeps the pocket so a repeated
    graph/kernel/seed request is still recognised as a repeat; this view drops it so one
    dataset can span many targets under one generation condition.
    """
    return {key: value for key, value in kernel.items() if key != "pocket_sha256"}


class Action(Schema):
    action_id: str
    kind: ActionKind
    candidate: EditCandidate | None = None
    region: list[int] = Field(default_factory=list)
    generation: GenerationParameters = Field(default_factory=GenerationParameters)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1, max_length=16)
    evidence_ids: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def action_fields(self):
        if (self.kind == ActionKind.CHEMICAL_EDIT) != (self.candidate is not None):
            raise ValueError("CHEMICAL_EDIT requires a candidate; other actions cannot carry one")
        if (self.kind == ActionKind.REDESIGN_REGION) != bool(self.region):
            raise ValueError("REDESIGN_REGION requires a region; other actions cannot carry one")
        if self.region and sorted(self.region) != sorted(self.generation.prior_free_atom_maps):
            raise ValueError("The action's region must be the generation's free atom maps")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Repeated seeds are not independent draws")
        return self


class EvaluationRecord(Schema):
    status: Literal["ok", "missing", "failed", "disabled"]
    values: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    version: str | None = None
    latency_seconds: float = Field(default=0, ge=0)


class Observation(Schema):
    observation_id: str
    graph_id: str
    evaluations: dict[str, EvaluationRecord]
    metrics: dict[str, float] = Field(default_factory=dict)
    graph_valid: bool = True
    graph_matches: bool = True
    hard_constraints_pass: bool = True
    feasible: bool = False
    target_met: bool = False
    incomplete_evaluation: bool = False
    utility: float | None = None
    failures: list[str] = Field(default_factory=list)


class WorkItem(Schema):
    item_id: str
    graph_id: str
    mapped_smiles: str
    sdf_path: str
    atom_mapping: dict[int, int] = Field(default_factory=dict)
    observation: Observation | None = None
    seed: int | None = None
    kernel_id: str | None = None
    source: Literal["initial", "fm_raw", "input_conformer"] = "initial"


class Evidence(Schema):
    evidence_id: str
    observation_id: str
    source: str
    fact: str


class State(Schema):
    state_id: str
    case_id: str
    original_ligand: WorkItem
    best_feasible: WorkItem | None = None
    work_item: WorkItem
    round_index: int = 0
    plateau: int = 0
    history: list[WorkItem] = Field(default_factory=list)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    graph_attempts: dict[str, int] = Field(default_factory=dict)
    generation_attempts: list[str] = Field(default_factory=list)
    budget_used: dict[str, float] = Field(default_factory=dict)
    remaining_budget: dict[str, float] = Field(default_factory=dict)
    candidate_history: list[str] = Field(default_factory=list)
    proposal: Proposal | None = None
    status: str = "running"


class GenerationRequest(Schema):
    graph_smiles: str
    graph_id: str
    pocket_path: str
    reference_sdf: str
    protected_atom_mapping: dict[int, int] = Field(default_factory=dict)
    parameters: GenerationParameters = Field(default_factory=GenerationParameters)
    seed: int
    output_dir: str


class GenerationResult(Schema):
    status: Literal["ok", "failed", "graph_mismatch", "capability_error", "oom"]
    sdf_path: str | None = None
    raw_draws: int = Field(default=0, ge=0)
    nfe: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0, ge=0)
    kernel_id: str | None = None
    graph_check: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    peak_memory_bytes: int = 0
