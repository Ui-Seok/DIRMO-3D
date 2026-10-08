from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

from ..chemistry.graphs import graph_check, read_sdf, smiles, write_sdf
from ..gpu import assert_worker_mask
from ..io import digest, environment, revision, sha256_file, write_json
from ..schemas import GenerationRequest, GenerationResult


class CapabilityError(RuntimeError):
    pass


def prepare_random_graph_prior(interpolant):
    """Fix the optional-feature random prior on this interpolant instance only."""
    sampler = interpolant.prior_sampler
    if sampler.n_hybridization_types is not None:
        return interpolant
    original = sampler.inpaint_graph

    def inpaint_graph(to_mol, from_mol, mask, mode="random"):
        if mode != "random" or mask is None or not mask.any():
            return original(to_mol, from_mol, mask, mode=mode)
        import torch
        from flowr.util.molrepr import GeometricMol
        # Match upstream's random branch, including its RNG draw and centering.
        # The base checkpoint has no hybridization vocabulary; upstream omitted
        # the None initialization in this branch. No model features are added.
        result = GeometricMol(
            sampler.coord_dist.sample((to_mol.seq_length, 3)),
            to_mol.atomics.clone(), charges=to_mol.charges.clone(), hybridization=None,
            bond_indices=to_mol.bond_indices.clone(), bond_types=to_mol.bond_types.clone(),
            fragment_mask=torch.ones(to_mol.seq_length, dtype=torch.bool, device=to_mol.coords.device),
            fragment_mode="graph",
        )
        return result.zero_com() if sampler.zero_com else result

    sampler.inpaint_graph = inpaint_graph
    return interpolant


def audit_paths(config) -> dict:
    blockers = []
    if not config.repo_path or not Path(config.repo_path).is_dir():
        blockers.append("Missing FLOWR.root repository")
    elif revision(config.repo_path) != config.repo_revision:
        blockers.append("FLOWR.root revision mismatch")
    elif __import__("subprocess").check_output(["git", "-C", config.repo_path, "status", "--porcelain"],
                                             text=True).strip():
        blockers.append("FLOWR.root working tree is dirty; pin an audited clean revision")
    if not config.checkpoint_path or not Path(config.checkpoint_path).is_file():
        blockers.append("Missing FLOWR.root checkpoint; older FLOWR and LiFT checkpoints are incompatible")
    if not config.python or not Path(config.python).is_file():
        blockers.append("Missing separate Python 3.12 FLOWR.root environment")
    return {"fixed_graph": not blockers, "fixed_graph_status": "blocked" if blockers else "pending_conformance",
            "local_fragment": False, "semantic_joint_generation": False, "native_scalar_guidance": False,
            "priors": ["random"], "solvers": ["euler"], "condition_changes": ["integration_steps"],
            "blockers": blockers, "checkpoint_compatibility": "unverified until real checkpoint load and probes"}


class FlowrBackend:
    def __init__(self, config):
        assert_worker_mask()
        audit = audit_paths(config)
        if audit["blockers"]:
            raise CapabilityError("; ".join(audit["blockers"]))
        if sys.version_info[:2] != (3, 12):
            raise CapabilityError("Pinned FLOWR.root installation requires Python 3.12")
        import torch
        self.torch, self.config = torch, config
        if torch.cuda.device_count() != 1:
            raise CapabilityError("FM worker must see one GPU only")
        sys.path.insert(0, config.repo_path)
        import flowr
        if not Path(flowr.__file__).resolve().is_relative_to(Path(config.repo_path).resolve()):
            raise CapabilityError("A different flowr package shadows the pinned FLOWR.root source")
        from flowr.gen.generate_from_pdb import get_args
        from flowr.scriptutil import load_model
        argv = sys.argv
        try:
            # Obtain upstream defaults without running its sampling/filter/oversampling loop.
            sys.argv = ["dirmo-fm", "--arch", "pocket", "--pocket_type", "holo",
                        "--graph_inpainting", "random", "--no_ligand_valence_repair"]
            self.args = get_args()
        finally:
            sys.argv = argv
        for k, v in {"ckpt_path": config.checkpoint_path, "gpus": 1, "num_workers": 0,
                     "batch_cost": 1, "max_sample_iter": 0, "dataset": "spindr", "cut_pocket": False,
                     "data_path": ".", "save_dir": ".", "filter_valid_unique": False,
                     "sample_mol_sizes": False, "max_pocket_size": config.max_pocket_atoms,
                     "integration_steps": config.parameters.integration_steps, "final_inpaint": True}.items():
            setattr(self.args, k, v)
        checkpoint_hash = sha256_file(config.checkpoint_path)
        if config.checkpoint_sha256 and checkpoint_hash != config.checkpoint_sha256:
            raise CapabilityError("Checkpoint SHA256 mismatch")
        values = load_model(self.args)
        self.model, self.hparams, *self.vocabs = values
        if self.hparams["pocket_noise"] != "fix" or self.hparams["coord_scale"] != 1.0:
            raise CapabilityError("This adapter requires a rigid pocket and coord_scale=1.0")
        if not self.model.graph_inpainting or self.model.inpainting_mode:
            raise CapabilityError("Whole-graph conditioning requires graph_inpainting with fragment modes off")
        if self.model.builder.ligand_valence_repair:
            raise CapabilityError("Raw evaluation requires valence repair disabled")
        if "coords" in self.model.feature_keys:
            raise CapabilityError("Final graph inpainting must fix discrete features, never coordinates")
        self.model.to("cuda:0").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.kernel_base = {"repo_revision": config.repo_revision, "checkpoint_sha256": checkpoint_hash,
                            "preparation": "fixed precomputed pocket; no recut; no protonation; upstream tensorization",
                            "graph_mode": "whole_graph", "prior": "random", "solver": "euler",
                            "final_inpaint": "discrete atom/bond/charge features only; coordinates stay generated",
                            "valence_repair": False, "relaxation": False, "filtering": False, "batch": 1,
                            "cat_noise_euler_guard": self.args.cat_noise_euler_guard,
                            "categorical_strategy": self.args.categorical_strategy,
                            "adapter_compatibility": ["random_graph_prior_optional_hybridization_v1"],
                            "sc_recycle_steps": getattr(self.model, "sc_recycle_steps", 1),
                            "hyperparameters_hash": digest(dict(self.hparams)), "environment": environment()}
        self.checkpoint_identity = {"path": config.checkpoint_path, "sha256": checkpoint_hash,
                                    "repo_revision": config.repo_revision, "hyperparameters": dict(self.hparams)}
        self.pocket_tensor_hashes = {}

    def capabilities(self):
        return {**audit_paths(self.config), "fixed_graph_status": "loaded_unverified_quality",
                "fixed_graph": True, "kernel_base": self.kernel_base}

    def generate_local_fragment(self, request):
        raise CapabilityError("Local fragment generation has not passed checkpoint conformance and is disabled")

    def generate_fixed_graph(self, request: GenerationRequest) -> GenerationResult:
        from flowr.data.dataset import GeometricDataset
        from flowr.gen import utils as util
        from flowr.gen.generate import generate_ligands_per_target
        from flowr.util.pocket import PocketComplexBatch
        from ..chemistry.graphs import graph_id
        start, torch = time.monotonic(), self.torch
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        expected = read_sdf(request.reference_sdf)
        if graph_id(expected) != request.graph_id or smiles(expected, True) != request.graph_smiles:
            raise CapabilityError("Reference input graph and typed generation request disagree")
        if expected.GetNumAtoms() > self.config.max_atoms:
            raise CapabilityError("Ligand atom cap exceeded")
        args = copy.copy(self.args)
        args.pdb_file, args.ligand_file = request.pocket_path, request.reference_sdf
        args.seed, args.save_dir = request.seed, str(output_dir)
        args.integration_steps = request.parameters.integration_steps
        self.model.integrator.steps = args.integration_steps
        args.solver, args.graph_inpainting = request.parameters.solver, request.parameters.prior
        kernel = {**self.kernel_base, "integration_steps": args.integration_steps,
                  "pocket_sha256": sha256_file(request.pocket_path)}
        kernel_id = digest(kernel)
        write_json(output_dir / "kernel.json", kernel)
        write_json(output_dir / "request.json", request)
        try:
            transform, interpolant = util.load_util(args, self.hparams, *self.vocabs[:4])
            interpolant = prepare_random_graph_prior(interpolant)
            system = util.load_data_from_pdb(args, remove_hs=self.hparams["remove_hs"],
                                             remove_aromaticity=self.hparams["remove_aromaticity"])
            pocket_tensor_hash = digest(system.holo.mol.coords.tolist())
            pocket_key = kernel["pocket_sha256"]
            previous = self.pocket_tensor_hashes.setdefault(pocket_key, pocket_tensor_hash)
            if previous != pocket_tensor_hash:
                raise CapabilityError("Fixed-pocket tensor coordinates changed between work items")
            write_json(output_dir / "pocket_frame.json", {"pocket_file_sha256": pocket_key,
                       "upstream_absolute_pocket_coords_sha256": pocket_tensor_hash,
                       "preprocessing_cut_pocket": False, "random_rotation": False,
                       "upstream_output_transform": "undo_zero_com_batch adds the fixed pocket COM"})
            dataset = GeometricDataset(PocketComplexBatch([system]), data_cls=PocketComplexBatch, transform=transform)
            prior, posterior, _, _ = next(iter(util.get_dataloader(args, dataset, interpolant)))
            # Decode reference tensors only; no model forward or draw has started.
            references = self.model._generate_ligs(posterior, lig_mask=posterior["lig_mask"].bool(), scale=1.0)
            if len(references) != 1 or references[0] is None or not graph_check(
                    expected, references[0], derive_stereo=True)["matches"]:
                raise CapabilityError("Upstream graph tensorization cannot preserve this element/charge/stereo graph")
        except CapabilityError:
            raise
        except Exception as exc:
            return GenerationResult(status="failed", raw_draws=0, nfe=0, kernel_id=kernel_id,
                reason=f"preprocessing: {type(exc).__name__}: {exc}",
                latency_seconds=time.monotonic() - start,
                peak_memory_bytes=torch.cuda.max_memory_allocated())
        nfe = 0

        def count_forward(_module, _args):
            nonlocal nfe
            nfe += 1

        handle = self.model.register_forward_pre_hook(count_forward)
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                generated = generate_ligands_per_target(args, self.model, prior, posterior, device="cuda:0")
            if len(generated) != 1:
                raise RuntimeError("Adapter expects exactly one raw result slot")
            if generated[0] is None:
                return GenerationResult(status="failed", raw_draws=1, nfe=nfe, kernel_id=kernel_id,
                    reason="raw_molecule_build_failure", latency_seconds=time.monotonic() - start,
                    peak_memory_bytes=torch.cuda.max_memory_allocated())
            raw = generated[0]
            raw_path = output_dir / "raw.sdf"
            write_sdf(raw_path, raw)
            check = graph_check(expected, raw, derive_stereo=True)
            write_json(output_dir / "graph_check.json", check)
            return GenerationResult(status="ok" if check["matches"] else "graph_mismatch",
                sdf_path=str(raw_path), raw_draws=1, nfe=nfe, kernel_id=kernel_id, graph_check=check,
                reason=None if check["matches"] else "graph_mismatch", latency_seconds=time.monotonic() - start,
                peak_memory_bytes=torch.cuda.max_memory_allocated())
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return GenerationResult(status="oom", raw_draws=1, nfe=nfe, kernel_id=kernel_id,
                reason="Batch=1 OOM; no internal oversampling/retry", latency_seconds=time.monotonic() - start,
                peak_memory_bytes=torch.cuda.max_memory_allocated())
        except Exception as exc:
            return GenerationResult(status="failed", raw_draws=1, nfe=nfe, kernel_id=kernel_id,
                reason=f"{type(exc).__name__}: {exc}", latency_seconds=time.monotonic() - start,
                peak_memory_bytes=torch.cuda.max_memory_allocated())
        finally:
            handle.remove()

    def nfe_upper_bound(self, steps: int) -> int:
        # Euler forward(s) per step + final corrector; matches inspected _generate_ode.
        return (steps + 1) * max(1, getattr(self.model, "sc_recycle_steps", 1))

    def health(self):
        return {"model_identity": self.checkpoint_identity, "capabilities": self.capabilities(),
                "nfe_per_step_upper_bound": max(1, getattr(self.model, "sc_recycle_steps", 1)),
                "device": "cuda:0", "device_uuid": str(self.torch.cuda.get_device_properties(0).uuid),
                "peak_memory_bytes": self.torch.cuda.max_memory_allocated()}
