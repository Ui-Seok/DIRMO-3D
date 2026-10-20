from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

from ..chemistry.graphs import graph_check, read_sdf, smiles, write_sdf
from ..gpu import assert_worker_mask
from ..io import digest, environment, revision, sha256_file, write_json
from ..schemas import GenerationRequest, GenerationResult, kernel_identity


class CapabilityError(RuntimeError):
    pass


def prepare_graph_prior(interpolant, prior: str = "random", noise_std: float = 0.0,
                        focus_indices=(), focus_std: float = 0.0, free_indices=()):
    """Fix the whole-graph prior on this interpolant instance only.

    Two separate concerns live here.

    ``hybridization=None``: upstream's random branch omits the None
    initialization when the checkpoint carries no hybridization vocabulary,
    raising ``UnboundLocalError: inp_hybridization``. No model feature is added.

    ``prior="reference"``: the prior coordinates become the reference conformer
    DIRMO already validated, instead of a Gaussian draw. ``to_mol`` is the data
    molecule, so its coordinates are the reference pose in the model's
    complex-centred internal frame, in DIRMO's own atom order. ``zero_com()`` is
    therefore skipped for this prior: it subtracts the ligand centre while the
    upstream output transform adds the complex centre back, which would displace
    the result by (complex_com - ligand_com). ``noise_std`` restores seeded diversity:
    the reference conformer is deterministic and this checkpoint integrates coordinates
    noiselessly, so without it two seeds produce the same draw.
    """
    if prior not in ("random", "reference", "fragment"):
        raise CapabilityError(f"Unsupported DIRMO graph prior: {prior}")
    if (prior == "fragment") != bool(len(free_indices)):
        raise CapabilityError("The fragment prior needs free atom indices, and only it takes them")
    if noise_std and prior != "reference":
        raise CapabilityError("prior_noise_std only applies to the reference prior")
    sampler = interpolant.prior_sampler
    if prior == "fragment":
        # Whichever branch _build_inference_prior takes, force DIRMO's mask rather than the
        # one upstream derives from its own flags.
        import torch
        native = sampler.inpaint_molecule

        def inpaint_molecule(to_mol, from_mol, fragment_mask, fragment_mode="", **kw):
            fixed = torch.ones(to_mol.seq_length, dtype=torch.bool, device=to_mol.coords.device)
            fixed[list(free_indices)] = False
            return native(to_mol, from_mol, fixed, fragment_mode="fragment", **kw)

        sampler.inpaint_molecule = inpaint_molecule
    if prior == "random" and sampler.n_hybridization_types is not None:
        return interpolant  # upstream's own random branch is already correct
    original = sampler.inpaint_graph

    def inpaint_graph(to_mol, from_mol, mask, mode="random"):
        if mode != "random" or mask is None or not mask.any():
            return original(to_mol, from_mol, mask, mode=mode)
        import torch
        from flowr.util.molrepr import GeometricMol
        if prior == "fragment":
            # Hand the partial specification to upstream's own fragment routine: it keeps
            # the fixed atoms' coordinates, elements, charges and the bonds whose BOTH ends
            # are fixed, and leaves everything else to the prior for the model to decide.
            # That routine initialises optional features correctly, so no patch is needed.
            fixed = torch.ones(to_mol.seq_length, dtype=torch.bool, device=to_mol.coords.device)
            fixed[list(free_indices)] = False
            if not bool(fixed.any()):
                raise CapabilityError("The fragment prior needs at least one fixed atom")
            return sampler.inpaint_molecule(to_mol, from_mol, fixed, fragment_mode="fragment")
        reference = prior == "reference"
        # The random branch keeps upstream's RNG draw and centering exactly.
        if reference:
            coords = to_mol.coords.clone()
            scale = torch.full((coords.shape[0],), float(noise_std),
                               dtype=coords.dtype, device=coords.device)
            if len(focus_indices):
                # Only the flagged atoms get the larger displacement; the rest hold the
                # measured pose in place.
                scale[list(focus_indices)] = float(focus_std)
            if bool((scale > 0).any()):
                # get_dataloader seeded the global RNG from the request seed, so this
                # displacement is reproducible and differs per seed.
                coords = coords + scale[:, None] * torch.randn_like(coords)
        else:
            coords = sampler.coord_dist.sample((to_mol.seq_length, 3))
        result = GeometricMol(
            coords,
            to_mol.atomics.clone(), charges=to_mol.charges.clone(),
            hybridization=(to_mol.hybridization.clone()
                           if sampler.n_hybridization_types is not None else None),
            bond_indices=to_mol.bond_indices.clone(), bond_types=to_mol.bond_types.clone(),
            fragment_mask=torch.ones(to_mol.seq_length, dtype=torch.bool, device=to_mol.coords.device),
            fragment_mode="graph",
        )
        if reference:
            return result
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
            "priors": ["random", "reference", "fragment"], "solvers": ["euler"], "condition_changes": ["integration_steps"],
            "blockers": blockers, "checkpoint_compatibility": "unverified until real checkpoint load and probes"}


class FlowrBackend:
    def __init__(self, config, *, de_novo: bool = False):
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
                        "--no_ligand_valence_repair"]
            if de_novo:
                pass  # Pocket-only generation: neither graph nor fragment inpainting.
            elif config.inpaint_mode == "fragment":
                # inpainting_mode_inf is "fragment" only when a fragment flag is on AND
                # graph_inpainting is off (fm_pocket.py:505). Only that combination routes the
                # integration loop to builder.inpaint_molecule, the one path without the
                # full-mask gate, so passing --graph_inpainting here would silently disable
                # the partial mask.
                sys.argv.append("--fragment_inpainting")
            else:
                sys.argv.extend(["--graph_inpainting", "random"])
            self.args = get_args()
        finally:
            sys.argv = argv
        for k, v in {"ckpt_path": config.checkpoint_path, "gpus": 1, "num_workers": 0,
                     "batch_cost": 1, "max_sample_iter": 0, "dataset": "spindr", "cut_pocket": False,
                     "data_path": ".", "save_dir": ".", "filter_valid_unique": False,
                     "sample_mol_sizes": False, "max_pocket_size": config.max_pocket_atoms,
                     "integration_steps": config.parameters.integration_steps, "final_inpaint": not de_novo}.items():
            setattr(self.args, k, v)
        checkpoint_hash = sha256_file(config.checkpoint_path)
        if config.checkpoint_sha256 and checkpoint_hash != config.checkpoint_sha256:
            raise CapabilityError("Checkpoint SHA256 mismatch")
        values = load_model(self.args)
        self.model, self.hparams, *self.vocabs = values
        if self.hparams["pocket_noise"] != "fix" or self.hparams["coord_scale"] != 1.0:
            raise CapabilityError("This adapter requires a rigid pocket and coord_scale=1.0")
        if de_novo:
            if self.model.graph_inpainting or self.model.inpainting_mode:
                raise CapabilityError("De novo generation must not condition on the reference ligand graph")
        elif config.inpaint_mode == "graph":
            if not self.model.graph_inpainting or self.model.inpainting_mode:
                raise CapabilityError("Whole-graph conditioning requires graph_inpainting with fragment modes off")
            if "coords" in self.model.feature_keys:
                # Whole-graph conditioning that also fixed coordinates would echo the
                # reference instead of generating a pose. Per-atom fragment masking is a
                # different thing: it pins only the held atoms, so the guard is scoped here
                # rather than dropped.
                raise CapabilityError("Whole-graph inpainting must fix discrete features, never coordinates")
        else:
            # Derived from the model's own attributes, not from the flags this adapter passed.
            if not self.model.inpainting_mode or self.model.inpainting_mode_inf != "fragment":
                raise CapabilityError(
                    f"Fragment conditioning needs inpainting_mode on and inpainting_mode_inf "
                    f"'fragment', got {self.model.inpainting_mode}/{self.model.inpainting_mode_inf}; "
                    "upstream would silently drop a partial mask")
        if self.model.builder.ligand_valence_repair:
            raise CapabilityError("Raw evaluation requires valence repair disabled")
        self.model.to("cuda:0").eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.kernel_base = {"repo_revision": config.repo_revision, "checkpoint_sha256": checkpoint_hash,
                            "preparation": "fixed precomputed pocket; no recut; no protonation; upstream tensorization",
                            "graph_mode": "de_novo" if de_novo else config.inpaint_mode, "solver": "euler",
                            "final_inpaint": False if de_novo else "discrete atom/bond/charge features only; coordinates stay generated",
                            "valence_repair": False, "relaxation": False, "filtering": False, "batch": 1,
                            "cat_noise_euler_guard": self.args.cat_noise_euler_guard,
                            "categorical_strategy": self.args.categorical_strategy,
                            "adapter_compatibility": ["random_graph_prior_optional_hybridization_v1",
                                                      "reference_conformer_graph_prior_v1"],
                            "sc_recycle_steps": getattr(self.model, "sc_recycle_steps", 1),
                            "hyperparameters_hash": digest(dict(self.hparams)), "environment": environment()}
        # load_model REWRITES the inpainting flags from the CLI args it is handed
        # ("hparams[...] = args....") and the model is built from args, never from the
        # file, so self.hparams reports what was REQUESTED. Read the checkpoint itself to
        # record what it was actually TRAINED for. One extra deserialisation per worker.
        stored = torch.load(config.checkpoint_path, map_location="cpu",
                            weights_only=False).get("hyper_parameters", {})
        trained_modes = [flag for flag in ("interaction_conditional", "scaffold_elaboration",
                         "scaffold_hopping", "linker_inpainting", "substructure_inpainting",
                         "fragment_inpainting", "fragment_growing", "core_growing")
                         if stored.get(flag)]
        legacy = [flag for flag in ("func_group_inpainting", "scaffold_inpainting")
                  if flag in stored]
        self.kernel_base["checkpoint_inpaint_modes_trained"] = trained_modes
        self.kernel_base["checkpoint_graph_inpainting_hparam"] = "graph_inpainting" in stored
        self.kernel_base["checkpoint_legacy_inpaint_hparams"] = legacy
        # Whole-graph conditioning is a separate capability from the fragment-style modes:
        # a checkpoint can be trained for fragment inpainting and still never have seen a
        # whole fixed graph, which is the mode this adapter runs in.
        self.kernel_base["checkpoint_requested_inpaint_modes"] = sorted(
            flag for flag in ("interaction_conditional", "scaffold_elaboration", "scaffold_hopping",
                              "linker_inpainting", "substructure_inpainting", "fragment_inpainting",
                              "fragment_growing", "core_growing") if self.hparams.get(flag))
        self.kernel_base["graph_condition_support"] = (
            "trained" if "graph_inpainting" in stored
            else f"whole-graph conditioning untrained; checkpoint trained for {trained_modes or 'nothing'}; "
                 "the reference prior carries the geometry")
        self.checkpoint_identity = {"path": config.checkpoint_path, "sha256": checkpoint_hash,
                                    "repo_revision": config.repo_revision, "hyperparameters": dict(self.hparams)}
        self.pocket_tensor_hashes = {}
        self._restore_extract = None

    def capabilities(self):
        return {**audit_paths(self.config), "fixed_graph_status": "loaded_unverified_quality",
                "fixed_graph": True, "graph_mode": self.config.inpaint_mode,
                "kernel_base": self.kernel_base}

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
        # A per-atom mask is given in DIRMO atom maps, which only means something if the
        # upstream tensor order matches the reference SDF order. That is verified below by
        # the reference round-trip, before any draw is kept.
        free_indices = ()
        if request.parameters.prior_free_atom_maps:
            from ..chemistry.graphs import map_indices as _map_indices
            index = _map_indices(expected)
            missing = [m for m in request.parameters.prior_free_atom_maps if m not in index]
            if missing:
                raise CapabilityError(f"prior_free_atom_maps not on this graph: {missing}")
            free_indices = tuple(index[m] for m in request.parameters.prior_free_atom_maps)
        focus_indices = ()
        if request.parameters.prior_noise_atom_maps:
            from ..chemistry.graphs import map_indices
            index = map_indices(expected)
            missing = [m for m in request.parameters.prior_noise_atom_maps if m not in index]
            if missing:
                raise CapabilityError(f"prior_noise_atom_maps not on this graph: {missing}")
            focus_indices = tuple(index[m] for m in request.parameters.prior_noise_atom_maps)
        args = copy.copy(self.args)
        args.pdb_file, args.ligand_file = request.pocket_path, request.reference_sdf
        args.seed, args.save_dir = request.seed, str(output_dir)
        args.integration_steps = request.parameters.integration_steps
        self.model.integrator.steps = args.integration_steps
        args.solver = request.parameters.solver
        # Upstream only knows its own prior names. In graph mode both DIRMO priors enter
        # through its "random" branch, which the instance patch intercepts. In fragment mode
        # graph_inpainting must stay off or inpainting_mode_inf flips to "graph".
        args.graph_inpainting = None if self.config.inpaint_mode == "fragment" else "random"
        kernel = kernel_identity(self.kernel_base, request.parameters,
                                 sha256_file(request.pocket_path))
        kernel_id = digest(kernel)
        write_json(output_dir / "kernel.json", kernel)
        write_json(output_dir / "request.json", request)
        try:
            transform, interpolant = util.load_util(args, self.hparams, *self.vocabs[:4])
            interpolant = prepare_graph_prior(interpolant, request.parameters.prior,
                                              request.parameters.prior_noise_std,
                                              focus_indices, request.parameters.prior_noise_focus_std,
                                              free_indices)
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
            if free_indices:
                # Upstream derives its own fragment mask from its flags and ignores the
                # caller's region (measured: it fixed 4 of 22 atoms when 20 were asked for).
                # Replace it here, and overwrite the prior's held positions with the
                # reference ligand, because upstream left noise wherever ITS mask was free.
                atoms = int(posterior["lig_mask"].bool().sum().item())
                if atoms != expected.GetNumAtoms():
                    raise CapabilityError(f"Upstream tensorised {atoms} ligand atoms, "
                                          f"reference has {expected.GetNumAtoms()}")
                held = torch.ones((1, atoms), dtype=torch.bool)
                for position in free_indices:
                    held[0, position] = False
                previous = prior.get("fragment_mask")
                prior["fragment_mask"] = held.to(
                    previous.device if previous is not None else "cpu")
                prior["fragment_mode"] = ["fragment"]
                native_extract = self.model.builder.extract_ligand_from_complex
                reference_tensors = native_extract(posterior)

                def extract(batch, _native=native_extract, _reference=reference_tensors,
                            _held=held, _target=prior):
                    out = _native(batch)
                    if batch is not _target:
                        return out
                    keep = _held.to(out["coords"].device)[0]
                    for key in ("coords", "atomics", "charges", "hybridization"):
                        source, target = _reference.get(key), out.get(key)
                        if source is not None and target is not None:
                            target[:, keep] = source[:, keep].to(target.dtype).to(target.device)
                    source, target = _reference.get("bonds"), out.get("bonds")
                    if source is not None and target is not None:
                        pair = (keep.unsqueeze(0) & keep.unsqueeze(1)).unsqueeze(0).unsqueeze(-1)
                        out["bonds"] = torch.where(pair, source.to(target.device), target)
                    return out

                self.model.builder.extract_ligand_from_complex = extract
                self._restore_extract = native_extract
            # Record the mask the model will actually receive. Upstream derives it from its
            # own flags, so a DIRMO region that never reaches this tensor is a silent no-op.
            received = prior.get("fragment_mask")
            write_json(output_dir / "fragment_mask.json", {
                "requested_free_atom_maps": list(request.parameters.prior_free_atom_maps),
                "requested_free_count": len(free_indices),
                "prior_has_fragment_mask": received is not None,
                "prior_fragment_mode": prior.get("fragment_mode"),
                "mask_shape": list(received.shape) if received is not None else None,
                "mask_fixed_count": int(received.sum().item()) if received is not None else None,
                "mask_total": int(received.numel()) if received is not None else None,
                "ligand_atoms": int(posterior["lig_mask"].bool().sum().item())
                                 if "lig_mask" in posterior else None,
                "note": "fixed_count should equal ligand_atoms minus requested_free_count"})
            # Decode reference tensors only; no model forward or draw has started.
            references = self.model._generate_ligs(posterior, lig_mask=posterior["lig_mask"].bool(), scale=1.0)
            reference_check = ({} if references and references[0] is not None
                               else {"matches": False})
            if not reference_check:
                reference_check = graph_check(expected, references[0], derive_stereo=True)
            if len(references) != 1 or references[0] is None or not reference_check["matches"]:
                raise CapabilityError("Upstream graph tensorization cannot preserve this element/charge/stereo graph")
            if (focus_indices or free_indices) and not all(
                    int(k) == v for k, v in reference_check["expected_index_to_actual_index"].items()):
                raise CapabilityError("Upstream reordered the atoms, so a per-atom mask given in "
                                      "atom maps would have been applied to the wrong atoms")
        except CapabilityError:
            if getattr(self, "_restore_extract", None) is not None:
                self.model.builder.extract_ligand_from_complex = self._restore_extract
                self._restore_extract = None
            raise
        except Exception as exc:
            if getattr(self, "_restore_extract", None) is not None:
                self.model.builder.extract_ligand_from_complex = self._restore_extract
                self._restore_extract = None
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
            if free_indices:
                # In fragment mode the product graph is an output, so whole-graph equality
                # is the wrong contract. The caller validates fragment preservation; the
                # comparison against the reference graph is recorded, not enforced.
                check = {**check, "contract": "fragment preservation, not whole-graph equality",
                         "free_atom_maps": list(request.parameters.prior_free_atom_maps)}
                return GenerationResult(status="ok", sdf_path=str(raw_path), raw_draws=1, nfe=nfe,
                    kernel_id=kernel_id, graph_check=check, latency_seconds=time.monotonic() - start,
                    peak_memory_bytes=torch.cuda.max_memory_allocated())
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
            if getattr(self, "_restore_extract", None) is not None:
                self.model.builder.extract_ligand_from_complex = self._restore_extract
                self._restore_extract = None

    def nfe_upper_bound(self, steps: int) -> int:
        # Euler forward(s) per step + final corrector; matches inspected _generate_ode.
        return (steps + 1) * max(1, getattr(self.model, "sc_recycle_steps", 1))

    def health(self):
        return {"model_identity": self.checkpoint_identity, "capabilities": self.capabilities(),
                "nfe_per_step_upper_bound": max(1, getattr(self.model, "sc_recycle_steps", 1)),
                "device": "cuda:0", "device_uuid": str(self.torch.cuda.get_device_properties(0).uuid),
                "peak_memory_bytes": self.torch.cuda.max_memory_allocated()}
