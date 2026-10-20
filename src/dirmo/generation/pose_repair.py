"""Batched pose-only FLOWR sampling with separate graph and coordinate masks.

All reference SDFs here are the corrupted input poses. The evaluator's native
ligand is never read by this backend. Upstream remains pinned and unmodified.
"""
from __future__ import annotations

import copy
from pathlib import Path
import time
import types

import numpy as np

from ..chemistry.graphs import graph_check, map_indices, read_sdf, stereo_from_coordinates, write_sdf
from ..io import sha256_file, write_json
from .flowr import CapabilityError, FlowrBackend, prepare_graph_prior


def pose_times(times, valid, fixed):
    """Every discrete feature is known; only held coordinates have time one."""
    batch, atoms = valid.shape
    continuous = times[0].reshape(batch, 1).expand(batch, atoms).masked_fill(fixed, 1.)
    discrete = times[1].reshape(batch, 1).expand(batch, atoms).masked_fill(valid.bool(), 1.)
    return [continuous.unsqueeze(-1), discrete.unsqueeze(-1)]


def clamp_pose(data, prediction, fixed, feature_keys, overwrite_with_zeros=False):
    """Freeze the whole graph and only selected XYZ; preserve free predictions."""
    for key in feature_keys:
        if key == "coords" or key not in prediction or key not in data:
            continue
        prediction[key] = (data[key] * 0 if overwrite_with_zeros else data[key].clone())
    if "coords" in prediction:
        import torch
        prediction["coords"] = torch.where(fixed.unsqueeze(-1), data["coords"], prediction["coords"])
    for key in ("fragment_mask", "fragment_mode", "mask"):
        prediction[key] = data[key]
    return prediction


class PoseRepairBackend:
    def __init__(self, config, *, allow_whole_pose=False):
        if config.inpaint_mode != "graph":
            raise CapabilityError("Pose-only repair requires the whole graph mode")
        self.backend = FlowrBackend(config)
        self.model, self.torch = self.backend.model, self.backend.torch
        self.config = config
        self.allow_whole_pose = allow_whole_pose
        self.active_fixed = None
        self.constraint_calls = 0
        self.largest_held_drift = 0.
        self.last_forward_calls = 0
        owner = self

        def get_times(_model, times, lig_mask, inpaint_mask):
            fixed = owner.active_fixed.to(lig_mask.device)
            return pose_times(times, lig_mask, fixed)

        def inpaint_graph(data, prediction, pocket_mask=None, keep_interactions=False,
                          feature_keys=None, overwrite_with_zeros=False):
            keys = feature_keys if feature_keys is not None else owner.model.feature_keys
            fixed = owner.active_fixed.to(data["coords"].device)
            out = clamp_pose(data, prediction, fixed, keys, overwrite_with_zeros)
            if "coords" in out and bool(fixed.any()):
                drift = float((out["coords"][fixed] - data["coords"][fixed]).abs().max().item())
                owner.largest_held_drift = max(owner.largest_held_drift, drift)
            owner.constraint_calls += 1
            return out

        self.model._get_times = types.MethodType(get_times, self.model)
        self.model.builder.inpaint_graph = inpaint_graph
        self.identity = {**self.backend.kernel_base,
                         "graph_condition_support": "training provenance unknown when absent from checkpoint",
                         "coordinate_condition": "independent held coordinate mask",
                         "initialization": "exact corrupted reference XYZ, no native free coordinates",
                         "backend_sha256": sha256_file(__file__)}

    def generate(self, items, steps=64):
        """One true batch with a shared target and per-input coordinate masks."""
        from flowr.data.dataset import GeometricDataset
        from flowr.gen import utils as util
        from flowr.gen.generate import generate_ligands_per_target
        from flowr.util.pocket import PocketComplexBatch

        torch = self.torch
        batch_size = len(items)
        if not items or batch_size > 64:
            raise ValueError("Pose batch must contain 1..64 samples")
        if len({(x["pocket_path"],tuple(x["free_maps"])) for x in items})!=1:
            raise CapabilityError("Mixed-size targets are disabled: padding-dependent sampling has not been validated")
        start = time.monotonic()
        self.last_forward_calls = 0
        args = copy.copy(self.backend.args)
        args.batch_cost, args.integration_steps = batch_size, steps
        args.graph_inpainting, args.seed = "random", items[0]["seed"]
        self.model.integrator.steps = steps
        systems, expected = [], []
        for item in items:
            args.pdb_file, args.ligand_file = item["pocket_path"], item["input_sdf"]
            systems.append(util.load_data_from_pdb(args, remove_hs=self.backend.hparams["remove_hs"],
                                                  remove_aromaticity=self.backend.hparams["remove_aromaticity"]))
            expected.append(read_sdf(item["input_sdf"]))
        if len({graph_check(expected[0],mol)["actual_smiles"] for mol in expected})!=1:
            raise ValueError("Pose batch input graphs differ")
        transform, interpolant = util.load_util(args, self.backend.hparams, *self.backend.vocabs[:4])
        # Seed-independent XYZ from each explicitly saved corrupted input. This
        # bypasses fragment fusion/mask selection and its reference-prior leakage.
        interpolant = prepare_graph_prior(interpolant, "reference", 0.)
        dataset = GeometricDataset(PocketComplexBatch(systems), data_cls=PocketComplexBatch, transform=transform)
        prior, posterior, _, _ = next(iter(util.get_dataloader(args, dataset, interpolant)))
        if len(prior["complex"]) != batch_size:
            raise RuntimeError("Data loader did not produce the requested batch")
        references = self.model._generate_ligs(posterior, lig_mask=posterior["lig_mask"].bool(), scale=1.)
        for mol, reference in zip(expected, references):
            check = graph_check(mol, reference, derive_stereo=True)
            if not check["matches"] or any(int(i) != j for i, j in check.get("expected_index_to_actual_index", {}).items()):
                raise CapabilityError("Tensorization changed the graph or atom order")
        ligand = self.model.builder.extract_ligand_from_complex(prior)
        absolute = self.model.builder.undo_zero_com_batch(ligand["coords"], ligand["mask"],
                                                         [system.com for system in prior["complex"]])
        fixed = torch.zeros_like(ligand["mask"], dtype=torch.bool)
        for index, (item, mol) in enumerate(zip(items, expected)):
            free = set(item["free_maps"])
            atom_maps = [a.GetAtomMapNum() for a in mol.GetAtoms()]
            if not free or not free <= set(atom_maps) or (free == set(atom_maps) and not self.allow_whole_pose):
                raise ValueError("Require both held and free atoms")
            valid = ligand["mask"][index].bool()
            if int(valid.sum()) != len(atom_maps) or not bool(valid[:len(atom_maps)].all()):
                raise CapabilityError("Unexpected ligand padding or atom count")
            positions = absolute[index, valid].detach().cpu().numpy()
            if not np.allclose(positions, mol.GetConformer().GetPositions(), atol=.001, rtol=0):
                raise CapabilityError("Sampler prior differs from the recorded corrupted input")
            fixed[index, :len(atom_maps)] = torch.tensor([m not in free for m in atom_maps])
            directory = Path(item["output_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory / "condition.json", {"free_atom_maps": sorted(free),
                       "coordinate_fixed_mask": fixed[index].tolist(),
                       "graph_fixed_mask": valid.tolist(), "prior_xyz": positions.tolist(),
                       "input_sdf": item["input_sdf"], "input_sha256": sha256_file(item["input_sdf"]),
                       "seed": item["seed"], "native_pose_read_by_model": False})
        self.active_fixed = fixed
        self.constraint_calls, self.largest_held_drift = 0, 0.
        nfe = 0

        def count_forward(_module, _args):
            nonlocal nfe
            nfe += 1
            self.last_forward_calls = nfe

        handle = self.model.register_forward_pre_hook(count_forward)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        inference_start = time.monotonic()
        try:
            with torch.inference_mode():
                molecules = generate_ligands_per_target(args,self.model,prior,posterior,device="cuda:0")
            torch.cuda.synchronize()
            elapsed = time.monotonic() - inference_start
            if len(molecules) != batch_size:
                raise RuntimeError("Batch output slots were lost")
            stats = {"status": "complete", "batch_size": batch_size, "steps": steps,
                     "forward_calls": nfe, "sample_nfe": nfe * batch_size,
                     "inference_seconds": elapsed, "samples_per_second": batch_size / elapsed,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                     "total_seconds": time.monotonic() - start, "constraint_calls": self.constraint_calls,
                     "held_tensor_max_drift": self.largest_held_drift,
                     "case_ids": [item.get("case_id") for item in items],
                     "heterogeneous_batch": len({item["pocket_path"] for item in items}) > 1,
                     "actual_gpu_uuid": str(torch.cuda.get_device_properties(0).uuid)}
            results = []
            for item, source, product in zip(items, expected, molecules):
                directory = Path(item["output_dir"])
                result = {"status": "build_failed", "sample": item, "batch_stats": stats,
                          "kernel": {**self.identity, "batch": batch_size, "steps": steps}}
                if product is not None:
                    product = stereo_from_coordinates(product)
                    output = directory / "raw.sdf"
                    write_sdf(output, product)
                    loaded = read_sdf(output)
                    check = graph_check(source, loaded, derive_stereo=True)
                    index = map_indices(source)
                    held = [i for m, i in index.items() if m not in item["free_maps"]]
                    if source.GetNumAtoms() != loaded.GetNumAtoms():
                        raise RuntimeError("Fixed graph output atom count changed")
                    drift = float(np.linalg.norm(loaded.GetConformer().GetPositions()[held] -
                                                 source.GetConformer().GetPositions()[held], axis=1).max()) if held else 0.
                    result.update(status="ok" if check["matches"] and drift <= .001 else "contract_failed",
                                  sdf_path=str(output), graph_check=check, held_max_drift=drift)
                write_json(directory / "result.json", result)
                results.append(result)
            return results, stats
        finally:
            handle.remove()
            self.active_fixed = None
