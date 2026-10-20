"""Raw pocket-only FLOWR batches, with no reference graph conditioning."""
from __future__ import annotations

import copy
from pathlib import Path
import time

from rdkit import Chem

from ..chemistry.graphs import mapped, stereo_from_coordinates, write_sdf
from ..io import digest, write_json
from .flowr import FlowrBackend


class DeNovoBackend:
    def __init__(self, config):
        self.backend = FlowrBackend(config, de_novo=True)
        self.model = self.backend.model
        self.torch = self.backend.torch

    def generate(self, case, indices, directory, steps=64, seed=20261015):
        from flowr.data.dataset import GeometricDataset
        from flowr.gen import utils as util
        from flowr.gen.generate import generate_ligands_per_target
        from flowr.util.pocket import PocketComplexBatch

        args = copy.copy(self.backend.args)
        args.pdb_file, args.ligand_file = case['pocket_path'], case['native_sdf']
        args.batch_cost, args.integration_steps, args.seed = len(indices), steps, seed
        args.graph_inpainting, args.final_inpaint = None, False
        args.sample_mol_sizes = False
        self.model.integrator.steps = steps
        assert not self.model.graph_inpainting and not self.model.inpainting_mode
        system = util.load_data_from_pdb(args, remove_hs=self.backend.hparams['remove_hs'],
                                        remove_aromaticity=self.backend.hparams['remove_aromaticity'])
        transform, interpolant = util.load_util(args, self.backend.hparams, *self.backend.vocabs[:4])
        dataset = GeometricDataset(PocketComplexBatch([copy.deepcopy(system) for _ in indices]),
                                   data_cls=PocketComplexBatch, transform=transform)
        prior, posterior, _, _ = next(iter(util.get_dataloader(args, dataset, interpolant)))
        assert len(prior['complex']) == len(indices)
        ligand = self.model.builder.extract_ligand_from_complex(prior)
        fragment = prior.get('fragment_mask')
        if fragment is not None and bool(fragment.any()):
            raise RuntimeError('Unexpected known ligand atoms in de novo prior')
        prior_hashes = [digest(ligand['coords'][i].tolist()) for i in range(len(indices))]
        if len(indices) > 1 and len(set(prior_hashes)) != len(indices):
            raise RuntimeError('De novo samples must have distinct random priors')
        count = 0

        def forward(_module, _args):
            nonlocal count
            count += 1

        handle = self.model.register_forward_pre_hook(forward)
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        try:
            with self.torch.inference_mode():
                molecules = generate_ligands_per_target(args, self.model, prior, posterior, device='cuda:0')
            self.torch.cuda.synchronize()
        finally:
            handle.remove()
        if len(molecules) != len(indices):
            raise RuntimeError('Upstream dropped output slots')
        stats = {'batch_size': len(indices), 'steps': steps, 'forward_calls': count,
                 'sample_nfe': count * len(indices), 'seconds': time.monotonic() - start,
                 'peak_allocated_bytes': self.torch.cuda.max_memory_allocated(),
                 'conditioning': 'pocket only; reference supplies atom count and binding-site centering, no graph or coordinate clamping',
                 'same_reference_atom_count': True, 'final_inpaint': False}
        for i, product, prior_hash in zip(indices, molecules, prior_hashes):
            path = Path(directory) / f'{i:03d}'
            result = {'status': 'build_failed', 'draw': i, 'batch': stats,
                      'batch_seed': seed, 'prior_xyz_sha256': prior_hash,
                      'kernel': {**self.backend.kernel_base, 'batch': len(indices), 'integration_steps': steps,
                                 'graph_condition_support': 'not applicable: de novo generation'}}
            if product is not None:
                # Preserve upstream raw output even if it cannot enter the connected-graph repair task.
                write_sdf(path / 'raw.sdf', product)
                result.update(status='built', raw_sdf=str(path / 'raw.sdf'))
                try:
                    clean = mapped(stereo_from_coordinates(Chem.RemoveHs(product)))
                    write_sdf(path / 'repair_input.sdf', clean)
                    result['repair_input_sdf'] = str(path / 'repair_input.sdf')
                except Exception as exc:
                    result['repair_input_error'] = f'{type(exc).__name__}: {exc}'
            write_json(path / 'result.json', result)
        return stats
