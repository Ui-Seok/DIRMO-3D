"""Auditable SBDD evaluation, separate from DIRMO's ligand-optimization studies.

The fixed budget is attempts, not a quota of successful/unique/PB-valid outputs.
Reference atom count and site definition are disclosed conditioning information.
No native graph, partial coordinate mask, objective guidance, or repair is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import csv
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdFingerprintGenerator
from rdkit.Chem import rdMolTransforms
from rdkit.Contrib.SA_Score import sascorer

from ..chemistry.graphs import write_sdf
from ..io import append_jsonl, environment, sha256_file, write_json
from .sbdd_progress import collect_progress, kst_now, save_progress

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'scripts/benchmark_crossdocked_sbdd.py'
DATA = Path('/LocalData1/uslee/datasets')
FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
PROTOCOL = 'crossdocked_sbdd_raw_v2_coordinate_matched'


def read(path):
    return json.loads(Path(path).read_text())


def canonical(mol):
    mol = Chem.RemoveHs(Chem.Mol(mol))
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def load_mol(path):
    mol = next(iter(Chem.SDMolSupplier(str(path), removeHs=True)), None)
    if mol is None or not mol.GetNumAtoms() or mol.GetNumConformers() != 1:
        raise ValueError('Missing, empty, or invalid 3D molecule')
    Chem.SanitizeMol(mol)
    if not np.isfinite(mol.GetConformer().GetPositions()).all():
        raise ValueError('Non-finite coordinates')
    return mol


def cut_pocket(source, ligand, destination, cutoff):
    """Retain entire residues having a heavy atom within the specified cutoff."""
    lines = Path(source).read_text().splitlines()
    residues = defaultdict(list)
    for line in lines:
        if line.startswith(('ATOM  ', 'HETATM')):
            residues[(line[21:22], line[22:27], line[17:20])].append(line)
    xyz = ligand.GetConformer().GetPositions()
    keep = set()
    for key, atoms in residues.items():
        positions = [list(map(float, (line[30:38], line[38:46], line[46:54])))
                     for line in atoms if line[76:78].strip() not in ('H', 'D')]
        if positions and np.min(np.linalg.norm(np.asarray(positions)[:, None] - xyz[None], axis=-1)) <= cutoff:
            keep.add(key)
    selected = [line for key, atoms in residues.items() if key in keep for line in atoms]
    if not selected:
        raise ValueError('Empty pocket')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(selected) + '\nEND\n')
    return sum(line[76:78].strip() not in ('H', 'D') for line in selected)


def prepare_rigid_receptor(source, destination):
    """Use one receptor protocol for every target, without moving heavy atoms.

    Pocket files can have truncated residues. Open Babel types the supplied
    structure without Meeko template completion or sidechain reorientation.
    All original heavy atoms must appear at their original coordinates and
    no new heavy atoms may be introduced. Warnings are retained in the log.
    """
    destination.mkdir(parents=True, exist_ok=True)
    receptor = destination / 'receptor.pdbqt'
    command = [str(Path(sys.executable).parent / 'obabel'), '-ipdb', str(source), '-opdbqt',
               '-O', str(receptor), '-xr', '-xc', '-xp', '-xn', '-h', '--partialcharge', 'gasteiger']
    with (destination / 'receptor-preparation.log').open('w') as stream:
        process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
    if process.returncode or not receptor.exists():
        raise ValueError(f'Open Babel receptor preparation exited {process.returncode}')
    original = [line for line in Path(source).read_text().splitlines()
                if line.startswith(('ATOM  ', 'HETATM')) and line[76:78].strip() not in ('H', 'D')]
    output = [line for line in receptor.read_text().splitlines()
              if line.startswith(('ATOM  ', 'HETATM')) and line.split()[-1] not in ('H', 'HD', 'HS')]
    def xyz(lines):
        return np.asarray([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines])
    if not original or len(original) != len(output):
        raise ValueError('Receptor preparation changed the heavy-atom count')
    distances = np.linalg.norm(xyz(original)[:, None] - xyz(output)[None], axis=-1)
    error = max(float(distances.min(axis=0).max()), float(distances.min(axis=1).max()))
    if error > .001:
        raise ValueError(f'Receptor heavy atoms moved by {error} angstrom')
    record = dict(command=command, heavy_atoms=len(original), max_nearest_heavy_coordinate_error=error,
                  protonation='Open Babel add H, Gasteiger charges; no pH optimization, no heavy-atom completion or relaxation',
                  source_sha256=sha256_file(source), receptor_sha256=sha256_file(receptor))
    write_json(destination / 'receptor-preparation.json', record)
    return str(receptor), record


def prepare(args):
    root = args.output
    settings = dict(protocol=PROTOCOL, draws=args.draws, steps=args.steps,
                    batch_size=args.batch_size, seed=args.seed, gpu=args.gpu,
                    checkpoint=args.checkpoint, pocket_cutoff=args.pocket_cutoff,
                    docking_exhaustiveness=args.exhaustiveness)
    if (root / 'study.json').exists():
        plan = read(root / 'study.json')
        if plan['settings'] != settings:
            raise ValueError('Existing study settings differ; use its original arguments or a new output')
        return plan
    raw = DATA / 'crossdocked_pocket10'
    processed = DATA / 'processed_crossdock_noH_full_temp'
    source = DATA / 'moljo/data_zip_extract/data/test_list.tsv'
    with source.open() as stream:
        entries = list(csv.DictReader(stream, delimiter='\t'))
    names = np.load(processed / 'test.npz')['names'].tolist()
    if len(entries) != 100 or set(names) != {row['Protein'] + '_' + row['Ligand'] for row in entries}:
        raise ValueError('Local test split sources disagree')
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    for index, row in enumerate(entries):
        case_id = f'{index:03d}_' + Path(row['Ligand']).stem
        directory = root / 'cases' / case_id
        source_pocket, native = raw / row['Protein'], raw / row['Ligand']
        case = dict(case_id=case_id, index=index, native_sdf=str(native),
                    evaluation_pocket=str(source_pocket),
                    pocket_path=str(directory / 'pocket7.pdb'),
                    source_ligand=row['Ligand'], source_protein=row['Protein'])
        try:
            mol = load_mol(native)
            case['pocket_atoms'] = cut_pocket(source_pocket, mol, Path(case['pocket_path']), args.pocket_cutoff)
            case['atoms'] = mol.GetNumAtoms()
            xyz = mol.GetConformer().GetPositions()
            case.update(box_center=xyz.mean(axis=0).tolist(),
                        box_size=np.maximum(np.ptp(xyz, axis=0) + 10., 22.5).tolist(),
                        native_sha256=sha256_file(native), pocket_sha256=sha256_file(case['pocket_path']),
                        evaluation_pocket_sha256=sha256_file(source_pocket))
            try:
                receptor, audit = prepare_rigid_receptor(source_pocket, directory / 'receptor_raw_coordinates')
                case.update(receptor_path=receptor, receptor_origin='Open Babel rigid original pocket10, original heavy atoms/coordinates verified',
                            receptor_coordinate_audit=audit)
            except Exception as exc:
                case.update(receptor_path=None, receptor_error=f'{type(exc).__name__}: {exc}')
            if case.get('receptor_path'):
                case['receptor_sha256'] = sha256_file(case['receptor_path'])
        except Exception as exc:
            case['preparation_error'] = f'{type(exc).__name__}: {exc}'
        cases.append(case)
    from .fm_quality import CHECKPOINTS
    filename, checksum = CHECKPOINTS[args.checkpoint]
    plan = dict(settings=settings, cases=cases, scheduled_draws=len(cases) * args.draws,
                checkpoint_path=str(Path('/LocalData1/uslee/checkpoints/FLOWR_root') / filename),
                checkpoint_sha256=checksum, environment=environment(),
                test_list=str(source), test_list_sha256=sha256_file(source),
                corroborating_test_npz=str(processed / 'test.npz'),
                test_npz_sha256=sha256_file(processed / 'test.npz'),
                conditioning='Rigid residue-level 7A pocket; reference heavy-atom count and binding-site/complex centering; random ligand prior, no graph/atom XYZ clamps',
                generation=f'{args.draws} fixed raw attempts per test entry; no validity/uniqueness/PB filtering, no resampling failed draws, no valence repair or relaxation',
                metric_protocol={
                    'PB': 'PoseBusters dock, mol_true=None, original pocket10; no native graph/RMSD identity test',
                    'Vina': 'score at generated pose, local optimize, redock (exhaustiveness configured); identical receptor/box/protocol for reference',
                    'strain': 'MMFF94s generated-pose minus locally minimized energy, 1000 iterations; convergence recorded; NOT certified identical to paper GenBench3D',
                    'geometry': 'Unbinned empirical bond-length and angle W1; by bond order/central element and pooled; NOT identical to paper binned training-statistics-weighted W1',
                    'novelty': 'Canonical isomeric SMILES absent from local CrossDocked train_smiles; not novelty against all FLOWR pretraining data'},
                limitations=['SPINDR checkpoint on CrossDocked, not the paper CrossDocked-trained checkpoint',
                             'FLOWR pretraining overlap with these targets/ligands is unknown',
                             'Reference atom count, pocket selection and site centering are disclosed conditions',
                             'Reference poses are supplied CrossDocked poses, which may be cross-docked rather than crystal poses',
                             'Docking uses original pocket10 heavy coordinates with Open Babel H/Gasteiger typing; truncated residues are not reconstructed and physiological protonation is not guaranteed',
                             'No experimental affinity labels, pKa, solubility, xTB/QM or FEP measurement'],
                code_sha256={str(path.relative_to(ROOT)): sha256_file(path) for path in
                             (Path(__file__), SCRIPT, ROOT / 'src/dirmo/generation/de_novo.py')})
    write_json(root / 'study.json', plan)
    return plan


def generate(args):
    plan = read(args.output / 'study.json')
    settings = plan['settings']
    from ..gpu import inventory, resolve_device, worker_environment
    device = resolve_device(settings['gpu'], inventory())
    os.environ.update(worker_environment(device, str(ROOT / 'src')))
    from ..config import FMConfig
    from ..generation.de_novo import DeNovoBackend
    state = dict(status='loading', pid=os.getpid(), physical_gpu=device, completed_cases=0)
    write_json(args.output / 'generation_worker.json', state)
    config = FMConfig(gpu=settings['gpu'], port=8877, python=sys.executable,
                      repo_path='/home/uslee/code/flowr_root', checkpoint_path=plan['checkpoint_path'],
                      checkpoint_sha256=plan['checkpoint_sha256'], max_atoms=128, max_pocket_atoms=800)
    backend = DeNovoBackend(config)
    write_json(args.output / 'kernel.json', {**backend.backend.kernel_base,
               'actual_cuda_device': str(backend.torch.cuda.get_device_properties(0)),
               'sampling_args': {k: v for k, v in vars(backend.backend.args).items()
                                 if isinstance(v, (str, int, float, bool, type(None)))}})
    for case in plan['cases']:
        directory = args.output / 'cases' / case['case_id'] / 'generated'
        pending = [i for i in range(settings['draws']) if not (directory / f'{i:03d}/result.json').exists()]
        capacity = settings['batch_size']
        while pending:
            indices = pending[:capacity]
            state.update(status='running', current_case=case['case_id'], remaining_in_case=len(pending))
            write_json(args.output / 'generation_worker.json', state)
            try:
                if case.get('preparation_error'):
                    raise ValueError(case['preparation_error'])
                stats = backend.generate(case, indices, directory, settings['steps'],
                                         seed=settings['seed'] + case['index'] * 10000 + indices[0])
                append_jsonl(args.output / 'generation_batches.jsonl', dict(case_id=case['case_id'], **stats))
                print(json.dumps(dict(status='batch_complete', case_id=case['case_id'], **stats)), flush=True)
            except backend.torch.OutOfMemoryError:
                backend.torch.cuda.empty_cache()
                append_jsonl(args.output / 'generation_batches.jsonl', dict(status='oom_retry', case_id=case['case_id'], batch_size=capacity))
                if capacity > 1:
                    capacity = max(1, capacity // 2)
                    continue
                write_json(directory / f'{indices[0]:03d}/result.json', dict(status='oom', draw=indices[0]))
            except Exception as exc:
                traceback.print_exc()
                for index in indices:
                    write_json(directory / f'{index:03d}/result.json',
                               dict(status='failed', draw=index, reason=f'{type(exc).__name__}: {exc}'))
            pending = [i for i in pending if not (directory / f'{i:03d}/result.json').exists()]
            backend.torch.cuda.empty_cache()
        write_json(directory / 'complete.json', dict(case_id=case['case_id'], draws=settings['draws']))
        state['completed_cases'] += 1
    state.update(status='complete')
    write_json(args.output / 'generation_worker.json', state)


def geometry(mol):
    conf = mol.GetConformer()
    lengths, angles = defaultdict(list), defaultdict(list)
    for bond in mol.GetBonds():
        lengths[str(bond.GetBondType())].append(float(rdMolTransforms.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
    for atom in mol.GetAtoms():
        for left, right in itertools.combinations([a.GetIdx() for a in atom.GetNeighbors()], 2):
            value = float(rdMolTransforms.GetAngleDeg(conf, left, atom.GetIdx(), right))
            if not np.isfinite(value):
                raise ValueError('Undefined bond angle (coincident coordinates)')
            angles[atom.GetSymbol()].append(value)
    return {'bond_lengths': dict(lengths), 'bond_angles': dict(angles)}


def properties(mol):
    hbd, hba, mw, logp = Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol), Descriptors.MolWt(mol), Crippen.MolLogP(mol)
    return dict(qed=float(QED.qed(mol)), sa_raw_1_to_10=float(sascorer.calculateScore(mol)),
                logp=float(logp), molecular_weight=float(mw), hbd=hbd, hba=hba,
                tpsa=float(Descriptors.TPSA(mol)), rotatable_bonds=Lipinski.NumRotatableBonds(mol),
                rings=mol.GetRingInfo().NumRings(), heavy_atoms=mol.GetNumHeavyAtoms(),
                lipinski_pass=(hbd <= 5 and hba <= 10 and mw <= 500 and logp <= 5))


def strain(mol):
    work = Chem.AddHs(Chem.Mol(mol), addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(work, mmffVariant='MMFF94s')
    if props is None:
        raise ValueError('MMFF94s atom parameters unavailable')
    field = AllChem.MMFFGetMoleculeForceField(work, props)
    initial = field.CalcEnergy()
    status = field.Minimize(maxIts=1000)
    final = field.CalcEnergy()
    return dict(status='ok', strain_kcal_mol=float(initial - final), initial_energy=float(initial),
                minimized_energy=float(final), converged=status == 0, minimizer_status=int(status))


def docking(mol, case, exhaustiveness, seed):
    if not case.get('receptor_path'):
        return dict(status='missing', reason=case.get('receptor_error', 'No prepared receptor'))
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from vina import Vina
    setups = MoleculePreparation().prepare(Chem.AddHs(Chem.Mol(mol), addCoords=True))
    if len(setups) != 1:
        raise ValueError('Multiple ligand preparations')
    pdbqt, valid, error = PDBQTWriterLegacy.write_string(setups[0])
    if not valid:
        raise ValueError(error)
    scorer = Vina(sf_name='vina', cpu=1, seed=seed, verbosity=0)
    scorer.set_receptor(case['receptor_path'])
    scorer.compute_vina_maps(center=case['box_center'], box_size=case['box_size'])
    scorer.set_ligand_from_string(pdbqt)
    # Preserve each available score if a later operation fails. Redocking starts
    # afresh from the original input, independent of local optimization.
    result = dict(status='ok', seed=seed, exhaustiveness=exhaustiveness)
    for name, operation in (
            ('score', lambda: scorer.score()[0]),
            ('minimize', lambda: scorer.optimize()[0]),
            ('dock', lambda: _redock(scorer, pdbqt, exhaustiveness))):
        try:
            value = float(operation())
            if not np.isfinite(value):
                raise ValueError('Non-finite docking score')
            result[name] = value
        except Exception as exc:
            result[name + '_error'] = f'{type(exc).__name__}: {exc}'
            result['status'] = 'partial'
    return result


def _redock(scorer, pdbqt, exhaustiveness):
    scorer.set_ligand_from_string(pdbqt)
    scorer.dock(exhaustiveness=exhaustiveness, n_poses=1)
    return scorer.energies(n_poses=1)[0][0]


def _bounded_docking_unlocked(path, case, settings, draw):
    """A native library/search failure cannot strand an entire pocket evaluator."""
    root = Path(os.environ['DIRMO_SBDD_RUN'])
    folder = root / 'cases' / case['case_id']
    prefix = folder / ('reference_docking' if draw < 0 else f'generated/{draw:03d}/docking')
    destination = prefix.with_suffix('.json')
    if destination.exists():
        return read(destination)
    command = [sys.executable, '-B', str(SCRIPT), 'dock-one', '--output', str(root),
               '--case-index', str(case['index']), '--draw', str(draw), '--sdf', str(path)]
    try:
        with prefix.with_suffix('.log').open('a') as log:
            process = subprocess.run(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''},
                                     stdout=log, stderr=subprocess.STDOUT, timeout=600)
        if process.returncode or not destination.exists():
            raise RuntimeError(f'Docking child exited {process.returncode}; see {prefix.with_suffix(".log")}')
        return read(destination)
    except Exception as exc:
        result = dict(status='missing', reason=f'{type(exc).__name__}: {exc}')
        write_json(destination, result)
        return result


def bounded_docking(path, case, settings, draw):
    root = Path(os.environ['DIRMO_SBDD_RUN'])
    folder = root / 'cases' / case['case_id']
    lock = folder / ('reference_docking.lock' if draw < 0 else f'generated/{draw:03d}/docking.lock')
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        return _bounded_docking_unlocked(path, case, settings, draw)


def dock_one(args):
    plan = read(args.output / 'study.json')
    case, settings = plan['cases'][args.case_index], plan['settings']
    folder = args.output / 'cases' / case['case_id']
    destination = folder / ('reference_docking.json' if args.draw < 0 else f'generated/{args.draw:03d}/docking.json')
    try:
        result = docking(load_mol(args.sdf), case, settings['docking_exhaustiveness'],
                         settings['seed'] + case['index'] * 10000 + max(args.draw, 0))
    except Exception as exc:
        result = dict(status='missing', reason=f'{type(exc).__name__}: {exc}')
    write_json(destination, result)


def evaluate_one(path, case, settings, draw):
    start = time.monotonic()
    result = dict(case_id=case['case_id'], draw=draw, sdf=str(path), rdkit_valid=False, connected=False)
    try:
        mol = load_mol(path)
        result.update(rdkit_valid=True, connected=len(Chem.GetMolFrags(mol)) == 1, smiles=canonical(mol))
        for key, function in (('properties', properties), ('geometry', geometry)):
            try:
                result[key] = function(mol)
            except Exception as exc:
                result[key + '_error'] = f'{type(exc).__name__}: {exc}'
        native = load_mol(case['native_sdf'])
        result['tanimoto_to_reference'] = float(DataStructs.TanimotoSimilarity(FP.GetFingerprint(mol), FP.GetFingerprint(native)))
        from .pb_worker import bounded_measure
        result['pb'] = bounded_measure(path, case['evaluation_pocket'])
        for key, function in (
                ('strain', lambda: strain(mol)),
                ('vina', lambda: bounded_docking(path, case, settings, draw))):
            try:
                result[key] = function()
            except Exception as exc:
                result[key] = dict(status='missing', reason=f'{type(exc).__name__}: {exc}')
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    result['seconds'] = time.monotonic() - start
    return result


def evaluate_case(args):
    os.environ['DIRMO_SBDD_RUN'] = str(args.output)
    plan = read(args.output / 'study.json')
    case = plan['cases'][args.case_index]
    directory = args.output / 'cases' / case['case_id']
    native_path = directory / 'reference_evaluation.json'
    if not native_path.exists():
        write_json(native_path, evaluate_one(case['native_sdf'], case, plan['settings'], -1))
    for draw in range(plan['settings']['draws']):
        folder = directory / 'generated' / f'{draw:03d}'
        output = folder / 'evaluation.json'
        if output.exists():
            continue
        built = read(folder / 'result.json')
        if built.get('raw_sdf'):
            result = evaluate_one(built['raw_sdf'], case, plan['settings'], draw)
        else:
            result = dict(case_id=case['case_id'], draw=draw, rdkit_valid=False, connected=False,
                          generation_status=built['status'], generation_error=built.get('reason'))
        write_json(output, result)
        print(json.dumps(dict(case_id=case['case_id'], draw=draw, rdkit_valid=result['rdkit_valid'], seconds=result.get('seconds'))), flush=True)
    write_json(directory / 'evaluation_complete.json', dict(case_id=case['case_id']))


def summary_stats(values):
    array = np.asarray([float(v) for v in values if v is not None and np.isfinite(v)])
    if not len(array):
        return dict(n=0, mean=None, median=None, std=None)
    return dict(n=len(array), mean=float(array.mean()), median=float(np.median(array)), std=float(array.std()))


def metric_summary(rows, reference=None):
    valid = [r for r in rows if r.get('rdkit_valid')]
    pb_known = [r for r in valid if 'pb' in r and not r['pb'].get('missing') and not r['pb'].get('missing_checks')]
    pb_pass = sum(r.get('pb', {}).get('passes', False) for r in valid)
    result = dict(scheduled=len(rows), rdkit_valid=len(valid), connected=sum(r.get('connected', False) for r in valid),
                  pb_measured=len(pb_known), pb_pass=pb_pass,
                  pb_missing_on_valid=len(valid) - len(pb_known),
                  pb_pass_per_attempt=pb_pass / len(rows) if rows else None,
                  pb_pass_per_rdkit_valid=pb_pass / len(valid) if valid else None,
                  pb_pass_per_measured=pb_pass / len(pb_known) if pb_known else None,
                  unique_smiles=len({r['smiles'] for r in valid}),
                  properties={key: summary_stats([r.get('properties', {}).get(key) for r in valid])
                              for key in ('qed', 'sa_raw_1_to_10', 'logp', 'molecular_weight', 'hbd', 'hba', 'tpsa', 'rotatable_bonds', 'rings', 'heavy_atoms', 'lipinski_pass')},
                  strain=summary_stats([r.get('strain', {}).get('strain_kcal_mol') for r in valid]),
                  strain_converged=sum(r.get('strain', {}).get('converged', False) for r in valid),
                  vina={key: summary_stats([r.get('vina', {}).get(key) for r in valid]) for key in ('score', 'minimize', 'dock')},
                  tanimoto_to_reference=summary_stats([r.get('tanimoto_to_reference') for r in valid]))
    if reference:
        result['high_affinity'] = {}
        for key in ('score', 'minimize', 'dock'):
            ref = reference.get('vina', {}).get(key)
            pairs = [r['vina'][key] for r in valid if ref is not None and key in r.get('vina', {})]
            result['high_affinity'][key] = dict(n=len(pairs), reference=ref,
                                               improved=sum(value < ref for value in pairs),
                                               fraction=sum(value < ref for value in pairs) / len(pairs) if pairs else None)
        fps = [FP.GetFingerprint(Chem.MolFromSmiles(r['smiles'])) for r in valid]
        distances = [1 - DataStructs.TanimotoSimilarity(a, b) for a, b in itertools.combinations(fps, 2)]
        result['intra_pocket_diversity'] = summary_stats(distances)
    return result


def distribution_comparison(rows, references):
    from scipy.stats import wasserstein_distance
    def collect(data, key):
        gathered = defaultdict(list)
        for row in data:
            for group, values in row.get('geometry', {}).get(key, {}).items():
                gathered[group].extend(values)
        return gathered
    result = {}
    for key in ('bond_lengths', 'bond_angles'):
        generated, truth = collect(rows, key), collect(references, key)
        groups = sorted(set(generated) | set(truth))
        per_group = {group: dict(generated_n=len(generated[group]), reference_n=len(truth[group]),
                    w1=float(wasserstein_distance(generated[group], truth[group])) if generated[group] and truth[group] else None)
                     for group in groups}
        g, t = list(itertools.chain.from_iterable(generated.values())), list(itertools.chain.from_iterable(truth.values()))
        result[key] = dict(pooled_w1=float(wasserstein_distance(g, t)) if g and t else None,
                           units='angstrom' if key == 'bond_lengths' else 'degrees', by_group=per_group)
    return result


def report(args):
    plan = read(args.output / 'study.json')
    rows, references, per_case = [], [], []
    for case in plan['cases']:
        folder = args.output / 'cases' / case['case_id']
        reference = read(folder / 'reference_evaluation.json') if (folder / 'reference_evaluation.json').exists() else None
        if reference:
            references.append(reference)
        current = []
        for draw in range(plan['settings']['draws']):
            path = folder / 'generated' / f'{draw:03d}/evaluation.json'
            if path.exists():
                current.append(read(path))
        rows.extend(current)
        per_case.append(dict(case_id=case['case_id'], **metric_summary(current, reference)))
    summary = dict(status='complete' if len(rows) == plan['scheduled_draws'] and len(references) == len(plan['cases']) else 'partial',
                   scheduled_draws=plan['scheduled_draws'], evaluated_draws=len(rows), reference_count=len(references),
                   generated=metric_summary(rows), reference=metric_summary(references), per_case=per_case,
                   distributions=distribution_comparison(rows, references), limitations=plan['limitations'])
    summary['macro_average'] = {key: summary_stats([case.get(key) for case in per_case])
                              for key in ('pb_pass_per_attempt', 'pb_pass_per_rdkit_valid', 'pb_pass_per_measured')}
    summary['vina_macro_average'] = {key: summary_stats([case['vina'][key]['mean'] for case in per_case])
                                     for key in ('score', 'minimize', 'dock')}
    # This novelty reference is explicitly CrossDocked training chemistry, not
    # FLOWR's unavailable full pretraining manifest.
    training_source = DATA / 'processed_crossdock_noH_full_temp/train_smiles.npy'
    cache = args.output / 'cache/canonical_train_smiles.json'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        training = set(read(cache))
    else:
        training = set()
        for text in np.load(training_source, allow_pickle=True).reshape(-1):
            mol = Chem.MolFromSmiles(str(text))
            if mol is not None:
                training.add(canonical(mol))
        write_json(cache, sorted(training))
    valid = [r for r in rows if r.get('smiles')]
    novel = [r for r in valid if r['smiles'] not in training]
    summary['crossdocked_train_novelty'] = dict(valid_n=len(valid), novel_n=len(novel),
          fraction=len(novel) / len(valid) if valid else None, reference_unique_smiles=len(training),
          source=str(training_source), source_sha256=sha256_file(training_source))
    # Flat molecule-level data are convenient for independent re-analysis.
    with (args.output / 'molecule_metrics.csv').open('w', newline='') as stream:
        fields = ['case_id', 'draw', 'rdkit_valid', 'connected', 'pb_pass', 'pb_missing',
                  'qed', 'sa_raw_1_to_10', 'logp', 'molecular_weight', 'heavy_atoms',
                  'strain_kcal_mol', 'strain_converged', 'vina_score', 'vina_minimize', 'vina_dock', 'smiles']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in references + rows:
            writer.writerow({**{k: row.get(k) for k in ('case_id', 'draw', 'rdkit_valid', 'connected', 'smiles')},
                **{k: row.get('properties', {}).get(k) for k in ('qed', 'sa_raw_1_to_10', 'logp', 'molecular_weight', 'heavy_atoms')},
                'pb_pass': row.get('pb', {}).get('passes'),
                'pb_missing': row.get('pb', {}).get('missing', bool(row.get('pb', {}).get('missing_checks'))) if 'pb' in row else True,
                'strain_kcal_mol': row.get('strain', {}).get('strain_kcal_mol'),
                'strain_converged': row.get('strain', {}).get('converged'),
                **{'vina_' + key: row.get('vina', {}).get(key) for key in ('score', 'minimize', 'dock')}})
    write_json(args.output / 'summary.json', summary)
    (args.output / 'REPORT.md').write_text(
        '# CrossDocked2020 SBDD benchmark\n\n'
        f"Status: {summary['status']}; evaluated {len(rows)}/{plan['scheduled_draws']} draws; references {len(references)}/100.\n\n"
        f"Settings: {json.dumps(plan['settings'])}\n\n"
        'Full metric summaries and per-pocket results: [summary.json](summary.json). '
        'Each raw attempt has result.json and evaluation.json; study.json records input hashes, conditioning, and limitations.\n\n'
        + '\n'.join('- ' + item for item in plan['limitations']) + '\n')
    return summary


def run(args):
    plan = prepare(args)
    root = args.output
    with (root / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   PYTHONPATH=str(ROOT / 'src'), TMPDIR=str(root / 'tmp'),
                   DIRMO_PB_MEASURE_CACHE=str(root / 'cache/pb'),
                   DIRMO_PB_ENERGY_CACHE=str(root / 'cache/energy'))
        (root / 'tmp').mkdir(exist_ok=True)
        with (root / 'generation.log').open('a') as stream:
            process = subprocess.Popen([sys.executable, '-B', str(SCRIPT), 'generate', '--output', str(root)],
                                       env=env, stdout=stream, stderr=subprocess.STDOUT)
        jobs, submitted = {}, set()
        def evaluate(index):
            path = root / 'cases' / plan['cases'][index]['case_id']
            with (path / 'evaluation.log').open('a') as log:
                child = subprocess.run([sys.executable, '-B', str(SCRIPT), 'evaluate-case', '--output', str(root),
                                        '--case-index', str(index)], env={**env, 'CUDA_VISIBLE_DEVICES': ''},
                                       stdout=log, stderr=subprocess.STDOUT)
            if child.returncode:
                raise RuntimeError(f'Case {index} evaluator exited {child.returncode}; see {path / "evaluation.log"}')
        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while True:
                for case in plan['cases']:
                    folder = root / 'cases' / case['case_id']
                    if (folder / 'generated/complete.json').exists() and case['index'] not in submitted:
                        submitted.add(case['index'])
                        jobs[case['index']] = pool.submit(evaluate, case['index'])
                finished = sum(f.done() for f in jobs.values())
                for index, job in list(jobs.items()):
                    if job.done() and job.exception() and index not in failures:
                        failures.append(index)
                        append_jsonl(root / 'errors.jsonl', dict(case_index=index, error=str(job.exception())))
                state = dict(status='running', pid=os.getpid(), generation_pid=process.pid,
                             generation_exit=process.poll(), generated_cases=len(submitted),
                             evaluated_case_jobs=finished, failed_evaluations=failures)
                progress = collect_progress(root, plan, state)
                state.update({key: progress[key] for key in ('evaluated_cases', 'evaluated_draws',
                              'scheduled_draws', 'evaluation_percent', 'updated_kst', 'latest_evaluation_kst')})
                write_json(root / 'status.json', state)
                save_progress(root / 'progress.json', progress)
                if process.poll() is not None and finished == len(jobs):
                    break
                time.sleep(10)
        summary = report(args)
        state.update(status=summary['status'], generation_exit=process.returncode,
                     evaluated_draws=summary['evaluated_draws'], updated_kst=kst_now())
        write_json(root / 'status.json', state)
        save_progress(root / 'progress.json', collect_progress(root, plan, state))
        if process.returncode or failures:
            raise RuntimeError('A worker failed; inspect status.json and logs; rerun original run command to resume')


def main():
    os.environ['TZ'] = 'Asia/Seoul'
    if hasattr(time, 'tzset'):
        time.tzset()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'generate', 'evaluate-case', 'dock-one', 'report', 'run'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--gpu', default='5')
    parser.add_argument('--draws', type=int, default=100)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261019)
    parser.add_argument('--checkpoint', choices=('base', 'v2'), default='base')
    parser.add_argument('--pocket-cutoff', type=float, default=7.)
    parser.add_argument('--exhaustiveness', type=int, default=8)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--draw', type=int, default=-1)
    parser.add_argument('--sdf', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if min(args.draws, args.steps, args.batch_size, args.workers, args.exhaustiveness) < 1:
        parser.error('Counts must be positive')
    {'prepare': prepare, 'generate': generate, 'evaluate-case': evaluate_case, 'dock-one': dock_one,
     'report': report, 'run': run}[args.phase](args)


if __name__ == '__main__':
    main()
