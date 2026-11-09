"""Train-source MMP -> exact-graph FM -> pocket UFF contract pilot.

This is exhaustive menu outcome collection, NOT a trained policy or closed loop.
No LLM, optimizer, test-source structures, or historical outcome labels are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA
from dirmo.chemistry.graphs import (ChemicalError, execute_edit, graph_id, mapped,
    map_indices, protected_graph, read_sdf, smiles, write_sdf)
from dirmo.evaluation.edited_graph import (anchor_indices, graph_from_record,
    graph_only_conformer, graph_record, unsupported_tensor_features)
from dirmo.io import digest, sha256_file
from dirmo.schemas import EditCandidate, Protection

CHECKPOINT = '/LocalData1/uslee/checkpoints/FLOWR_root/flowr_root_base_spindr.ckpt'
CHECKSUM = '78ef9ebe4459df1ce2668685785602a37288594db83d2bb5d970c8bc06e1ce9d'


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def log(event, **fields):
    print(json.dumps(dict(time_kst=now(), event=event, **fields), ensure_ascii=False), flush=True)


def split_pair(name):
    protein, ligand = name.split('.pdb_', 1)
    return protein + '.pdb', ligand


def pdb_ids(name):
    p, l = split_pair(name)
    return {Path(p).name.split('_')[0].lower(), Path(l).name.split('_')[3].lower()}


def rule_index(path):
    index = defaultdict(list)
    with open(path) as stream:
        for rowno, row in enumerate(csv.DictReader(stream), 2):
            if row['consistent'] != 'True' or int(float(row['count'])) < 10:
                continue
            left, right = row['transform'].split('>>')
            for a, b, direction in ((left, right, 1), (right, left, -1)):
                index[a].append(dict(rhs=b, csv_line=rowno, direction=direction,
                    count=int(float(row['count'])), agree_rate=float(row['agree_rate']),
                    aggregate_delta_pchembl=direction * float(row['median_delta'])))
    return index


def fragment_key(mol):
    mol = Chem.Mol(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(1 if atom.GetAtomicNum() == 0 else 0)
    return Chem.MolToSmiles(mol).replace('[*:1]', '*')


def join(rest, rhs, fresh):
    right = Chem.MolFromSmiles(rhs)
    if right is None:
        raise ValueError('Invalid RHS')
    for atom in right.GetAtoms():
        atom.SetAtomMapNum(0 if atom.GetAtomicNum() == 0 else fresh)
        if atom.GetAtomicNum():
            fresh += 1
    rw = Chem.RWMol(Chem.CombineMols(rest, right))
    dummies = [a.GetIdx() for a in rw.GetAtoms() if not a.GetAtomicNum()]
    if len(dummies) != 2 or any(rw.GetAtomWithIdx(i).GetDegree() != 1 for i in dummies):
        raise ValueError('Only single-attachment substitutions supported')
    ends = [rw.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() for i in dummies]
    rw.AddBond(*ends, Chem.BondType.SINGLE)
    for i in sorted(dummies, reverse=True):
        rw.RemoveAtom(i)
    mol = rw.GetMol()
    mol.RemoveAllConformers()
    Chem.SanitizeMol(mol)
    return mol


def validate_edit(parent, product):
    guess = sorted(map_indices(product))
    for _ in range(2):
        try:
            return execute_edit(parent, EditCandidate(candidate_id='mmp',
                parent_graph_id=graph_id(parent), changed_atom_maps=guess,
                mapped_product_smiles=Chem.MolToSmiles(product),
                protection_explanation='train-source contract pilot'), parent, Protection())
        except ChemicalError as exc:
            match = re.search(r'measured=\[([0-9,\s]*)\]', str(exc))
            if not match:
                raise
            guess = [int(x) for x in match[1].split(',') if x.strip()]
    raise ValueError('Edit validation retry exhausted')


def menus(parent, index, seed):
    """At most three small terminal sites, two hash-ordered products per site.

    Selection never reads pChEMBL direction or generated outcomes. It is a
    deterministic finite menu, not a random policy with known propensity.
    """
    parent_maps = set(map_indices(parent))
    proposals, seen = [], set()
    for _, chains in rdMMPA.FragmentMol(parent, maxCuts=1, resultsAsMols=True):
        if chains is None:
            continue
        pieces = Chem.GetMolFrags(chains, asMols=True, sanitizeFrags=False)
        if len(pieces) != 2:
            continue
        for rest, part in (pieces, pieces[::-1]):
            removed = sorted(a.GetAtomMapNum() for a in part.GetAtoms() if a.GetAtomicNum())
            # Pilot scope: small graph changes with a substantial retained core.
            if not 1 <= len(removed) <= min(8, parent.GetNumAtoms() // 3):
                continue
            dummy = next(a for a in rest.GetAtoms() if not a.GetAtomicNum())
            attachment = dummy.GetNeighbors()[0].GetAtomMapNum()
            lhs = fragment_key(part)
            for rule in index.get(lhs, []):
                key = (tuple(removed), rule['rhs'])
                if key in seen:
                    continue
                seen.add(key)
                proposals.append((digest([seed, list(removed), rule['rhs']]), rest, removed, attachment, lhs, rule))
    proposals.sort(key=lambda x: x[0])
    chosen, audit, sites = [], [], Counter()
    mandatory = set(parent_maps)
    for rank, rest, removed, attachment, lhs, rule in proposals:
        site = tuple(removed)
        item = dict(menu_key=rank, site_parent_maps=removed, lhs=lhs, **rule)
        if sites[site] >= 2 or (site not in sites and len(sites) >= 3):
            item['status'] = 'excluded_site_quota'
            audit.append(item)
            continue
        try:
            product, change = validate_edit(parent, join(rest, rule['rhs'], max(parent_maps) + 1))
            if not 6 <= product.GetNumAtoms() <= 64 or unsupported_tensor_features(product):
                raise ValueError('Unsupported tensor features/atom count')
            added = set(map_indices(product)) - parent_maps
            if not added or len(added) > 10:
                raise ValueError('Require 1..10 replacement heavy atoms')
            if any(x['graph']['graph_id'] == graph_id(product) for x in chosen):
                raise ValueError('Duplicate product graph')
            # Strict holds the retained fragment; relaxed additionally frees its attachment atom.
            relaxed = added | {attachment}
            remaining = mandatory - set(removed) - {attachment}
            if len(remaining) < 3:
                raise ValueError('Would erase common mandatory core')
            lookup = map_indices(product)
            free_indices = [lookup[x] for x in added]
            anchor_indices(product, [lookup[x] for x in relaxed],
                next(i for m, i in lookup.items() if m not in relaxed))
            record = graph_record(f'edit{len(chosen):02d}', product, None, free_indices)
            chosen.append(dict(action_kind='CHEMICAL_EDIT', graph=record,
                site_parent_maps=removed, attachment_map=attachment,
                strict_free_maps=sorted(added), relaxed_free_maps=sorted(relaxed),
                edit_validation=change, mmp=item.copy()))
            mandatory = remaining
            sites[site] += 1
            item['status'] = 'selected'
        except Exception as exc:
            item['status'] = 'excluded_contract'
            item['reason'] = f'{type(exc).__name__}: {exc}'
        audit.append(item)
    if not chosen:
        reasons = Counter(x.get('reason', x['status']) for x in audit)
        raise ValueError(f'No supported MMP edits: proposals={len(proposals)}, reasons={reasons.most_common(3)}')
    for action in chosen:
        protected_graph(parent, graph_from_record(action['graph']), Protection(atom_maps=sorted(mandatory)))
    # Same first-site masks for two no-chemical-edit pose controls.
    first = chosen[0]
    controls = dict(action_kind='POSE_RESAMPLE', graph=graph_record('pose_control', parent, None, []),
        site_parent_maps=first['site_parent_maps'], attachment_map=first['attachment_map'],
        strict_free_maps=first['site_parent_maps'],
        relaxed_free_maps=sorted(set(first['site_parent_maps']) | {first['attachment_map']}))
    return chosen + [controls], sorted(mandatory), audit


def prepare(args):
    from dirmo.evaluation.pocket import prepare_fixed_pocket, protein_atoms
    root = args.output
    if (root / 'study.json').exists():
        raise FileExistsError('Frozen study already exists; run it without preparing again')
    processed = Path('/LocalData1/uslee/datasets/processed_crossdock_noH_full_temp')
    raw = Path('/LocalData1/uslee/datasets/crossdocked_pocket10')
    def names(split):
        with np.load(processed / f'{split}.npz') as data:
            return data['names'].tolist()
    train, test, val = names('train'), names('test'), names('val')
    forbidden = {p for n in test for p in pdb_ids(n)}
    assert not set(train) & set(test)
    test_graphs = set()
    for name in test:
        test_graphs.add(graph_id(read_sdf(raw / split_pair(name)[1])))
    index = rule_index(args.rules)
    log('rules_indexed', fragments=len(index))
    selected, excluded, targets, parents = [], [], set(), set()
    order = np.random.default_rng(args.seed).permutation(len(train)).tolist()
    for source_index in order:
        name = train[source_index]
        protein, ligand = split_pair(name)
        target = Path(protein).parent.name
        if target in targets:
            continue
        cid = f'cdtrain-{source_index:06d}-{digest(name)[:10]}'
        directory = root / 'cases' / cid
        try:
            if pdb_ids(name) & forbidden:
                raise ValueError('Test source PDB overlap')
            native = mapped(read_sdf(raw / ligand))
            gid = graph_id(native)
            if gid in parents or gid in test_graphs:
                raise ValueError('Duplicate parent or exact test ligand graph')
            if not 12 <= native.GetNumAtoms() <= 64 or unsupported_tensor_features(native):
                raise ValueError('Unsupported parent graph size/features')
            log('preparing_case', case_id=cid, heavy_atoms=native.GetNumAtoms())
            actions, mandatory, menu_audit = menus(native, index, args.seed)
            write_sdf(directory / 'initial.sdf', native)
            pocket = directory / 'pocket.pdb'
            prepare_fixed_pocket(str(raw / protein), str(directory / 'initial.sdf'), str(pocket), 6.)
            pos, elements, _ = protein_atoms(str(pocket))
            if len(pos) > 1000 or set(elements) - {'C', 'N', 'O', 'P', 'S'}:
                raise ValueError('Unsupported pocket size/elements')
            save(directory / 'menu_audit.json', menu_audit)
            parent_xyz = native.GetConformer().GetPositions()
            parent_lookup = map_indices(native)
            initialized = []
            for ai, action in enumerate(actions):
                graph = graph_from_record(action['graph'])
                lookup = map_indices(graph)
                held_xyz = {i: parent_xyz[parent_lookup[m]].tolist() for m, i in lookup.items()
                            if m not in action['strict_free_maps']}
                first = next(i for i in held_xyz)
                anchors = anchor_indices(graph, [lookup[m] for m in action['strict_free_maps']], first)
                priors = []
                for draw in range(args.draws):
                    seed = args.seed + len(selected) * 10000 + ai * 100 + draw
                    path = directory / 'inputs' / action['graph']['name'] / f'{draw:02d}.sdf'
                    prior = dict(draw=draw, seed=seed)
                    try:
                        mol = graph_only_conformer(graph, held_xyz, anchors, seed)
                        write_sdf(path, mol)
                        prior.update(status='ready', sdf=str(path), sha256=sha256_file(path))
                    except Exception as exc:
                        prior.update(status='initialization_failed', reason=f'{type(exc).__name__}: {exc}')
                    priors.append(prior)
                action['priors'] = priors
                initialized.append(action)
            case = dict(case_id=cid, index=len(selected), source_name=name, source_index=source_index,
                split='train', target_id=target, parent_graph_id=gid,
                initial_sdf=str(directory / 'initial.sdf'), pocket_path=str(pocket),
                source_protein=str(raw / protein), source_ligand=str(raw / ligand),
                source_hashes=dict(protein=sha256_file(raw / protein), ligand=sha256_file(raw / ligand)),
                mandatory_protected_maps=mandatory, actions=initialized,
                uff_only_control=(len(selected) % 5 == 0),
                box_center=parent_xyz.mean(axis=0).tolist(), box_size=[40., 40., 40.])
            save(directory / 'case.json', case)
            selected.append(case)
            targets.add(target)
            parents.add(gid)
            save(root / 'preparation_status.json', dict(time_kst=now(), cases=len(selected), requested=args.cases))
            log('prepared_case', case_id=cid, count=len(selected), actions=2 * len(actions))
            if len(selected) == args.cases:
                break
        except Exception as exc:
            excluded.append(dict(source_name=name, reason=f'{type(exc).__name__}: {exc}'))
            log('excluded_case', case_id=cid, reason=f'{type(exc).__name__}: {exc}')
            save(root / 'eligibility_exclusions.json', excluded)
            if len(excluded) >= 500 and not selected:
                raise RuntimeError('500 consecutive preparation failures; inspect exclusions')
    save(root / 'eligibility_exclusions.json', excluded)
    if len(selected) != args.cases:
        raise ValueError(f'Only {len(selected)} eligible cases')
    plan = dict(created_at_kst=now(), cases=[c['case_id'] for c in selected], source_split='train',
        physical_gpu=args.gpu, steps=args.steps, draws_per_action=args.draws,
        planned_fm_draws=sum(2 * len(c['actions']) * args.draws for c in selected),
        selection_seed=args.seed, checkpoint=CHECKPOINT, checkpoint_sha256=CHECKSUM,
        collector_sha256=sha256_file(__file__), rules=str(args.rules), rules_sha256=sha256_file(args.rules),
        source_train_names_sha256=digest(train), source_test_names_sha256=digest(test),
        test_source_or_pdb_overlap=0, exact_test_parent_graph_overlap=0,
        existing_val_overlaps_train=len(set(val) & set(train)),
        cohort='50-root protocol-development pilot; all train, no held-out ML claim',
        selection='Seed-shuffled train; unique target directory and parent graph, small supported MMP substitutions; no measured-outcome filtering',
        menu='At most 3 sites x 2 deterministic hash-ordered edits; both MMP directions, count>=10; exhaustive selected menu',
        mmp_limit='Aggregated external vocabulary, not assay-matched activity labels; source-pair/held-out evidence leakage NOT yet audited',
        modes=['strict', 'relaxed'],
        condition='Entire target graph clamped; strict holds retained fragment XYZ; relaxed frees attachment atom too; same mandatory core for every action',
        initialization='Independent seeded graph-only ETKDG, held-anchor fit, retained XYZ overwrite; SAME prior in both modes',
        uff='Fixed supplied 6A protein pocket, all pocket atoms fixed, held ligand atoms fixed, UFF500 with interfragment interactions; no pocket recutting',
        evaluation='PB dock20, exact graph+stereo, held drift<=0.001A; Vina score_only, same fixed receptor/40A box',
        uff_only='Every fifth root, same three priors and masks, first MMP edit and pose control',
        missing='Timeout/measurement missing is NULL, not physical failure or zero delta',
        fm_training=False, router_training=False, llm_calls=0, closed_loop=False,
        pretraining_overlap='unknown', auto_scale=False)
    plan['protocol_id'] = digest(plan)
    save(root / 'study.json', plan)
    log('prepared', cases=len(selected), fm_draws=plan['planned_fm_draws'])
    return plan


def generate(args):
    import fcntl
    from dirmo.config import FMConfig
    from dirmo.gpu import inventory, resolve_device, worker_environment
    from dirmo.generation.pose_repair import PoseRepairBackend
    plan, root = read(args.output / 'study.json'), args.output
    if sha256_file(__file__) != plan['collector_sha256']:
        raise ValueError('Collector changed after freezing; use a new protocol directory')
    lock = (root / 'generation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    device = resolve_device(plan['physical_gpu'], inventory())
    os.environ.update(worker_environment(device, str(ROOT / 'src')))
    state = dict(status='loading', time_kst=now(), pid=os.getpid(), physical_gpu=plan['physical_gpu'],
        gpu_uuid=device['uuid'], planned_draws=plan['planned_fm_draws'], completed_draws=0)
    save(root / 'generation_status.json', state)
    backend = PoseRepairBackend(FMConfig(gpu=plan['physical_gpu'], port=8878, python=sys.executable,
        repo_path='/home/uslee/code/flowr_root', checkpoint_path=CHECKPOINT, checkpoint_sha256=CHECKSUM))
    save(root / 'kernel.json', backend.identity)
    consecutive_errors = 0
    try:
        for cid in plan['cases']:
            case = read(root / 'cases' / cid / 'case.json')
            for action in case['actions']:
                for mode in plan['modes']:
                    items = []
                    for prior in action['priors']:
                        directory = root / 'draws' / cid / action['graph']['name'] / mode / f"{prior['draw']:02d}"
                        if (directory / 'result.json').exists():
                            state['completed_draws'] += 1
                            continue
                        if prior['status'] != 'ready':
                            save(directory / 'result.json', prior)
                            state['completed_draws'] += 1
                            continue
                        items.append(dict(case_id=cid, draw=prior['draw'], seed=prior['seed'],
                            pocket_path=case['pocket_path'], input_sdf=prior['sdf'],
                            output_dir=str(directory), free_maps=action[f'{mode}_free_maps']))
                    if items:
                        try:
                            results, stats = backend.generate(items, steps=plan['steps'])
                            save(Path(items[0]['output_dir']).parent / 'batch_stats.json', dict(time_kst=now(), **stats))
                            consecutive_errors = 0
                        except Exception as exc:
                            consecutive_errors += 1
                            for item in items:
                                save(Path(item['output_dir']) / 'result.json', dict(status='generation_error',
                                    time_kst=now(), sample=item, reason=f'{type(exc).__name__}: {exc}'))
                            log('generation_error', case=cid, mode=mode, reason=str(exc))
                        state['completed_draws'] += len(items)
                    state.update(status='running', time_kst=now(), current_case=cid,
                        current_action=action['graph']['name'], current_mode=mode)
                    save(root / 'generation_status.json', state)
                    log('generation_progress', completed=state['completed_draws'], total=state['planned_draws'], case=cid)
                    if consecutive_errors >= 4:
                        raise RuntimeError('Four consecutive generation batch errors; diagnostic stop')
            save(root / 'cases' / cid / 'generation_done.json', dict(time_kst=now()))
        state.update(status='complete', time_kst=now())
        save(root / 'generation_status.json', state)
    except BaseException as exc:
        state.update(status='failed', time_kst=now(), reason=f'{type(exc).__name__}: {exc}')
        save(root / 'generation_status.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'generate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, default=50)
    parser.add_argument('--draws', type=int, default=3)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--rules', type=Path, default=Path('/home/uslee/code/dataset_analyze/mmp_local_rules.csv'))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog('rdApp.warning')
    if args.command == 'prepare':
        import faulthandler
        faulthandler.dump_traceback_later(120, repeat=True)
        prepare(args)
    else:
        generate(args)


if __name__ == '__main__':
    main()
