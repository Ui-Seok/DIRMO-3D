"""Audit compatible pilot reuse and freeze grouped splits BEFORE new outcomes."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from router_contract_pilot import ROOT, read, save, now, digest, sha256_file, read_sdf, smiles, pdb_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(); source=a.source.resolve(); root=a.output.resolve()
    if (root/'split.json').exists():
        raise FileExistsError('Split already frozen')
    old=read(source/'study.json'); plan=read(root/'study.json')
    for key in ('collector_sha256','rules_sha256','checkpoint_sha256','steps','draws_per_action','selection_seed','modes','uff','evaluation'):
        assert old[key]==plan[key], key
    for path, sha in read(source/'execution_protocol.json')['code_hashes'].items():
        assert sha256_file(ROOT/path)==sha, path
    assert plan['cases'][:len(old['cases'])]==old['cases'], 'Cohort prefix changed'
    cases=[read(root/'cases'/cid/'case.json') for cid in plan['cases']]
    # Strip paths only after validating the actual bytes at every input path.
    for c in cases[:len(old['cases'])]:
        prev=read(source/'cases'/c['case_id']/'case.json')
        for key in ('source_name','target_id','parent_graph_id','source_hashes','mandatory_protected_maps','uff_only_control','box_center','box_size'):
            assert c[key]==prev[key], (c['case_id'],key)
        for key in ('initial_sdf','pocket_path'):
            assert sha256_file(c[key])==sha256_file(prev[key])
        assert len(c['actions'])==len(prev['actions'])
        for action, before in zip(c['actions'],prev['actions']):
            assert {k:v for k,v in action.items() if k!='priors'}=={k:v for k,v in before.items() if k!='priors'}
            assert len(action['priors'])==len(before['priors'])
            for p,q in zip(action['priors'],before['priors']):
                assert {k:v for k,v in p.items() if k!='sdf'}=={k:v for k,v in q.items() if k!='sdf'}
                if p['status']=='ready':assert sha256_file(p['sdf'])==sha256_file(q['sdf'])
    parent=list(range(len(cases))); seen={}; keys=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]]; i=parent[i]
        return i
    for i,c in enumerate(cases):
        mol=read_sdf(c['initial_sdf'])
        for atom in mol.GetAtoms():atom.SetAtomMapNum(0)
        scaffold=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol),isomericSmiles=False)
        # Exact target-family prefix is a conservative local proxy, NOT sequence clustering.
        ks=[('target',c['target_id']),('family','_'.join(c['target_id'].split('_')[:2])),('parent',c['parent_graph_id'])]
        ks += [('pdb',p) for p in pdb_ids(c['source_name'])]
        if scaffold:ks.append(('scaffold',scaffold))
        keys.append(ks)
        for k in ks:
            if k in seen:parent[find(i)]=find(seen[k])
            else:seen[k]=i
    components={}
    for i,c in enumerate(cases):components.setdefault(find(i),[]).append(c['case_id'])
    oldset=set(old['cases']); assignment={}; groups={}; novel=[]
    for n,ids in enumerate(components.values()):
        gid=f'group_{n:03d}'
        for cid in ids:groups[cid]=gid
        if oldset.intersection(ids):
            for cid in ids:assignment[cid]='train'
        else:novel.append(ids)
    np.random.default_rng(20261105).shuffle(novel)
    # Whole groups; approximate 15 roots per held-out set, never split a component.
    counts=Counter(assignment.values())
    for ids in novel:
        split='validation' if counts['validation']<=counts['test'] else 'test'
        if counts['validation']>=15 and counts['test']>=15:split='train'
        for cid in ids:assignment[cid]=split
        counts[split]+=len(ids)
    groupcounts={s:len({groups[cid] for cid in assignment if assignment[cid]==s}) for s in ('train','validation','test')}
    if min(groupcounts.values())<3:
        raise ValueError(f'Insufficient independent groups for fixed holdout: {groupcounts}; no training started')
    for key in set(k for ks in keys for k in ks):
        splits={assignment[cases[i]['case_id']] for i,ks in enumerate(keys) if key in ks}
        assert len(splits)==1, (key,splits)
    split=dict(created_at_kst=now(),assignment=assignment,groups=groups,root_counts=dict(counts),group_counts=groupcounts,
        seed=20261105,previously_inspected_roots_train_only=sorted(oldset),
        grouping='Union of target directory, target-family prefix, source PDBs, exact parent graph, exact nonempty Murcko scaffold; all candidates/modes/seeds inherit root group',
        limitation='Not sequence-identity/pocket-similarity clustering. Official test-source/PDB/exact ligand excluded by collector. MMP source-assay leakage and FM pretraining overlap unknown.',
        role='Internal fixed holdout within CrossDocked train; no official benchmark claim')
    save(root/'split.json',split)
    reuse=[]
    for cid in old['cases']:
        for kind in ('draws','evaluations'):
            shutil.copytree(source/kind/cid,root/kind/cid)
        shutil.copy2(source/'cases'/cid/'generation_done.json',root/'cases'/cid/'generation_done.json')
        reuse.append(dict(case_id=cid,origin=str(source),input_bytes_verified=True,
            rows_sha256=sha256_file(source/'evaluations'/cid/'rows.json')))
    save(root/'reuse_audit.json',dict(time_kst=now(),cases=reuse,
        note='Copied result records retain original absolute artifact paths; source files remain immutable. New manifest inputs byte-match original inputs. No new observations fabricated.'))
    plan.update(cohort=f'{len(cases)} roots, with frozen internal train/validation/test groups',
        reuse_source=str(source),reuse_cases=len(reuse),split_sha256=sha256_file(root/'split.json'))
    plan.pop('protocol_id',None);plan['protocol_id']=digest(plan);save(root/'study.json',plan)
    print({'roots':len(cases),'reused':len(reuse),'split':dict(counts),'groups':groupcounts},flush=True)


if __name__=='__main__':main()
