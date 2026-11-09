"""Frozen full-ligand SBMO external evaluation; no LLM or parameter updates."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMMPA
from router_contract_pilot import (ROOT, CHECKPOINT, CHECKSUM, now, save, read, log,
    digest, sha256_file, join, fragment_key, validate_edit)
from molgenbench_data import DATA, identity
from dirmo.chemistry.graphs import mapped, map_indices, read_sdf, write_sdf, graph_id, protected_graph, smiles
from dirmo.schemas import Protection
from dirmo.evaluation.edited_graph import graph_record, graph_from_record, graph_only_conformer, anchor_indices, unsupported_tensor_features

ROUTER=ROOT/'runs/router_contract_gpu1_100_20261105_v1/router_training/router.json'

def menu(parent, scaffold, rules, seed, limit=8):
    query=Chem.MolFromSmarts(scaffold)
    matches=parent.GetSubstructMatches(query,uniquify=True)
    if not matches:raise ValueError('Scaffold does not match parent')
    core={parent.GetAtomWithIdx(i).GetAtomMapNum() for i in min(matches)}
    pmaps=set(map_indices(parent)); proposals=[];seen=set();audit=[]
    for _,chain in rdMMPA.FragmentMol(parent,maxCuts=1,resultsAsMols=True):
        if chain is None:continue
        pieces=Chem.GetMolFrags(chain,asMols=True,sanitizeFrags=False)
        if len(pieces)!=2:continue
        for rest,part in (pieces,pieces[::-1]):
            removed={a.GetAtomMapNum() for a in part.GetAtoms() if a.GetAtomicNum()}
            if removed&core or not 1<=len(removed)<=min(8,parent.GetNumAtoms()//3):continue
            dummy=next(a for a in rest.GetAtoms() if not a.GetAtomicNum())
            attachment=dummy.GetNeighbors()[0].GetAtomMapNum()
            lhs=fragment_key(part)
            for rule in rules.get(lhs,[]):
                key=(tuple(sorted(removed)),rule['rhs'])
                if key in seen:continue
                seen.add(key)
                proposals.append((digest([seed,key]),rest,sorted(removed),attachment,lhs,rule))
    chosen=[]
    for rank,rest,removed,attachment,lhs,rule in sorted(proposals,key=lambda x:x[0]):
        if len(chosen)>=limit:break
        entry=dict(rank=rank,lhs=lhs,rhs=rule['rhs'],site=removed)
        try:
            product,change=validate_edit(parent,join(rest,rule['rhs'],max(pmaps)+1))
            if not 6<=product.GetNumAtoms()<=64 or unsupported_tensor_features(product):
                raise ValueError('Unsupported graph size/tensor features')
            protected_graph(parent,product,Protection(atom_maps=sorted(core)))
            if any(a['graph']['graph_id']==graph_id(product) for a in chosen):raise ValueError('Duplicate product')
            added=set(map_indices(product))-pmaps
            if not 1<=len(added)<=10:raise ValueError('Replacement atom count')
            lookup=map_indices(product)
            relaxed=added|{attachment}
            anchor_indices(product,[lookup[m] for m in relaxed],next(i for m,i in lookup.items() if m not in relaxed))
            chosen.append(dict(action_kind='CHEMICAL_EDIT',
                graph=graph_record(f'edit{len(chosen):02d}',product,None,[lookup[m] for m in added]),
                site_parent_maps=removed,attachment_map=attachment,strict_free_maps=sorted(added),
                relaxed_free_maps=sorted(relaxed),edit_validation=change,
                mmp=dict(lhs=lhs,**rule),chemical_protected_maps=sorted(core)))
            entry['status']='selected'
        except Exception as exc:entry.update(status='excluded',reason=f'{type(exc).__name__}: {exc}')
        audit.append(entry)
    # Chemical core stays identical in both arms. Relaxed permits the attachment
    # coordinate to move even when it belongs to MCS; this exception is explicit.
    mandatory=core-{a['attachment_map'] for a in chosen}
    return chosen,sorted(core),sorted(mandatory),audit

def prepare(args):
    root=args.output.resolve(); split=read(root/'target_split.json')
    if (root/'study.json').exists():return read(root/'study.json')
    rules=defaultdict(list)
    for rule in read(root/'development_rules.json'):
        for left,right in ((rule['lhs'],rule['rhs']),(rule['rhs'],rule['lhs'])):
            assert all(w['target'] not in split['evaluation_targets'] for w in rule['witnesses'])
            rules[left].append(dict(rhs=right,witnesses=rule['witnesses'],witness_count=rule['witness_count']))
    cases=[];failures=[]
    for record in split['selected']:
        cid=record['target']+'_'+record['series']; directory=root/'cases'/cid
        try:
            parent=mapped(read_sdf(record['reference_sdf']))
            # Official reference SDFs carry affinity/Glide metadata. The model
            # receives only the molecular graph and coordinates, not those labels.
            for prop in list(parent.GetPropNames(includePrivate=True,includeComputed=False)):
                parent.ClearProp(prop)
            parent.SetProp('_Name',cid)
            if unsupported_tensor_features(parent):raise ValueError('Unsupported parent tensor features')
            actions,core,mandatory,audit=menu(parent,record['scaffold_smarts'],rules,args.seed)
            save(directory/'menu_audit.json',audit)
            write_sdf(directory/'initial.sdf',parent)
            from dirmo.evaluation.pocket import prepare_fixed_pocket,protein_atoms
            pocket=directory/'pocket.pdb'
            prepare_fixed_pocket(record['protein_pdb'],str(directory/'initial.sdf'),str(pocket),6.)
            positions,elements,_=protein_atoms(str(pocket))
            if len(positions)>1000 or set(elements)-{'C','N','O','P','S'}:raise ValueError('Unsupported pocket')
            xyz=parent.GetConformer().GetPositions();plookup=map_indices(parent)
            for ai,action in enumerate(actions):
                graph=graph_from_record(action['graph']);lookup=map_indices(graph)
                held={i:xyz[plookup[m]].tolist() for m,i in lookup.items() if m not in action['strict_free_maps']}
                anchors=anchor_indices(graph,[lookup[m] for m in action['strict_free_maps']],next(iter(held)))
                priors=[]
                for draw in range(3):
                    seed=args.seed+len(cases)*10000+ai*100+draw
                    path=directory/'inputs'/action['graph']['name']/f'{draw:02d}.sdf'
                    prior=dict(draw=draw,seed=seed)
                    try:
                        mol=graph_only_conformer(graph,held,anchors,seed);write_sdf(path,mol)
                        prior.update(status='ready',sdf=str(path),sha256=sha256_file(path))
                    except Exception as exc:prior.update(status='initialization_failed',reason=str(exc))
                    priors.append(prior)
                action['priors']=priors
            case=dict(case_id=cid,index=len(cases),target_id=record['target'],series=record['series'],split='external_test',
                parent_graph_id=graph_id(parent),initial_sdf=str(directory/'initial.sdf'),pocket_path=str(pocket),
                source_ligand=record['reference_sdf'],source_protein=record['protein_pdb'],
                source_hashes=dict(ligand=sha256_file(record['reference_sdf']),protein=sha256_file(record['protein_pdb'])),
                source_pose='Glide-docked reference; not experimental co-crystal ligand pose',
                scaffold_smarts=record['scaffold_smarts'],chemical_protected_maps=core,
                mandatory_protected_maps=mandatory,actions=actions,uff_only_control=True,uff_only_all_actions=True,
                box_center=xyz.mean(axis=0).tolist(),box_size=[40.,40.,40.])
            save(directory/'case.json',case);cases.append(cid)
            log('molgenbench_prepared_case',case_id=cid,chemical_candidates=len(actions))
        except Exception as exc:
            failures.append(dict(case_id=cid,reason=f'{type(exc).__name__}: {exc}'))
            log('molgenbench_preparation_failure',case_id=cid,reason=str(exc))
    shutil.copy2(args.router,root/'frozen_router.json')
    plan=dict(created_at_kst=now(),cases=cases,requested_targets=20,selected_targets=len(split['selected']),
        preparation_failures=failures,empty_menus=[c for c in cases if not read(root/'cases'/c/'case.json')['actions']],
        planned_fm_draws=sum(6*len(read(root/'cases'/c/'case.json')['actions']) for c in cases),
        physical_gpu=args.gpu,steps=64,draws_per_action=3,modes=['strict','relaxed'],selection_seed=args.seed,
        checkpoint=CHECKPOINT,checkpoint_sha256=CHECKSUM,collector_sha256=sha256_file(ROOT/'scripts/router_contract_pilot.py'),
        router=str(root/'frozen_router.json'),router_sha256=sha256_file(root/'frozen_router.json'),
        split_sha256=sha256_file(root/'target_split.json'),rules_sha256=sha256_file(root/'development_rules.json'),
        adapter_sha256=sha256_file(__file__),fm_training=False,router_training=False,llm_calls=0,closed_loop=False,
        regime='MolGenBench full-reference-ligand SBMO adaptation, offline frozen policy evaluation',
        source_pose='Glide-docked',uff_only='Every candidate and mode, same ETKDG prior; no extra FM draw',
        condition='Entire edited stereograph clamped; strict fixes retained XYZ; relaxed frees attachment XYZ too; MCS chemical identities/bonds preserved',
        evaluation='Historical DIRMO PB dock20 + Vina score_only, fixed 6A pocket and 40A box; not official MolGenBench leaderboard metrics',
        selection='Frozen pre-outcome scores; failures remain in intended policy menu; no hindsight complete-menu selection',
        activity='Evaluation-only reference rediscovery, same series; unmatched is unknown, not inactive',
        limitations=['Historical Router MMP source exposure unknown','FM pretraining membership unknown',
                    'Scaffold alone is not input: full reference ligand supplied','No experimental activity for novel generated graphs'])
    plan['protocol_id']=digest(plan);save(root/'study.json',plan)
    return plan

def prescore(root):
    """Use exactly the original feature names and frozen model scaling.

    Only initial metrics and proposed graphs enter here, before reading any FM
    outcomes. Failure/availability/seed are never features or menu filters.
    """
    from train_contract_router import FP,predict
    from dirmo.chemistry.properties import graph_properties
    from dirmo.evaluation.pocket import protein_atoms
    from dirmo.routing.spatial import SPATIAL_FEATURE_NAMES,region_spatial_features
    artifact=read(root/'frozen_router.json');plan=read(root/'study.json')
    if sha256_file(root/'frozen_router.json')!=plan['router_sha256']:raise ValueError('Router changed')
    props=sorted(graph_properties('CC'));names=artifact['feature_names'];outputs=[]
    for cid in plan['cases']:
        path=root/'decisions'/f'{cid}.json'
        if path.exists():outputs.append(read(path));continue
        case=read(root/'cases'/cid/'case.json');native_path=root/'evaluations'/cid/'initial_metrics.json'
        if not native_path.exists():continue
        native=read(native_path);parent=mapped(read_sdf(case['initial_sdf']));protein=protein_atoms(case['pocket_path'])
        pp=graph_properties(smiles(parent));pfp=FP.GetFingerprintAsNumPy(parent).astype(float);options=[]
        for action in case['actions']:
            graph=graph_from_record(action['graph']);product=action['graph']['properties']
            fp=FP.GetFingerprintAsNumPy(graph).astype(float)
            for mode in plan['modes']:
                region=set(action['site_parent_maps'])
                if mode=='relaxed':region.add(action['attachment_map'])
                f=dict(zip(SPATIAL_FEATURE_NAMES,region_spatial_features(parent,*protein,region,case['mandatory_protected_maps'])))
                for k in props:f.update({f'parent_{k}':pp[k],f'product_{k}':product[k],f'delta_{k}':product[k]-pp[k]})
                for name in names:
                    if name.startswith('parent_pb_missing_'):
                        f[name]=float(native['pb']['checks'].get(name[len('parent_pb_missing_'):]) is None)
                    elif name.startswith('parent_pb_'):
                        f[name]=float(native['pb']['checks'].get(name[len('parent_pb_'):]) is True)
                vina=native['vina'].get('score');n=graph.GetNumAtoms();free=len(action[f'{mode}_free_maps'])
                f.update(parent_vina=vina or 0.,parent_vina_missing=float(vina is None),chemical_edit=1.,relaxed=float(mode=='relaxed'),
                    product_atoms=n,atom_count_change=n-parent.GetNumAtoms(),site_atoms=len(action['site_parent_maps']),
                    free_product_atoms=free,held_product_fraction=(n-free)/n,protected_fraction_product=len(case['mandatory_protected_maps'])/n)
                f.update({f'product_morgan_{i}':float(v) for i,v in enumerate(fp)})
                f.update({f'morgan_difference_{i}':float(v) for i,v in enumerate(fp-pfp)})
                x=np.array([[f[name] for name in names]])
                if not np.isfinite(x).all():raise ValueError('Nonfinite input features')
                prediction=np.mean([predict(model,x)[0] for model in artifact['models']],axis=0).tolist()
                options.append(dict(key=[action['graph']['name'],mode],prediction=prediction,delta_qed=product['qed']-pp['qed'],
                    delta_sa=product['sa']-pp['sa'],edited_atoms=len(action['edit_validation'].get('changed_atom_maps',[]))))
        options.sort(key=lambda a:a['key'])
        best=max(options,key=lambda a:(a['prediction'][2],-a['edited_atoms'])) if options else None
        rule=max(options,key=lambda a:(a['delta_qed'],-a['edited_atoms'])) if options else None
        decision=dict(time_kst=now(),case_id=cid,options=options,
            learned=best['key'] if best and best['prediction'][2]>0 else 'STOP',
            property_rule=rule['key'] if rule else 'STOP',random='uniform_over_all_proposed_candidate_modes',
            initial_pb_valid=native['pb'].get('passes') if not native['pb'].get('measurement_missing') else None,
            initial_metrics_sha256=sha256_file(native_path),router_sha256=plan['router_sha256'],
            provenance='Pre-action features only; no generated outcomes read')
        save(path,decision);outputs.append(decision)
    return outputs

def run(args):
    """Reuse the tested collector/evaluator; independent child logs and resumability."""
    root=args.output.resolve();plan=read(root/'study.json')
    if not plan['cases']:raise ValueError('No prepared cases; see preparation_failures')
    # Measure and freeze decisions before FM generation, including empty menus.
    import evaluate_router_contract_pilot as evaluation
    from dirmo.evaluation.crossdocked_sbdd import prepare_rigid_receptor
    evaluation.configure(root)
    def measure_initial(cid):
        path=root/'evaluations'/cid/'initial_metrics.json'
        if path.exists():return
        case=read(root/'cases'/cid/'case.json');mol=read_sdf(case['initial_sdf'])
        try:case['receptor_path']=prepare_rigid_receptor(case['pocket_path'],path.parent)[0]
        except Exception as exc:case['receptor_path']=None;save(path.parent/'receptor_error.json',dict(reason=str(exc)))
        save(path,evaluation.assess(case['initial_sdf'],mol,mol,set(),case))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(measure_initial,plan['cases']))
    prescore(root)
    command=[sys.executable,str(ROOT/'scripts/run_router_contract_pilot.py'),'--output',str(root),'--workers',str(args.workers)]
    subprocess.run(command,check=True)
    report(root)

def report(root):
    from molgenbench_report import report as make_report
    return make_report(root)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','run','report','prescore'])
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--router',type=Path,default=ROUTER)
    ap.add_argument('--gpu',default='3');ap.add_argument('--seed',type=int,default=20261105)
    ap.add_argument('--workers',type=int,default=6)
    args=ap.parse_args();args.output=args.output.resolve();RDLogger.DisableLog('rdApp.*')
    if args.command=='prepare':prepare(args)
    elif args.command=='run':run(args)
    elif args.command=='prescore':prescore(args.output)
    else:report(args.output)

if __name__=='__main__':main()
