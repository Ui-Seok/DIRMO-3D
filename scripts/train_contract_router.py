"""GPU-only, grouped Router training from measured exact-graph pilot outcomes.

No LLM/FM calls. Separate candidate scorer, not a legacy utility-router artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
import time

from router_contract_pilot import ROOT, now, read, save, sha256_file, digest
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from dirmo.chemistry.graphs import mapped, read_sdf, smiles
from dirmo.chemistry.properties import graph_properties
from dirmo.evaluation.edited_graph import graph_from_record
from dirmo.evaluation.pocket import protein_atoms
from dirmo.routing.spatial import SPATIAL_FEATURE_NAMES, region_spatial_features

HEADS=['physical','feasible_delta_vina','accepted_positive_gain','log_amortized_seconds']
SEEDS=[20261105,20261106,20261107]
LRS=[1e-4,3e-4,1e-3]
WIDTHS=[(32,16),(128,64)]
FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=128)


def prepare(root, output):
    plan=read(root/'study.json'); split=read(root/'split.json')
    assert read(root/'status.json')['status']=='complete', 'Collection/evaluation must finish first'
    assert sha256_file(root/'split.json')==plan['split_sha256']
    props=sorted(graph_properties('CC'))
    first=read(root/'evaluations'/plan['cases'][0]/'initial_metrics.json')
    pbkeys=sorted(first['pb']['checks'])
    names=(SPATIAL_FEATURE_NAMES + [f'parent_{k}' for k in props]
        + [f'product_{k}' for k in props] + [f'delta_{k}' for k in props]
        + [f'parent_pb_{k}' for k in pbkeys]+[f'parent_pb_missing_{k}' for k in pbkeys]
        + ['parent_vina','parent_vina_missing','chemical_edit','relaxed','product_atoms',
           'atom_count_change','site_atoms','free_product_atoms','held_product_fraction','protected_fraction_product']
        + [f'product_morgan_{i}' for i in range(128)]+[f'morgan_difference_{i}' for i in range(128)])
    xs=[]; ys=[]; meta=[]; input_files={}
    for cid in plan['cases']:
        casefile=root/'cases'/cid/'case.json'; case=read(casefile)
        native=read(root/'evaluations'/cid/'initial_metrics.json')
        rowsfile=root/'evaluations'/cid/'rows.json'; rows=read(rowsfile)
        input_files[str(casefile)]=sha256_file(casefile);input_files[str(rowsfile)]=sha256_file(rowsfile)
        parent=mapped(read_sdf(case['initial_sdf'])); protein=protein_atoms(case['pocket_path'])
        pp=graph_properties(smiles(parent)); pfp=FP.GetFingerprintAsNumPy(parent).astype(float)
        lookup={(x['action'],x['mode'],x['draw']):x for x in rows}
        parent_vina=native['vina'].get('score'); parent_pb=native['pb'].get('passes',False)
        for action in case['actions']:
            graph=graph_from_record(action['graph']); product=action['graph']['properties']
            fp=FP.GetFingerprintAsNumPy(graph).astype(float)
            for mode in plan['modes']:
                region=set(action['site_parent_maps'])
                if mode=='relaxed':region.add(action['attachment_map'])
                spatial=region_spatial_features(parent,*protein,region,case['mandatory_protected_maps'])
                vector=spatial+[pp[k] for k in props]+[product[k] for k in props]+[product[k]-pp[k] for k in props]
                checks=native['pb']['checks']
                vector += [float(checks.get(k) is True) for k in pbkeys]+[float(checks.get(k) is None) for k in pbkeys]
                free=len(action[f'{mode}_free_maps']); n=graph.GetNumAtoms()
                vector += [parent_vina if parent_vina is not None else 0.,float(parent_vina is None),
                    float(action['action_kind']=='CHEMICAL_EDIT'),float(mode=='relaxed'),n,
                    n-parent.GetNumAtoms(),len(action['site_parent_maps']),free,(n-free)/n,
                    len(case['mandatory_protected_maps'])/n]
                vector += fp.tolist()+(fp-pfp).tolist()
                assert len(vector)==len(names) and np.isfinite(vector).all()
                for prior in action['priors']:
                    row=lookup[(action['graph']['name'],mode,prior['draw'])]
                    result=row.get('refined'); physical=result.get('joint_physical') if result else None
                    delta=result.get('vina_improvement_vs_initial') if result and physical is True else None
                    # Actual acceptance policy payoff, NOT an imputed affinity label.
                    gain=(max(0.,delta) if physical and delta is not None else 0. if physical is False else None) if parent_pb else None
                    stats=row['generation'].get('batch_stats'); cost=None
                    if stats and row.get('raw') and result:
                        cost=(stats['total_seconds']/stats['batch_size']+row['raw']['measurement_seconds']+
                            result['measurement_seconds']+result['uff']['seconds'])
                    label=[float(physical) if physical is not None else np.nan,
                        delta if delta is not None else np.nan,gain if gain is not None else np.nan,
                        float(np.log1p(cost)) if cost is not None else np.nan]
                    xs.append(vector);ys.append(label)
                    meta.append(dict(case_id=cid,group=split['groups'][cid],split=split['assignment'][cid],
                        action=action['graph']['name'],kind=action['action_kind'],mode=mode,seed=prior['seed'],draw=prior['draw'],
                        parent_pb_valid=bool(parent_pb),physical=physical,delta_vina=delta,accepted_gain=gain,
                        amortized_seconds=cost,delta_qed=product['qed']-pp['qed'],delta_sa=product['sa']-pp['sa'],
                        edited_atoms=len(action.get('edit_validation',{}).get('changed_atom_maps',[])),
                        generation_status=row['generation_status']))
    x=np.asarray(xs,dtype=np.float32);y=np.asarray(ys,dtype=np.float32)
    weights=np.array([1/Counter(r['case_id'] for r in meta)[r['case_id']] for r in meta],dtype=np.float32)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'arrays.npz',x=x,y=y,weights=weights)
    save(output/'rows.json',meta);save(output/'feature_names.json',names)
    audit=dict(time_kst=now(),roots=len(plan['cases']),rows=len(meta),features=len(names),heads=HEADS,
        labeled_rows={h:int(np.isfinite(y[:,i]).sum()) for i,h in enumerate(HEADS)},
        split_counts=dict(Counter(r['split'] for r in meta)),root_split_counts=split['root_counts'],
        group_split_counts=split['group_counts'],input_files=input_files,
        split_sha256=sha256_file(root/'split.json'),preparer_sha256=sha256_file(__file__),
        protocol=plan['protocol_id'],
        feature_provenance='Pre-action parent SDF/pocket, initial metrics, candidate graph and masks only; no generated/refined geometry, PB, Vina, seed, outcome availability, root IDs or initialization-failure flags as input',
        initialization_failure='No fake physical or affinity label; remains missing in supervised outcome heads',
        accepted_gain='For PB-valid parents: measured joint-physical output with improved Vina gets max(delta,0); observed physical failure/non-improvement rejected for realized payoff 0; missing stays missing. No QED/SA constraints imposed.',
        cost='Amortized observed FM batch wall-time + raw/refined evaluation and UFF; excludes startup, LLM and input preparation; shared hardware/cache dependent',
        scope='Internal offline candidate-ranking study on initialization-supported measured menu; not a closed-loop or causal guarantee')
    save(output/'audit.json',audit)
    return x,y,weights,meta,names


def fit(x,y,w,train,val,widths,lr,seed,epochs=200,forced_epochs=None):
    import torch
    from torch import nn
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    xm=x[train].mean(0);xs=x[train].std(0);xs[xs<1e-6]=1.
    ym=np.zeros(4,dtype=np.float32);ys=np.ones(4,dtype=np.float32)
    for j in range(1,4):
        known=train[np.isfinite(y[train,j])]
        if len(known)<2:raise ValueError(f'Insufficient train labels for {HEADS[j]}')
        ym[j]=np.mean(y[known,j]);ys[j]=max(np.std(y[known,j]),.05)
    z=(y-ym)/ys;mask=np.isfinite(z)
    tensors=[torch.as_tensor(a,device='cuda:0') for a in ((x-xm)/xs,np.nan_to_num(z),mask,w)]
    tx,ty,tm,tw=tensors
    model=nn.Sequential(nn.Linear(x.shape[1],widths[0]),nn.ReLU(),nn.Linear(widths[0],widths[1]),nn.ReLU(),nn.Linear(widths[1],4)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    ti=torch.tensor(train,device='cuda:0');vi=torch.tensor(val,device='cuda:0')
    def loss(ids):
        p=model(tx[ids]);terms=[]
        for j in range(4):
            keep=tm[ids,j]
            if not keep.any():continue
            v=(nn.functional.binary_cross_entropy_with_logits(p[keep,j],ty[ids[keep],j],reduction='none') if j==0 else
               nn.functional.smooth_l1_loss(p[keep,j],ty[ids[keep],j],reduction='none'))
            ww=tw[ids[keep]];terms.append((v*ww).sum()/ww.sum())
        return torch.stack(terms).mean()
    best=float('inf');best_epoch=0;best_state=None;stale=0;history=[]
    for epoch in range(1,(forced_epochs or epochs)+1):
        model.train();opt.zero_grad(set_to_none=True);tl=loss(ti)
        if not torch.isfinite(tl):raise ValueError('Nonfinite training loss')
        tl.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();model.eval()
        with torch.inference_mode():vl=float(loss(vi).item())
        history.append([epoch,float(tl.item()),vl])
        if forced_epochs or vl<best-1e-5:
            best=vl;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if not forced_epochs and epoch>=40 and stale>=25:break
    model.load_state_dict(best_state)
    layers=[dict(weight=l.weight.detach().cpu().tolist(),bias=l.bias.detach().cpu().tolist()) for l in model if isinstance(l,nn.Linear)]
    artifact=dict(widths=list(widths),lr=lr,seed=seed,best_epoch=best_epoch,best_validation_loss=best,
        x_mean=xm.tolist(),x_scale=xs.tolist(),y_mean=ym.tolist(),y_scale=ys.tolist(),layers=layers,
        parameters=sum(p.numel() for p in model.parameters()),device='cuda:0',history=history)
    del model,opt,tx,ty,tm,tw,tensors,ti,vi
    return artifact


def predict(model,x):
    z=(np.asarray(x)-model['x_mean'])/model['x_scale']
    for i,layer in enumerate(model['layers']):
        z=z@np.asarray(layer['weight']).T+layer['bias']
        if i<len(model['layers'])-1:z=np.maximum(z,0.)
    result=z*np.asarray(model['y_scale'])+np.asarray(model['y_mean'])
    result[:,0]=1/(1+np.exp(-np.clip(z[:,0],-30,30)))
    return result


def diagnostics(y,p,ids):
    from sklearn.metrics import average_precision_score,roc_auc_score
    result={}
    for j,h in enumerate(HEADS):
        ii=ids[np.isfinite(y[ids,j])]
        if not len(ii):result[h]={'n':0};continue
        a,b=y[ii,j],p[ii,j];item=dict(n=len(ii),mae=float(np.mean(np.abs(a-b))))
        if j==0:
            item.update(prevalence=float(a.mean()),brier=float(np.mean((a-b)**2)),
                average_precision=float(average_precision_score(a,b)) if a.sum()>0 else None,
                auroc=float(roc_auc_score(a,b)) if len(set(a))==2 else None)
        result[h]=item
    return result


def policies(rows,p,split_name):
    """Common complete 3-seed menus; expected payoff averaged, never best seed."""
    bycase=defaultdict(lambda:defaultdict(list));coverage=Counter()
    for i,r in enumerate(rows):
        if r['split']==split_name:bycase[r['case_id']][(r['action'],r['mode'])].append(i)
    outputs=[]
    for cid,actions in bycase.items():
        if not rows[next(iter(actions.values()))[0]]['parent_pb_valid']:
            coverage['parent_invalid_roots_excluded']+=1;continue
        menu=[]
        for key,ii in sorted(actions.items()):
            coverage['total_candidate_modes']+=1
            if len(ii)!=3 or any(rows[i]['accepted_gain'] is None for i in ii):
                coverage['incomplete_candidate_modes_excluded']+=1;continue
            first=rows[ii[0]]
            menu.append(dict(key=list(key),kind=first['kind'],pred=float(p[ii,2].mean()),
                gain=float(np.mean([rows[i]['accepted_gain'] for i in ii])),
                physical=float(np.mean([rows[i]['physical'] for i in ii])),
                cost=float(np.mean([rows[i]['amortized_seconds'] for i in ii])) if all(rows[i]['amortized_seconds'] is not None for i in ii) else None,
                rule=first['delta_qed'],edited_atoms=first['edited_atoms']))
        if len(menu)<2:coverage['fewer_than_two_complete_candidates']+=1;continue
        best=max(menu,key=lambda a:(a['pred'],-a['edited_atoms']))
        rule=max(menu,key=lambda a:(a['rule'],-a['edited_atoms']))
        poses=[a for a in menu if a['kind']=='POSE_RESAMPLE']
        selected=best if best['pred']>0 else None
        outputs.append(dict(case_id=cid,group=rows[next(iter(actions.values()))[0]]['group'],
            candidates=len(menu),selected=selected['key'] if selected else 'STOP',
            learned=selected['gain'] if selected else 0.,learned_physical=selected['physical'] if selected else 1.,
            learned_cost=selected['cost'] if selected else 0.,
            random=float(np.mean([a['gain'] for a in menu])),property_rule=rule['gain'],stop=0.,
            pose_random=float(np.mean([a['gain'] for a in poses])) if poses else None,
            oracle=max(a['gain'] for a in menu)))
    means={k:float(np.mean([o[k] for o in outputs if o[k] is not None])) if outputs and any(o[k] is not None for o in outputs) else None
        for k in ('learned','random','property_rule','stop','pose_random','oracle','learned_physical','learned_cost')}
    means['stopped_roots']=sum(o['selected']=='STOP' for o in outputs)
    ci=None
    groups=sorted({r['group'] for r in outputs})
    if len(groups)>=3:
        rng=np.random.default_rng(20261105);diff=[]
        for _ in range(2000):
            sampled=rng.choice(groups,size=len(groups),replace=True)
            obs=[r for g in sampled for r in outputs if r['group']==g]
            diff.append(float(np.mean([r['learned']-r['random'] for r in obs])))
        ci=np.quantile(diff,[.025,.975]).tolist()
    return dict(roots=len(outputs),coverage=dict(coverage),means=means,learned_minus_random_group_bootstrap_95ci=ci,
        rows=outputs,note='Retrospective, initialization-supported complete 3-seed common menus. Actual invalid/non-improving outputs rejected to keep parent. Conditional diagnostic, not unbiased unrestricted-policy performance. Pose baseline may cover fewer roots.')


def train(args):
    from dirmo.gpu import inventory, resolve_device, worker_environment
    device=resolve_device(args.gpu,inventory());os.environ.update(worker_environment(device,str(ROOT/'src')))
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(1)
    root=args.source.resolve(); output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    save(output/'status.json',dict(status='preparing_dataset',time_kst=now(),pid=os.getpid(),gpu=args.gpu))
    x,y,w,rows,names=prepare(root,output/'data')
    ids={s:np.array([i for i,r in enumerate(rows) if r['split']==s],dtype=int) for s in ('train','validation','test')}
    for key in ('case_id','group'):
        sets=[{rows[i][key] for i in ids[s]} for s in ids]
        assert all(not sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2)))
    save(output/'training_config.json',dict(time_kst=now(),widths=WIDTHS,lrs=LRS,seeds=SEEDS,
        max_epochs=200,early_stopping_patience=25,minimum_epochs=40,optimizer='AdamW',weight_decay=.01,
        loss='Equal mean of masked root-weighted BCE(physical) and SmoothL1(feasible delta, realized positive accepted gain, log cost); scales fit on train only',
        policy='Select highest predicted expected accepted positive Vina gain, STOP if <=0; tie break by fewer edited atoms; no arbitrary utility or cost multiplier',
        objective='PB-valid parent optimization, reject graph/held/PB failures and nonpositive Vina gain; QED/SA reported but no thresholds silently added',
        trainer_sha256=sha256_file(__file__),split_sha256=sha256_file(root/'split.json'),gpu_uuid=device['uuid']))
    start=time.monotonic();trials=[]
    for widths in WIDTHS:
        for lr in LRS:
            models=[]
            for seed in SEEDS:
                model=fit(x,y,w,ids['train'],ids['validation'],widths,lr,seed)
                tag=f'{widths[0]}_{widths[1]}_{lr}_{seed}'
                save(output/'trials'/f'{tag}.json',model);models.append(model)
                save(output/'status.json',dict(status='validation_search',time_kst=now(),pid=os.getpid(),gpu=args.gpu,
                    trial=tag,completed_trials=len(trials)*len(SEEDS)+len(models),planned_trials=18))
            trial=dict(widths=list(widths),lr=lr,validation_loss=float(np.mean([m['best_validation_loss'] for m in models])),
                epochs=[m['best_epoch'] for m in models])
            # Evaluate validation only here. Test labels never select architecture/LR/epochs.
            vp=np.mean([predict(m,x[ids['validation']]) for m in models],axis=0)
            trial['validation_diagnostics']=diagnostics(y[ids['validation']],vp,np.arange(len(vp)))
            trials.append(trial);save(output/'validation_search.json',trials)
    winner=min(trials,key=lambda t:t['validation_loss'])
    epochs=int(np.median(winner['epochs']));trainval=np.concatenate([ids['train'],ids['validation']])
    final=[]
    for seed in SEEDS:
        # Fixed validation-selected epoch count; no held-out test used during fitting.
        m=fit(x,y,w,trainval,trainval,winner['widths'],winner['lr'],seed,forced_epochs=epochs)
        m['fit_scope']='train+validation refit, fixed epochs; best_validation_loss field is refit train loss'
        final.append(m)
    prediction=np.mean([predict(m,x) for m in final],axis=0)
    np.save(output/'predictions.npy',prediction)
    artifact=dict(format='dirmo-contract-router-v1',heads=HEADS,feature_names=names,models=final,
        protocol=read(root/'study.json')['protocol_id'],training_config=read(output/'training_config.json'),
        integration='Offline candidate scorer only. Feature extraction and acceptance protocol must match; not compatible with legacy utility router.json.',
        inference='Mean of 3 independently seeded MLP predictions; sigmoid head0, inverse-standardize heads1..3; head3 is log1p amortized seconds')
    save(output/'router.json',artifact)
    # Exactly one final evaluation on the locked test after all choices are fixed.
    metrics=dict(time_kst=now(),selected=winner,refit_epochs=epochs,
        test_heads=diagnostics(y,prediction,ids['test']),test_policy=policies(rows,prediction,'test'),
        split=read(root/'split.json'),elapsed_seconds=time.monotonic()-start,
        actual_gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limitations=['100 parent states, repeated seeds are not independent targets',
            'Previously inspected 50 roots and connected groups used only for training',
            'Internal holdout uses exact/family/scaffold groups, not full sequence/pocket clustering',
            'Frozen LLM not called; FM weights not trained; no closed-loop deployment',
            'MMP assay provenance and source-evidence leakage remain unaudited',
            'Choice metrics are restricted to complete measured 3-seed candidate menus'])
    save(output/'metrics.json',metrics)
    save(output/'status.json',dict(status='complete',time_kst=now(),pid=os.getpid(),gpu=args.gpu,
        model=str(output/'router.json'),metrics=str(output/'metrics.json'),refit_epochs=epochs))
    print(json.dumps({'status':'complete','output':str(output),'test':metrics['test_heads'],'policy':metrics['test_policy']['means']},ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--gpu',default='1')
    args=ap.parse_args()
    try:train(args)
    except BaseException as exc:
        if args.output.exists():save(args.output/'status.json',dict(status='failed',time_kst=now(),reason=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':main()
