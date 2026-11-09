"""GPU-only A/B/C Router development training, with grouped out-of-fold evaluation.

No worker lifecycle calls and no FM/LLM requests. Artifacts are versioned offline
scorers, not drop-in artifacts for the legacy pocket-specific LearnedRouter.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import numpy as np
from sklearn.model_selection import GroupKFold,GroupShuffleSplit
from sklearn.metrics import roc_auc_score

MODELS=['A','B','C']
SEEDS=[20261026,20261027,20261028]
LRS=[.0003,.001,.003]
LAMBDAS=[0.,.0002,.001,.01]


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def emit(**data):
    print(json.dumps({'time_kst':now(),**data},ensure_ascii=False,allow_nan=False),flush=True)


def masks_for(y,variant):
    mask=np.isfinite(y[:,:2])
    if variant in ['A','B']:
        # Legacy-like baseline: both regression targets use utility-observed rows.
        mask[:,1]=mask[:,0]
    else:
        # C explicitly predicts utility conditional on usable/feasible output.
        mask[:,0] &= y[:,2]>.5
    return mask


def fit(x,y,train,val,variant,lr,seed,max_epochs=500,forced_epochs=None):
    import torch
    from torch import nn
    device=torch.device('cuda:0')
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    mask=masks_for(y,variant)
    scaling_ids=train[mask[train].any(axis=1)]
    xm=x[scaling_ids].mean(axis=0);xs=x[scaling_ids].std(axis=0);xs[xs<1e-6]=1.
    ym=[];ys=[]
    for j in range(2):
        selected=train[mask[train,j]]
        if not len(selected):raise ValueError('No observed training labels for head')
        ym.append(float(y[selected,j].mean()));ys.append(max(float(y[selected,j].std()),1e-6))
    ym=np.array(ym,dtype=np.float32);ys=np.array(ys,dtype=np.float32)
    labels=np.nan_to_num((y[:,:2]-ym)/ys,nan=0.)
    tx=torch.tensor((x-xm)/xs,dtype=torch.float32,device=device)
    ty=torch.tensor(labels,dtype=torch.float32,device=device)
    tm=torch.tensor(mask,dtype=torch.bool,device=device)
    tf=torch.tensor(y[:,2],dtype=torch.float32,device=device)
    ti=torch.tensor(train,dtype=torch.long,device=device)
    vi=torch.tensor(val if len(val) else train,dtype=torch.long,device=device)
    model=nn.Sequential(nn.Linear(x.shape[1],32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),
                        nn.Linear(16,3 if variant=='C' else 2)).to(device)
    if any(p.device.type!='cuda' for p in model.parameters()):raise AssertionError('CPU training forbidden')
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    def loss(indices):
        pred=model(tx[indices]);terms=[]
        for j in range(2):
            keep=tm[indices,j]
            if keep.any():terms.append(nn.functional.smooth_l1_loss(pred[keep,j],ty[indices,j][keep]))
        value=torch.stack(terms).mean()
        if variant=='C':value=value+nn.functional.binary_cross_entropy_with_logits(pred[:,2],tf[indices])
        return value
    best=float('inf');best_epoch=0;best_state=None;stale=0;last_train=None
    epochs=forced_epochs or max_epochs
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        value=loss(ti);value.backward();optimizer.step()
        if not torch.isfinite(value):raise ValueError('Nonfinite GPU loss')
        if epoch%5 and epoch!=epochs:continue
        last_train=float(value.detach().item());model.eval()
        with torch.inference_mode():vl=float(loss(vi).item())
        if forced_epochs or vl<best-1e-5:
            best=vl;best_epoch=epoch;stale=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if not forced_epochs and epoch>=50 and stale>=12:break
    model.load_state_dict(best_state);model.eval()
    with torch.inference_mode():z=model(tx).detach().cpu().numpy()
    pred=z[:,:2]*ys+ym
    probability=1/(1+np.exp(-np.clip(z[:,2],-30,30))) if variant=='C' else np.full(len(x),np.nan)
    layers=[{'weight':l.weight.detach().cpu().numpy().tolist(),'bias':l.bias.detach().cpu().numpy().tolist()}
            for l in model if isinstance(l,nn.Linear)]
    artifact={'variant':variant,'mean':xm.tolist(),'scale':xs.tolist(),
              'label_mean':ym.tolist(),'label_scale':ys.tolist(),'layers':layers,
              'best_epoch':best_epoch,'best_validation_loss':best,'last_training_loss':last_train,
              'lr':lr,'seed':seed,'training_device':str(device),
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'utility_target':'delta conditional on recorded feasibility' if variant=='C' else 'delta conditional on metric observation'}
    del model,optimizer,tx,ty,tm,tf,ti,vi
    return artifact,pred,probability


def predict_artifact(artifact,x):
    x=(np.asarray(x)-artifact['mean'])/artifact['scale']
    for i,layer in enumerate(artifact['layers']):
        x=x@np.asarray(layer['weight']).T+layer['bias']
        if i<len(artifact['layers'])-1:x=np.maximum(0,x)
    reg=x[:,:2]*artifact['label_scale']+artifact['label_mean']
    p=1/(1+np.exp(-np.clip(x[:,2],-30,30))) if artifact['variant']=='C' else None
    return reg,p


def grouped_metadata(rows,indices):
    return {key:sorted({rows[i][key] for i in indices}) for key in
            ['state_id','target_id','series_id','parent_graph_id','group_id']}


def verify_split(rows,train,val,test):
    for key in ['state_id','target_id','series_id','parent_graph_id','group_id']:
        sets=[{rows[i][key] for i in indices} for indices in [train,val,test]]
        if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:
            raise AssertionError(f'{key} leakage')


def repeated_outcomes(rows,prediction,probability,variant,lam,stop):
    """Choose an action, then average its observed seeds; never select a lucky seed."""
    states=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(rows):states[row['state_id']][row['action_id']].append(i)
    result=[]
    for state,actions in sorted(states.items()):
        menus=[]
        for action,ids in sorted(actions.items()):
            delta=float(prediction[ids,0].mean());cost=float(prediction[ids,1].mean())
            p=float(probability[ids].mean()) if variant=='C' else 1.
            score=(p*max(0.,delta) if variant=='C' else delta)-lam*max(0.,cost)
            feasible=[rows[i] for i in ids if rows[i]['feasible']]
            gains=[max(0.,rows[i]['utility_delta']) if rows[i]['feasible'] and rows[i]['utility_delta'] is not None else 0. for i in ids]
            menus.append({'action':action,'ids':ids,'score':score,'p':p,
                          'gain':float(np.mean(gains)),'feasible':len(feasible)/len(ids),
                          'cost':float(np.mean([rows[i]['wall_seconds'] for i in ids])),
                          'deltas':{k:float(np.mean([r['metric_deltas'][k] for r in feasible if k in r['metric_deltas']]))
                                    for k in ['vina','qed','sa'] if any(k in r['metric_deltas'] for r in feasible)}})
        selected=max(menus,key=lambda m:(m['score'],m['p'] if variant=='C' else 0.))
        parent=rows[menus[0]['ids'][0]]
        stopped=bool(stop and selected['score']<=0)
        gain=(0. if stopped else selected['gain']) if parent['parent_feasible'] else None
        oracle=max([0.]+[m['gain'] for m in menus]) if parent['parent_feasible'] else None
        result.append({'state_id':state,'target_id':parent['target_id'],'group_id':parent['group_id'],
                       'selected_row':None if stopped else selected['ids'][0],
                       'selected_rows':[] if stopped else selected['ids'],
                       'selected_action':'STOP' if stopped else selected['action'],
                       'parent_feasible':parent['parent_feasible'],
                       'selected_feasible':0. if stopped else selected['feasible'],
                       'stopped':stopped,'wall_seconds':0. if stopped else selected['cost'],
                       'best_feasible_gain':gain,'observed_oracle_gain':oracle,
                       'observed_regret':oracle-gain if oracle is not None else None,
                       'selected_metric_deltas':{} if stopped else selected['deltas']})
    return result


def outcomes(rows,prediction,probability,variant,lam=0.,stop=True,average_seeds=False):
    if average_seeds:return repeated_outcomes(rows,prediction,probability,variant,lam,stop)
    grouped=defaultdict(list)
    for i,row in enumerate(rows):grouped[row['state_id']].append(i)
    result=[]
    for state,ids in sorted(grouped.items()):
        # C models accepted positive gain; negative children leave the feasible
        # incumbent unchanged. This also avoids preferring p~0 solely because it
        # attenuates a negative utility prediction toward zero.
        scores=np.array([(max(0.,prediction[i,0])*probability[i] if variant=='C' else prediction[i,0])-
                         lam*max(0.,prediction[i,1]) for i in ids])
        best=np.flatnonzero(scores==scores.max())
        pos=int(best[np.argmax([probability[ids[j]] for j in best])]) if variant=='C' else int(best[0])
        picked=ids[pos]
        chosen=None if stop and scores.max()<=0 else picked
        source=rows[chosen] if chosen is not None else None
        parent=rows[ids[0]]
        feasible=bool(source and source['feasible'])
        gain=(max(0.,source['utility_delta']) if source and feasible and source['utility_delta'] is not None else 0.) if parent['parent_feasible'] else None
        oracle=max([0.]+[max(0.,rows[i]['utility_delta']) for i in ids
                         if rows[i]['feasible'] and rows[i]['utility_delta'] is not None]) if parent['parent_feasible'] else None
        result.append({'state_id':state,'target_id':parent['target_id'],'group_id':parent['group_id'],
                       'selected_row':chosen,'selected_action':source['action_id'] if source else 'STOP',
                       'parent_feasible':parent['parent_feasible'],'selected_feasible':feasible,
                       'stopped':source is None,'wall_seconds':source['wall_seconds'] if source else 0.,
                       'best_feasible_gain':gain,'observed_oracle_gain':oracle,
                       'observed_regret':oracle-gain if oracle is not None else None,
                       'selected_metric_deltas':source['metric_deltas'] if feasible else {}})
    return result


def aggregate(decisions):
    known=[r for r in decisions if r['best_feasible_gain'] is not None]
    valid=[r for r in decisions if not r['stopped']]
    return {'states':len(decisions),'executed_states':len(valid),
            'stop_rate':sum(r['stopped'] for r in decisions)/len(decisions),
            'feasible_yield':sum(r['selected_feasible'] for r in decisions)/len(decisions),
            'feasible_rate_among_executed':sum(r['selected_feasible'] for r in valid)/len(valid) if valid else None,
            'mean_wall_seconds':float(np.mean([r['wall_seconds'] for r in decisions])),
            'parent_feasible_states':len(known),
            'mean_best_feasible_gain':float(np.mean([r['best_feasible_gain'] for r in known])) if known else None,
            'mean_observed_regret':float(np.mean([r['observed_regret'] for r in known])) if known else None,
            'gain_ge_0.01_rate':sum(r['best_feasible_gain']>=.01 for r in known)/len(known) if known else None,
            'selected_metric_delta_means':{k:float(np.mean([r['selected_metric_deltas'][k] for r in decisions
                                            if k in r['selected_metric_deltas']])) for k in ['vina','qed','sa']
                                         if any(k in r['selected_metric_deltas'] for r in decisions)}}


def paired_ci(a,b,field):
    d={r['state_id']:r for r in b};groups=defaultdict(list)
    for r in a:
        if r[field] is not None and d[r['state_id']][field] is not None:
            groups[r['group_id']].append(float(r[field])-float(d[r['state_id']][field]))
    means=np.array([np.mean(v) for _,v in sorted(groups.items())]);rng=np.random.default_rng(20261026)
    samples=means[rng.integers(len(means),size=(5000,len(means)))].mean(axis=1)
    return {'group_mean_difference':float(means.mean()),'ci95':[float(v) for v in np.quantile(samples,[.025,.975])],
            'groups':len(means),'resamples':5000,'seed':20261026}


def random_oracle(rows,average_seeds=False):
    by=defaultdict(list)
    for r in rows:by[r['state_id']].append(r)
    random=[];oracle=[]
    for state,rs in by.items():
        parent=rs[0]['parent_feasible'];g=[max(0.,r['utility_delta']) if r['feasible'] and r['utility_delta'] is not None else 0. for r in rs]
        f=[r['feasible'] for r in rs]
        if average_seeds:
            groups=defaultdict(list)
            for i,r in enumerate(rs):groups[r['action_id']].append(i)
            g=[float(np.mean([g[i] for i in ids])) for ids in groups.values()]
            f=[float(np.mean([rs[i]['feasible'] for i in ids])) for ids in groups.values()]
        random.append({'state_id':state,'expected_feasible':float(np.mean(f)),
                       'expected_best_feasible_gain':float(np.mean(g)) if parent else None})
        oracle.append({'state_id':state,'any_feasible':max(f),
                       'best_feasible_gain':max(g) if parent else None})
    return {'random_expected_forced_feasible_rate':float(np.mean([r['expected_feasible'] for r in random])),
            'random_expected_best_feasible_gain':float(np.mean([r['expected_best_feasible_gain'] for r in random if r['expected_best_feasible_gain'] is not None])),
            'observed_oracle_any_feasible_rate':float(np.mean([r['any_feasible'] for r in oracle])),
            'observed_oracle_best_feasible_gain':float(np.mean([r['best_feasible_gain'] for r in oracle if r['best_feasible_gain'] is not None])),
            'note':('Metric-specific observed action-seed-mean upper bounds; not a best-seed choice.' if average_seeds else
                    'Metric-specific observed single-seed upper bounds; unknown/unmeasured alternatives are not included.')}


def run(args):
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one visible CUDA GPU required; CPU fallback is forbidden')
    device=torch.cuda.get_device_properties(0)
    prepared=args.prepared.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    audit=json.loads((prepared/'audit.json').read_text());rows=json.loads((prepared/'rows.json').read_text())
    full_train=audit.get('scope')=='full_crossdocked_train'
    if full_train and (audit.get('source_split')!='train' or audit.get('benchmark_test_derived_rows')!=0
                       or any(r.get('source_split')!='train' for r in rows)):
        raise ValueError('Full-train fitting refuses non-train or unaudited rows')
    arrays=np.load(prepared/'arrays.npz');x=arrays['x'];y=arrays['y'];groups=arrays['groups']
    if len(set(groups))<10:raise ValueError('Insufficient independent development groups')
    config={'created_at_kst':now(),'prepared':str(prepared),'prepared_audit':audit,
            'training_device':'cuda:0','physical_gpu':args.physical_gpu,'gpu_name':device.name,'gpu_uuid':str(device.uuid),
            'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'seeds':SEEDS,'learning_rates':LRS,
            'max_epochs':args.max_epochs,'outer_folds':5,'inner_validation_group_fraction':.2,
            'weight_decay':.01,'early_stopping':'Validation loss every 5 updates, minimum 50 epochs, patience 12 checks',
            'source_sha256':sha(__file__),'artifact_kind':'offline_router_ablation_v1',
            'scope_warning':audit['source_warning'],'production_compatibility':'Not a legacy LearnedRouter artifact; no deployment performed',
            'C_definition':'Feasibility BCE on all recorded actions; utility regression conditional on feasible measured outputs; observed cost regression. Score=p_feasible*max(0,predicted_delta)-lambda*cost, forced-action ties prefer p_feasible. This changes loss masking and scoring and is a plug-in gain heuristic, not an exact expected-improvement distribution.'}
    save(out/'study.json',config);shutil.copytree(prepared,out/'prepared')
    shutil.copy2(__file__,out/'training_script_snapshot.py')
    status={'status':'training','started_at_kst':now(),'updated_at_kst':now(),'completed_folds':0,
            'total_folds':5,'completed_fits':0,'planned_cv_fits':5*len(MODELS)*len(SEEDS)*len(LRS),'gpu':config['gpu_uuid']}
    save(out/'status.json',status);emit(event='start',**status)
    n=len(rows);oof={m:np.full((n,2),np.nan) for m in MODELS};oof_prob=np.full(n,np.nan)
    fold_results=[];fits=0;all_candidates=defaultdict(list)
    splitter=GroupKFold(n_splits=5)
    for fold,(development,test) in enumerate(splitter.split(x,groups=groups)):
        split=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=20261026+fold)
        t,v=next(split.split(x[development],groups=groups[development]));train=development[t];val=development[v]
        verify_split(rows,train,val,test)
        fold_dir=out/f'fold_{fold}';fold_dir.mkdir()
        split_meta={k:grouped_metadata(rows,ids) for k,ids in [('train',train),('validation',val),('test',test)]}
        split_meta.update(train_rows=len(train),validation_rows=len(val),test_rows=len(test));save(fold_dir/'split.json',split_meta)
        for variant in MODELS:
            features=x[:,:95] if variant=='A' else x
            candidates=[]
            for lr in LRS:
                artifacts=[];preds=[];probs=[]
                for seed in SEEDS:
                    model,pred,prob=fit(features,y,train,val,variant,lr,seed+fold,args.max_epochs)
                    artifacts.append(model);preds.append(pred);probs.append(prob);fits+=1
                    emit(event='fit',fold=fold,model=variant,lr=lr,seed=seed,epoch=model['best_epoch'],
                         validation_loss=model['best_validation_loss'],training_device=model['training_device'])
                    status.update(completed_fits=fits,updated_at_kst=now());save(out/'status.json',status)
                candidates.append({'lr':lr,'mean_validation_loss':float(np.mean([a['best_validation_loss'] for a in artifacts])),
                                   'artifacts':artifacts,'prediction':np.mean(preds,axis=0),
                                   'probability':np.mean(probs,axis=0) if variant=='C' else None})
            best=min(candidates,key=lambda c:c['mean_validation_loss']);prediction=best['prediction']
            oof[variant][test]=prediction[test]
            if variant=='C':oof_prob[test]=best['probability'][test]
            feature_names=audit['baseline_features']+([] if variant=='A' else audit['spatial_features'])
            bundle={'artifact_kind':'offline_router_ablation_v1','variant':variant,'feature_names':feature_names,
                    'generation_protocol_id':audit['generation_protocol_id'],'objective_sha256':audit['objective_sha256'],
                    'supported_action_kinds':audit.get('supported_action_kinds',['REDESIGN_REGION']),
                    'training_groups':split_meta['train']['group_id'],'validation_groups':split_meta['validation']['group_id'],
                    'test_groups':split_meta['test']['group_id'],'members':best['artifacts'],'deployment_status':'offline_only'}
            save(fold_dir/f'{variant}.json',bundle)
            for art in best['artifacts']:all_candidates[variant].append(art)
            mask=masks_for(y,variant)
            metrics={}
            for j,name in enumerate(['utility','cost']):
                kept=test[mask[test,j]]
                metrics[name+'_mae']=float(np.mean(np.abs(prediction[kept,j]-y[kept,j]))) if len(kept) else None
            if variant=='C':
                metrics['feasibility_brier']=float(np.mean((best['probability'][test]-y[test,2])**2))
            record={'fold':fold,'model':variant,'chosen_lr':best['lr'],'epochs':[a['best_epoch'] for a in best['artifacts']],
                    'mean_validation_loss':best['mean_validation_loss'],**metrics}
            fold_results.append(record);emit(event='fold_model_complete',**record)
        status.update(completed_folds=fold+1,updated_at_kst=now());save(out/'status.json',status)
    if any(not np.isfinite(p).all() for p in oof.values()) or not np.isfinite(oof_prob).all():
        raise AssertionError('Missing OOF predictions')
    # Serialize honest out-of-fold predictions separately from all-data final fits.
    predictions=[]
    for i,row in enumerate(rows):
        predictions.append({'row_id':i,'state_id':row['state_id'],'group_id':row['group_id'],
                            'target_id':row['target_id'],'action_id':row['action_id'],
                            'models':{m:{'utility':float(oof[m][i,0]),'wall_seconds':float(oof[m][i,1]),
                                         'feasibility':float(oof_prob[i]) if m=='C' else None} for m in MODELS}})
    save(out/'oof_predictions.json',predictions);save(out/'fold_results.json',fold_results)
    summaries={};decisions={}
    for variant in MODELS:
        prob=oof_prob if variant=='C' else np.full(n,np.nan)
        forced=outcomes(rows,oof[variant],prob,variant,0.,False,average_seeds=full_train)
        decisions[variant]=forced
        model_summary={'forced_one_action':aggregate(forced),'stop_policies':{}}
        for lam in LAMBDAS:
            d=outcomes(rows,oof[variant],prob,variant,lam,True,average_seeds=full_train)
            model_summary['stop_policies'][str(lam)]=aggregate(d)
            save(out/f'decisions_{variant}_lambda_{lam}.json',d)
        if variant=='C':
            model_summary['feasibility_brier']=float(np.mean((oof_prob-y[:,2])**2))
            model_summary['feasibility_auroc']=float(roc_auc_score(y[:,2],oof_prob))
        summaries[variant]=model_summary;save(out/f'decisions_{variant}_forced.json',forced)
    ci={}
    for current,control in [('B','A'),('C','B'),('C','A')]:
        ci[f'{current}-{control}']={field:paired_ci(decisions[current],decisions[control],field)
                                  for field in ['selected_feasible','best_feasible_gain','observed_regret']}
    # Full-data refits are usable development scorers, not the models used above.
    final=out/'final_models';final.mkdir()
    for variant in MODELS:
        artifacts=all_candidates[variant];lr=Counter(a['lr'] for a in artifacts).most_common(1)[0][0]
        epochs=int(np.median([a['best_epoch'] for a in artifacts if a['lr']==lr]));members=[]
        features=x[:,:95] if variant=='A' else x
        for seed in SEEDS:
            art,pred,prob=fit(features,y,np.arange(n),np.array([],dtype=int),variant,lr,seed,
                              args.max_epochs,forced_epochs=max(5,epochs))
            check_reg,check_prob=predict_artifact(art,features)
            if not np.allclose(check_reg,pred,rtol=1e-4,atol=2e-5):raise AssertionError('NumPy export mismatch')
            if variant=='C' and not np.allclose(check_prob,prob,atol=2e-5):raise AssertionError('Probability export mismatch')
            members.append(art)
        save(final/f'{variant}.json',{'artifact_kind':'offline_router_ablation_v1','variant':variant,
             'feature_names':audit['baseline_features']+([] if variant=='A' else audit['spatial_features']),
             'generation_protocol_id':audit['generation_protocol_id'],'objective_sha256':audit['objective_sha256'],
             'supported_action_kinds':audit.get('supported_action_kinds',['REDESIGN_REGION']),
             'members':members,'refit_rows':n,'refit_groups':len(set(groups)),
             'scope_warning':audit['source_warning'],'deployment_status':'offline_only_not_legacy_router_compatible',
             'not_used_for_reported_oof_metrics':True})
    torch.cuda.synchronize()
    summary={'finished_at_kst':now(),'models':summaries,'paired_group_bootstrap_95':ci,
             'reference_baselines':random_oracle(rows,average_seeds=full_train),'rows':n,'groups':len(set(groups)),
             'states':audit['states'],'utility_labels':audit['known_utility_rows'],'feasible_labels':audit['feasible_rows'],
             'training_device':'cuda:0','physical_gpu':args.physical_gpu,'gpu_name':device.name,'gpu_uuid':str(device.uuid),
             'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
             'source_warning':audit['source_warning'],'selection_evaluation':('Offline action choice with observed seed-mean outcomes; not online rollout or official benchmark test.' if full_train else
             'Offline replay of observed, mostly single-seed candidate menus; not online rollout or official benchmark test.'),
             'incumbent_policy':'For originally feasible states, reject unusable/nonimproving children and retain parent. Missing utility is never imputed as a training label.',
             'lambda_protocol':'Predeclared sensitivity values, not test-tuned optimal weights. Forced candidate comparison uses lambda=0.',
             'limitations':[audit['source_warning'],
                            'Target/series/exact parent graph grouping, not sequence-homology clustering.',
                            'Observed finite-seed outcomes; observed oracle is not the true expectation.',
                            'C changes both feasibility conditioning and scoring; not an isolated BCE-head effect.',
                            'Production worker/policy integration is not modified or deployed.']}
    save(out/'summary.json',summary)
    status.update(status='complete',updated_at_kst=now(),finished_at_kst=now(),summary=str(out/'summary.json'),
                  final_models=str(final),peak_cuda_allocated_bytes=summary['peak_cuda_allocated_bytes'])
    save(out/'status.json',status);emit(event='complete',**status)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--physical-gpu',type=int,required=True)
    p.add_argument('--max-epochs',type=int,default=500);args=p.parse_args()
    if args.max_epochs<50:p.error('At least 50 epochs required by this fixed protocol')
    run(args)
