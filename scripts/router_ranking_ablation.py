"""Validation-only paired ranking-loss experiment; no test inference or FM calls."""
import argparse
from collections import defaultdict
import itertools
from pathlib import Path
import time
import os
import numpy as np
from router_contract_pilot import ROOT, read, save, now, sha256_file
from train_contract_router import predict, diagnostics, policies

SEEDS=[20261105,20261106,20261107]
LAMBDAS=[0.,.1,.3]


def soft_pairs(rows,y,train):
    """Exact 3-with-replacement bootstrap means; not nine independent labels."""
    actions=defaultdict(lambda:defaultdict(list))
    for i in train:
        r=rows[i];actions[r['case_id']][r['action'],r['mode']].append(int(i))
    pairs=[];audit=[]
    draws=np.array(list(itertools.product(range(3),repeat=3)))
    for cid,menu in sorted(actions.items()):
        candidates=[]
        for key,ii in sorted(menu.items()):
            if len(ii)!=3 or not np.isfinite(y[ii,2]).all():continue
            v=y[ii,2].astype(float);bs=v[draws].mean(1)
            candidates.append((key,ii[0],v,bs))
        local=[]
        for a,b in itertools.combinations(candidates,2):
            diff=a[3][:,None]-b[3][None,:]
            q=float(np.mean(diff>1e-9)+.5*np.mean(abs(diff)<=1e-9))
            confidence=2*abs(q-.5)
            if confidence<1e-8:continue
            local.append([a[1],b[1],q,confidence])
        total=sum(p[3] for p in local)
        for p in local:p[3]/=total
        pairs.extend(local)
        audit.append(dict(case_id=cid,complete_actions=len(candidates),informative_pairs=len(local),
            note='Bootstrap probability is a heuristic soft target from 3 repeats, not independent evidence or a calibrated confidence level'))
    if not pairs:raise ValueError('No informative training pairs')
    return np.asarray(pairs,dtype=float),audit


def fit(x,y,w,tr,va,pairs,lam,seed):
    import torch
    from torch import nn
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    xm=x[tr].mean(0);xs=x[tr].std(0);xs[xs<1e-6]=1.
    ym=np.zeros(4,dtype=np.float32);ys=np.ones(4,dtype=np.float32)
    for j in range(1,4):
        v=y[tr,j];v=v[np.isfinite(v)];ym[j]=v.mean();ys[j]=max(v.std(),.05)
    z=(y-ym)/ys
    tx=torch.tensor((x-xm)/xs,device='cuda:0');ty=torch.tensor(np.nan_to_num(z),device='cuda:0')
    tm=torch.tensor(np.isfinite(z),device='cuda:0');tw=torch.tensor(w,device='cuda:0')
    ti=torch.tensor(tr,device='cuda:0');vi=torch.tensor(va,device='cuda:0')
    ia=torch.tensor(pairs[:,0],dtype=torch.long,device='cuda:0');ib=torch.tensor(pairs[:,1],dtype=torch.long,device='cuda:0')
    tq=torch.tensor(pairs[:,2],dtype=torch.float32,device='cuda:0');pw=torch.tensor(pairs[:,3],dtype=torch.float32,device='cuda:0')
    model=nn.Sequential(nn.Linear(x.shape[1],128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,4)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    def base_loss(p,ids):
        terms=[]
        for j in range(4):
            mask=tm[ids,j]
            if not mask.any():continue
            v=(nn.functional.binary_cross_entropy_with_logits(p[mask,j],ty[ids[mask],j],reduction='none') if j==0 else
               nn.functional.smooth_l1_loss(p[mask,j],ty[ids[mask],j],reduction='none'))
            wt=tw[ids[mask]];terms.append((v*wt).sum()/wt.sum())
        return torch.stack(terms).mean()
    best=float('inf');stale=0;history=[];best_state=None;epoch_best=None
    for epoch in range(1,201):
        model.train();opt.zero_grad(set_to_none=True)
        base=base_loss(model(tx[ti]),ti)
        rank=base.new_tensor(0.)
        if lam:
            logits=model(tx[ia])[:,2]-model(tx[ib])[:,2]
            rank=(nn.functional.binary_cross_entropy_with_logits(logits,tq,reduction='none')*pw).sum()/pw.sum()
        loss=base+lam*rank
        assert torch.isfinite(loss)
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();model.eval()
        with torch.inference_mode():vl=float(base_loss(model(tx[vi]),vi).item())
        history.append(dict(epoch=epoch,base=float(base.item()),rank=float(rank.item()),validation_common_base=vl))
        if vl<best-1e-5:
            best=vl;epoch_best=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:stale+=1
        if epoch>=40 and stale>=25:break
    model.load_state_dict(best_state)
    artifact=dict(x_mean=xm.tolist(),x_scale=xs.tolist(),y_mean=ym.tolist(),y_scale=ys.tolist(),
        layers=[dict(weight=l.weight.detach().cpu().tolist(),bias=l.bias.detach().cpu().tolist()) for l in model if isinstance(l,nn.Linear)],
        rank_lambda=lam,seed=seed,best_epoch=epoch_best,validation_common_base_loss=best,history=history)
    return artifact


def paired(left,right):
    l={r['case_id']:r for r in left['rows']};r={x['case_id']:x for x in right['rows']}
    assert set(l)==set(r)
    ids=sorted(l);d=np.array([l[i]['learned']-r[i]['learned'] for i in ids]);groups=sorted({l[i]['group'] for i in ids})
    rng=np.random.default_rng(20261105);boot=[]
    for _ in range(5000):
        gg=rng.choice(groups,len(groups),replace=True)
        idx=[k for g in gg for k,i in enumerate(ids) if l[i]['group']==g]
        boot.append(float(d[idx].mean()))
    return dict(mean=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
        wins=int((d>1e-12).sum()),ties=int((abs(d)<=1e-12).sum()),losses=int((d< -1e-12).sum()))


def uff_control(source,allowed,out):
    """Paired original-prior controls only, train/validation roots; no test access."""
    observations=[]
    for cid in sorted(allowed):
        parent=read(source/'evaluations'/cid/'initial_metrics.json')
        for r in read(source/'evaluations'/cid/'rows.json'):
            if r.get('uff_only') is None:continue
            fm=r.get('refined');uff=r['uff_only']
            obs=dict(case_id=cid,action=r['action'],kind=r['action_kind'],mode=r['mode'],draw=r['draw'],
                fm_available=fm is not None,uff_available=True)
            for name,m in [('fm',fm),('uff',uff)]:
                obs[name+'_physical']=m.get('joint_physical') if m else None
                gain=m.get('vina_improvement_vs_initial') if m else None
                phys=obs[name+'_physical']
                obs[name+'_gain']=(max(0.,gain) if phys and gain is not None else 0. if phys is False else None) if parent['pb'].get('passes') else None
                obs[name+'_vina']=m['vina'].get('score') if m else None
            observations.append(obs)
    summaries={}
    for mode in ('all','strict','relaxed'):
        rows=[r for r in observations if mode=='all' or r['mode']==mode]
        phys=[r for r in rows if r['fm_physical'] is not None and r['uff_physical'] is not None]
        gains=[r for r in rows if r['fm_gain'] is not None and r['uff_gain'] is not None]
        roots=sorted({r['case_id'] for r in gains})
        deltas=np.array([np.mean([r['fm_gain']-r['uff_gain'] for r in gains if r['case_id']==cid]) for cid in roots])
        rng=np.random.default_rng(20261105)
        ci=np.quantile(rng.choice(deltas,(5000,len(deltas)),replace=True).mean(1),[.025,.975]).tolist() if len(deltas)>=3 else None
        summaries[mode]=dict(paired_draws=len(phys),paired_gain_draws=len(gains),roots=len(roots),
            fm_physical=float(np.mean([r['fm_physical'] for r in phys])) if phys else None,
            uff_physical=float(np.mean([r['uff_physical'] for r in phys])) if phys else None,
            root_mean_fm_minus_uff_gain=float(deltas.mean()) if len(deltas) else None,root_bootstrap_ci95=ci)
    save(out/'uff_paired_controls.json',dict(time_kst=now(),summary=summaries,rows=observations,
        note='Train/validation only, predetermined every-fifth-root first-edit and pose controls. Identical input conformer and mask; observed paired outputs only. No checkpoint/native-mode generalization.'))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--gpu',default='3');args=ap.parse_args()
    source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    from dirmo.gpu import inventory,resolve_device,worker_environment
    dev=resolve_device(args.gpu,inventory());os.environ.update(worker_environment(dev,str(ROOT/'src')))
    import torch
    torch.set_num_threads(1);assert torch.cuda.device_count()==1
    status=dict(status='running',time_kst=now(),pid=os.getpid(),gpu=args.gpu,gpu_uuid=dev['uuid'])
    save(out/'status.json',status)
    try:
        data=source/'router_training/data';meta=read(data/'rows.json');arrays=np.load(data/'arrays.npz')
        # Load stored array for transport; immediately discard test rows before any training/scoring/statistics.
        keep=np.array([i for i,r in enumerate(meta) if r['split'] in ('train','validation')])
        x=arrays['x'][keep];y=arrays['y'][keep];w=arrays['weights'][keep];rows=[meta[i] for i in keep]
        del arrays,meta
        tr=np.array([i for i,r in enumerate(rows) if r['split']=='train']);va=np.array([i for i,r in enumerate(rows) if r['split']=='validation'])
        assert len({rows[i]['case_id'] for i in tr})==70 and len({rows[i]['case_id'] for i in va})==15
        assert not {rows[i]['group'] for i in tr}&{rows[i]['group'] for i in va}
        pairs,paudit=soft_pairs(rows,y,tr)
        assert set(pairs[:,:2].astype(int).ravel())<=set(tr)
        save(out/'pair_audit.json',dict(pairs=len(pairs),roots=paudit,train_rows=len(tr),validation_rows=len(va),test_rows_used=0))
        config=dict(time_kst=now(),gpu=args.gpu,gpu_uuid=dev['uuid'],seeds=SEEDS,rank_lambdas=LAMBDAS,
            primary_lambda=.1,sensitivity_lambda=.3,widths=[128,64],lr=.0003,weight_decay=.01,max_epochs=200,patience=25,
            checkpoints='Both arms use identical common base validation loss; no incomparable augmented loss or test labels for checkpoint selection',
            rank_loss='Pairwise BCE on standardized accepted-gain score differences; soft q from exact independent within-action 3-draw bootstrap means; weight 2*abs(q-.5), normalize within root; exact uncertain ties omitted',
            scope='Only loss addition varies; no QED/SA or STOP gate changes, no refit on validation, no new FM/LLM calls, no prior test metrics/predictions accessed',
            expansion_gate='Primary lambda .1 only: validation mean gain > baseline and QED rule and random; paired group bootstrap lower bound >0 vs baseline; at least 2/3 matched seeds beat baseline; physical rate >= baseline minus .05. Lambda .3 sensitivity cannot authorize expansion.',
            next_if_gate_passes='Freeze primary design, collect new 50-root heldout excluding previous 100 family/source/scaffold/parent groups before confirmatory evaluation. Validation alone is not proof.',
            source_arrays_sha256=sha256_file(data/'arrays.npz'),split_sha256=sha256_file(source/'split.json'),script_sha256=sha256_file(__file__))
        save(out/'design.json',config)
        (out/'code.py').write_text(Path(__file__).read_text())
        uff_control(source,{r['case_id'] for r in rows},out)
        results={};start=time.monotonic()
        for lam in LAMBDAS:
            models=[];perseed=[]
            for seed in SEEDS:
                m=fit(x,y,w,tr,va,pairs,lam,seed);models.append(m)
                save(out/'models'/f'lambda_{lam}_seed_{seed}.json',m)
                pp=np.zeros((len(rows),4));pp[va]=predict(m,x[va]);perseed.append(policies(rows,pp,'validation'))
                save(out/'status.json',{**status,'time_kst':now(),'lambda':lam,'seed':seed,'completed_models':len(results)*3+len(models),'planned_models':9})
            pp=np.zeros((len(rows),4));pp[va]=np.mean([predict(m,x[va]) for m in models],axis=0)
            np.save(out/f'validation_predictions_lambda_{lam}.npy',pp[va])
            results[str(lam)]=dict(policy=policies(rows,pp,'validation'),heads=diagnostics(y,pp,va),
                seeds=[dict(seed=s,epoch=m['best_epoch'],policy=p) for s,m,p in zip(SEEDS,models,perseed)])
        baseline=results['0.0'];comparisons={}
        for lam in LAMBDAS[1:]:
            r=results[str(lam)];comp=paired(r['policy'],baseline['policy'])
            comp['seed_wins']=sum(a['policy']['means']['learned']>b['policy']['means']['learned'] for a,b in zip(r['seeds'],baseline['seeds']))
            comparisons[str(lam)]=comp
        r=results['0.1']['policy']['means'];b=baseline['policy']['means'];c=comparisons['0.1']
        gate=bool(r['learned']>max(b['learned'],r['random'],r['property_rule']) and c['ci95'][0]>0 and c['seed_wins']>=2 and r['learned_physical']>=b['learned_physical']-.05)
        save(out/'results.json',dict(time_kst=now(),results=results,comparisons=comparisons,expansion_gate_passed=gate,
            elapsed_seconds=time.monotonic()-start,actual_gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),peak_memory_bytes=torch.cuda.max_memory_allocated(),
            note='Validation ablation, not new test evidence. No test predictions produced. No automatic selection of favorable sensitivity setting.'))
        save(out/'status.json',{**status,'status':'complete','time_kst':now(),'models':9,'expansion_gate_passed':gate})
        print({'status':'complete','expansion_gate_passed':gate,'comparisons':comparisons},flush=True)
    except BaseException as exc:
        save(out/'status.json',{**status,'status':'failed','time_kst':now(),'reason':f'{type(exc).__name__}: {exc}'});raise


if __name__=='__main__':main()
