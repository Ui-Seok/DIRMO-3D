"""Score prepared region candidates with an A/B/C ensemble; no outcome labels used."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from train_router_ablation import predict_artifact


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cost-weight',type=float,default=0.)
    args=p.parse_args()
    if args.cost_weight<0:p.error('cost-weight must be nonnegative')
    model=json.loads(args.model.read_text());audit=json.loads((args.prepared/'audit.json').read_text())
    if model['artifact_kind']!='offline_router_ablation_v1':raise ValueError('Artifact schema mismatch')
    if model['generation_protocol_id']!=audit['generation_protocol_id'] or model['objective_sha256']!=audit['objective_sha256']:
        raise ValueError('Generation protocol/objective mismatch')
    names=audit['baseline_features']+([] if model['variant']=='A' else audit['spatial_features'])
    if model['feature_names']!=names:raise ValueError('Feature schema mismatch')
    # Only pre-action feature arrays and candidate identity are read for inference.
    x=np.load(args.prepared/'arrays.npz')['x'][:,:len(names)]
    raw=json.loads((args.prepared/'rows.json').read_text())
    candidates=[{k:r[k] for k in ['row_id','state_id','action_id','region']} for r in raw]
    def action_kind(row):
        identity=row['action_id']
        if identity.startswith('redesign:'):return 'REDESIGN_REGION'
        if identity.startswith('edit:'):return 'CHEMICAL_EDIT'
        if identity=='pose_resample':return 'POSE_RESAMPLE'
        if identity=='change_steps':return 'CHANGE_GENERATION_CONDITION'
        raise ValueError('Unknown candidate action')
    supported=set(model.get('supported_action_kinds',['REDESIGN_REGION']))
    if any(action_kind(r) not in supported for r in candidates):
        raise ValueError('Candidate action was not covered by this artifact')
    prediction=[predict_artifact(member,x) for member in model['members']]
    mean=np.mean([r[0] for r in prediction],axis=0)
    probability=np.mean([r[1] for r in prediction],axis=0) if model['variant']=='C' else None
    states=defaultdict(list)
    for i,c in enumerate(candidates):
        delta,cost=map(float,mean[i]);pf=float(probability[i]) if probability is not None else None
        score=(pf*max(0.,delta) if pf is not None else delta)-args.cost_weight*max(0.,cost)
        states[c['state_id']].append({**c,'predicted_utility_delta':delta,'predicted_seconds':cost,
                                     'predicted_feasibility':pf,'score':score})
    result=[]
    for state,options in sorted(states.items()):
        options.sort(key=lambda r:(r['score'],r['predicted_feasibility'] or 0.),reverse=True)
        result.append({'state_id':state,'selected_action':options[0]['action_id'] if options[0]['score']>0 else 'STOP',
                       'candidates':options})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'model':str(args.model),'inference_backend':'numpy',
                                      'cost_weight':args.cost_weight,'states':result},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'states':len(result),'output':str(args.output),'model':model['variant']}))


if __name__=='__main__':main()
