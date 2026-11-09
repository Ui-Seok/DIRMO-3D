"""Build the A/B/C development dataset exclusively from saved pre-action inputs."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from dirmo.chemistry.graphs import read_sdf
from dirmo.evaluation.pocket import protein_atoms
from dirmo.routing.features import FEATURE_NAMES
from dirmo.routing.spatial import SPATIAL_FEATURE_NAMES,region_spatial_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def prepare(dataset,output):
    output.mkdir(parents=True,exist_ok=False)
    raw=[json.loads(l) for l in dataset.read_text().splitlines() if l.strip()]
    if not raw or any(r['feature_names']!=FEATURE_NAMES for r in raw):
        raise ValueError('A baseline requires the current, exact 95-feature schema')
    for key in ['generation_kernel_sha256','objective_sha256']:
        values={r.get(key) for r in raw}
        if len(values)!=1 or None in values:
            raise ValueError(f'Expected one declared nonmissing {key}')
    if any(r['action']['kind']!='REDESIGN_REGION' for r in raw):
        raise ValueError('This first ablation isolates the saved REDESIGN_REGION action menu')
    contexts={};metadata=[];x=[];labels=[];source_splits=Counter()
    for index,row in enumerate(raw):
        snapshot_path=Path(row['snapshot_path'])
        if str(snapshot_path) not in contexts:
            state=json.loads(snapshot_path.read_text());directory=snapshot_path.parent
            manifest=json.loads((directory/'manifest.json').read_text());case=manifest['case']
            sdf=Path(state['work_item']['sdf_path']);pocket=directory/'inputs/fixed_pocket.pdb'
            mol=read_sdf(sdf);protein=protein_atoms(str(pocket))
            context={'state':state,'case':case,'mol':mol,'protein':protein,'sdf':str(sdf),
                     'pocket':str(pocket),'sdf_sha256':sha(sdf),'pocket_sha256':sha(pocket),
                     'parent_graph_id':state['work_item']['graph_id']}
            contexts[str(snapshot_path)]=context
            source_splits['benchmark_test_derived' if '/test/' in case['initial_ligand_path'] else 'other']+=1
        ctx=contexts[str(snapshot_path)]
        maps=row['action']['region']
        spatial=region_spatial_features(ctx['mol'],*ctx['protein'],maps,
                                       ctx['case']['protected']['atom_maps'])
        before=row['observation_before'];after=row['resulting_work_item'].get('observation') or {}
        utility=row.get('utility_change');seconds=row['compute_cost']['wall_seconds']
        if not np.isfinite(seconds) or seconds<0:
            raise ValueError('Missing/negative measured execution cost')
        # A missing score never becomes an invented affinity/utility target.
        label=[np.nan if utility is None else utility,seconds,float(row['feasible'])]
        parent_metrics=before.get('metrics',{});child_metrics=after.get('metrics',{})
        item={'row_id':index,'state_id':row['pre_state_id'],'target_id':row['target_id'],
              'series_id':row['series_id'],'parent_graph_id':ctx['parent_graph_id'],
              'action_id':row['action']['action_id'],'region':maps,'seed':row['seed'],
              'feasible':bool(row['feasible']),'parent_feasible':bool(before['feasible']),
              'parent_utility':before.get('utility'),'child_utility':after.get('utility'),
              'utility_delta':utility,'wall_seconds':seconds,
              'generation_status':row['generation_result']['status'],
              'failure_reason':row['generation_result'].get('reason'),
              'failures':after.get('failures',[]),'parent_metrics':parent_metrics,'child_metrics':child_metrics,
              'metric_deltas':{k:child_metrics[k]-parent_metrics[k] for k in ['vina','qed','sa']
                               if k in parent_metrics and k in child_metrics},
              'snapshot_path':str(snapshot_path),'pre_sdf':ctx['sdf'],'pre_pocket':ctx['pocket'],
              'sdf_sha256':ctx['sdf_sha256'],'pocket_sha256':ctx['pocket_sha256']}
        metadata.append(item);x.append(row['pre_action_features']+spatial);labels.append(label)
    # Unite target, series, exact parent graph and snapshot; nothing from one parent
    # can cross train/validation/test, including a parent bound to multiple targets.
    parent=list(range(len(metadata)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,row in enumerate(metadata):
        for name in ['target_id','series_id','parent_graph_id','state_id']:
            key=(name,row[name])
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    component={}
    for i,row in enumerate(metadata):
        root=find(i)
        if root not in component:component[root]=f'group_{len(component):03d}'
        row['group_id']=component[root]
    matrix=np.asarray(x,dtype=np.float32);targets=np.asarray(labels,dtype=np.float32)
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite pre-action features')
    old,new=defaultdict(set),defaultdict(set)
    for row,a,b in zip(metadata,matrix[:,:len(FEATURE_NAMES)],matrix):
        old[(row['state_id'],tuple(a))].add(tuple(row['region']))
        new[(row['state_id'],tuple(b))].add(tuple(row['region']))
    audit={'created_at_kst':now(),'dataset':str(dataset.resolve()),'dataset_sha256':sha(dataset),
           'rows':len(metadata),'states':len({r['state_id'] for r in metadata}),
           'targets':len({r['target_id'] for r in metadata}),'groups':len(component),
           'source_split':dict(source_splits),'source_warning':'Test-derived historical data are used for exploratory development only, not official benchmark performance.',
           'generation_protocol_id':raw[0]['generation_kernel_sha256'],
           'objective_sha256':raw[0]['objective_sha256'],'steps':sorted({r['integration_steps'] for r in raw}),
           'baseline_features':FEATURE_NAMES,'spatial_features':SPATIAL_FEATURE_NAMES,
           'model_feature_count':matrix.shape[1],
           'known_utility_rows':int(np.isfinite(targets[:,0]).sum()),'feasible_rows':int(targets[:,2].sum()),
           'baseline_collision_states':len({s for (s,_),v in old.items() if len(v)>1}),
           'spatial_collision_states':len({s for (s,_),v in new.items() if len(v)>1}),
           'feature_sources':'Saved pre-action parent SDF, fixed pocket PDB, region and manifest protection; no generated product features or ProLIF outcome features.',
           'preparation_script_sha256':sha(__file__),
           'spatial_source_sha256':sha(Path(__file__).resolve().parents[1]/'src/dirmo/routing/spatial.py')}
    np.savez_compressed(output/'arrays.npz',x=matrix,y=targets,
                        groups=np.array([r['group_id'] for r in metadata]))
    (output/'rows.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.dataset,a.output)
