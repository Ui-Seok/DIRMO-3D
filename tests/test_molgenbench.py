"""Scientific-contract checks for the external evaluation adapter."""
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from rdkit import Chem

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from molgenbench_study import menu, prescore
from molgenbench_report import payoff,mean_complete
from molgenbench_report import core_drift
from molgenbench_data import fragments
from router_contract_pilot import mapped,save
from dirmo.evaluation.edited_graph import graph_from_record
from dirmo.chemistry.graphs import protected_graph
from dirmo.schemas import Protection

def test_scaffold_chemistry_and_coordinate_exception():
    parent=mapped(Chem.MolFromSmiles('CCOc1ccccc1'))
    rules={'CC*':[dict(rhs='CN*',witnesses=[{'target':'DEV'}],witness_count=1)]}
    actions,core,mandatory,audit=menu(parent,'c1ccccc1',rules,20261105)
    assert actions and audit[0]['status']=='selected'
    for action in actions:
        assert not set(action['site_parent_maps'])&set(core)
        assert set(action['relaxed_free_maps'])-set(action['strict_free_maps'])=={action['attachment_map']}
        protected_graph(parent,graph_from_record(action['graph']),Protection(atom_maps=core))
    assert set(mandatory)<=set(core)

def test_failures_are_not_affinity_imputation():
    assert payoff({'generation_status':'initialization_failed','refined':None})==0
    assert payoff({'generation_status':'ok','refined':None}) is None
    assert payoff({'generation_status':'ok','refined':{'joint_physical':None}}) is None
    assert payoff({'generation_status':'ok','refined':{'joint_physical':False}})==0
    assert mean_complete([1.,None]) is None
    assert mean_complete([1.,0.])==.5

def test_fragment_retained_core_is_large():
    for core,fragment in fragments('CCOc1ccccc1'):
        assert Chem.MolFromSmiles(fragment).GetNumHeavyAtoms()<=3
        assert Chem.MolFromSmiles(core).GetNumHeavyAtoms()>=6

def test_core_coordinates_do_not_require_output_atom_maps():
    parent=mapped(Chem.MolFromSmiles('CCO'))
    conf=Chem.Conformer(3)
    for i in range(3):conf.SetAtomPosition(i,(float(i),0.,0.))
    parent.AddConformer(conf)
    actual=Chem.Mol(parent)
    for atom in actual.GetAtoms():atom.SetAtomMapNum(0)
    actual.GetConformer().SetAtomPosition(0,(.2,0.,0.))
    assert core_drift(parent,parent,actual,[1,2])['mcs_max_drift']==pytest.approx(.2)
    changed=Chem.RWMol(actual);changed.ReplaceAtom(0,Chem.Atom('N'))
    with pytest.raises(ValueError):core_drift(parent,parent,changed.GetMol(),[1,2])

def test_frozen_feature_and_prediction_parity(tmp_path):
    """New external scorer must reproduce actual trained feature ordering."""
    source=ROOT/'runs/router_contract_gpu1_100_20261105_v1'
    if not (source/'router_training/data/arrays.npz').exists():pytest.skip('Historical local artifact unavailable')
    from router_contract_pilot import read,sha256_file
    from train_contract_router import predict
    meta=read(source/'router_training/data/rows.json');cid=meta[0]['case_id']
    case=read(source/'cases'/cid/'case.json')
    case['actions']=[a for a in case['actions'] if a['action_kind']=='CHEMICAL_EDIT']
    save(tmp_path/'cases'/cid/'case.json',case)
    save(tmp_path/'evaluations'/cid/'initial_metrics.json',read(source/'evaluations'/cid/'initial_metrics.json'))
    shutil.copy2(source/'router_training/router.json',tmp_path/'frozen_router.json')
    save(tmp_path/'study.json',dict(cases=[cid],modes=['strict','relaxed'],router_sha256=sha256_file(tmp_path/'frozen_router.json')))
    decision=prescore(tmp_path)[0]
    models=read(tmp_path/'frozen_router.json')['models']
    x=np.load(source/'router_training/data/arrays.npz')['x']
    for option in decision['options']:
        idx=next(i for i,r in enumerate(meta) if r['case_id']==cid and [r['action'],r['mode']]==option['key'])
        expected=np.mean([predict(m,x[idx:idx+1])[0] for m in models],axis=0)
        np.testing.assert_allclose(option['prediction'],expected,rtol=1e-5,atol=1e-6)
