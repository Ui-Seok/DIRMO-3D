import importlib.util
from pathlib import Path

import numpy as np
import unittest
from rdkit import Chem

from dirmo.routing.spatial import region_spatial_features,SPATIAL_FEATURE_NAMES


def parent():
    m=Chem.MolFromSmiles('CCC');c=Chem.Conformer(3);c.Set3D(True)
    for i,xyz in enumerate([(0.,0.,0.),(1.5,0.,0.),(2.8,.6,0.)]):
        c.SetAtomPosition(i,xyz);m.GetAtomWithIdx(i).SetAtomMapNum(i+1)
    m.AddConformer(c)
    return m


def features(m,px,region):
    return region_spatial_features(m,px,['N','O'],['A:LYS:1','A:ASP:2'],region,[2])


def test_spatial_features_are_rigid_transform_invariant():
    m=parent();px=np.array([[.1,2.8,1.],[.8,-2.2,1.4]])
    original=features(m,px,[1]);rot=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    translation=np.array([13.,-7.,4.])
    transformed=Chem.Mol(m);xyz=m.GetConformer().GetPositions()@rot.T+translation
    for i,p in enumerate(xyz):transformed.GetConformer().SetAtomPosition(i,p)
    assert np.allclose(original,features(transformed,px@rot.T+translation,[1]),atol=1e-6)


def test_atom_map_numbers_are_not_numerical_predictors():
    m=parent();px=np.array([[.1,2.8,1.],[.8,-2.2,1.4]])
    x=features(m,px,[1])
    for a in m.GetAtoms():a.SetAtomMapNum(a.GetAtomMapNum()*10)
    y=region_spatial_features(m,px,['N','O'],['A:LYS:1','A:ASP:2'],[10],[20])
    assert np.allclose(x,y)


def test_distinct_regions_can_be_distinguished_by_environment():
    m=parent();px=np.array([[.1,2.8,1.],[.8,-2.2,1.4]])
    a=features(m,px,[1]);b=features(m,px,[3])
    assert len(a)==len(SPATIAL_FEATURE_NAMES)
    assert not np.allclose(a,b)
    with unittest.TestCase().assertRaises(ValueError):features(m,px,[99])


def runner():
    path=Path(__file__).resolve().parents[1]/'scripts/train_router_ablation.py'
    spec=importlib.util.spec_from_file_location('ablation_test',path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def test_conditional_gain_does_not_prefer_invalid_negative_action():
    mod=runner();rows=[]
    for i in range(2):
        rows.append(dict(state_id='s',target_id='t',group_id='g',action_id=str(i),
                         parent_feasible=True,feasible=True,utility_delta=.1,wall_seconds=1.,metric_deltas={}))
    pred=np.array([[-.1,1.],[-.01,1.]])
    result=mod.outcomes(rows,pred,np.array([.9,.1]),'C',0.,False)
    assert result[0]['selected_row']==0
    assert mod.outcomes(rows,pred,np.array([.9,.1]),'C',0.,True)[0]['stopped']


def test_split_rejects_shared_parent_even_when_targets_differ():
    mod=runner();rows=[]
    for i in range(3):
        rows.append(dict(state_id=str(i),target_id=str(i),series_id=str(i),parent_graph_id='same' if i<2 else 'other',group_id=str(i)))
    with unittest.TestCase().assertRaises(AssertionError):mod.verify_split(rows,[0],[1],[2])


def load_tests(loader, tests, pattern):
    functions=[v for k,v in globals().items() if k.startswith("test_") and callable(v)]
    return unittest.TestSuite(unittest.FunctionTestCase(f) for f in functions)


if __name__=="__main__":
    unittest.main()
