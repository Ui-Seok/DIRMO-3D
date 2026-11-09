import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/run_crossdocked_train_router.py'
    spec=importlib.util.spec_from_file_location('full_train_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def name(receptor, ligand):
    return f'CLUSTER/{receptor}_A_rec_{ligand}_ABC_lig_tt_min_0_pocket10.pdb_CLUSTER/{receptor}_A_rec_{ligand}_ABC_lig_tt_min_0.sdf'


class FullTrainPreparation(unittest.TestCase):
    def setup_data(self, root, train, val, test):
        mod=module();processed=root/'processed';processed.mkdir();raw=root/'raw';raw.mkdir()
        for split,values in [('train',train),('val',val),('test',test)]:
            np.savez(processed/f'{split}.npz', names=np.array(values))
        for value in train:
            for member in mod.source_pair(value):
                path=raw/member;path.parent.mkdir(exist_ok=True,parents=True);path.write_text('placeholder')
        return mod,SimpleNamespace(output=root/'run',processed=processed,raw=raw,minimum_disk_gib=15)

    def test_entire_train_index_and_val_overlap_are_preserved_and_audited(self):
        with TemporaryDirectory() as temp:
            train=[name('1aaa','2aaa'),name('3aaa','4aaa'),name('5aaa','6aaa')]
            mod,args=self.setup_data(Path(temp),train,[train[1]],[name('7aaa','8aaa')])
            study=mod.prepare(args)
            self.assertEqual(study['source_entries'],3)
            self.assertEqual(study['existing_val_entries_also_in_train'],1)
            self.assertEqual(study['source_test_pairs_overlap'],0)
            self.assertEqual(len((args.output/'cases.jsonl').read_text().splitlines()),3)

    def test_test_pair_is_refused_before_manifest_collection(self):
        with TemporaryDirectory() as temp:
            value=name('1aaa','2aaa')
            mod,args=self.setup_data(Path(temp),[value],[],[value])
            with self.assertRaisesRegex(ValueError,'Train/test source-pair overlap'):
                mod.prepare(args)

    def test_shared_test_pdb_is_refused_even_for_different_pairs(self):
        with TemporaryDirectory() as temp:
            mod,args=self.setup_data(Path(temp),[name('1aaa','2aaa')],[],[name('1aaa','3aaa')])
            with self.assertRaisesRegex(ValueError,'PDB overlap'):
                mod.prepare(args)

    def test_replay_averages_seeds_instead_of_choosing_lucky_seed(self):
        path=Path(__file__).resolve().parents[1]/'scripts/train_router_ablation.py'
        spec=importlib.util.spec_from_file_location('train_repeat_test',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        rows=[]
        for action,gains in [('a',[0.,1.]),('b',[.6,.6])]:
            for gain in gains:
                rows.append(dict(state_id='s',target_id='t',group_id='g',action_id=action,
                                 parent_feasible=True,feasible=True,utility_delta=gain,
                                 wall_seconds=1.,metric_deltas={}))
        prediction=np.array([[.7,1.],[.7,1.],[.6,1.],[.6,1.]])
        result=mod.outcomes(rows,prediction,np.full(4,np.nan),'B',0.,False,average_seeds=True)[0]
        self.assertEqual(result['selected_action'],'a')
        self.assertAlmostEqual(result['best_feasible_gain'],.5)
        self.assertAlmostEqual(result['observed_oracle_gain'],.6)
        random=mod.random_oracle(rows,average_seeds=True)
        self.assertAlmostEqual(random['random_expected_best_feasible_gain'],.55)
        self.assertAlmostEqual(random['observed_oracle_best_feasible_gain'],.6)


if __name__=='__main__':
    unittest.main()
