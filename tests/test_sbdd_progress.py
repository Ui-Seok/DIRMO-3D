import json
import os
from pathlib import Path
import tempfile
import unittest

from dirmo.evaluation.sbdd_progress import collect_progress, kst_now, save_progress


class SBDDProgressTests(unittest.TestCase):
    def test_partial_draws_advance_without_a_finished_pocket(self):
        plan = dict(cases=[dict(case_id='a'), dict(case_id='b')], settings=dict(draws=2), scheduled_draws=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'cases/a/generated/000'
            folder.mkdir(parents=True)
            (folder / 'result.json').write_text('{}')
            (folder / 'evaluation.json').write_text('{}')
            result = collect_progress(root, plan, dict(pid=os.getpid(), status='running'))
            self.assertEqual(result['evaluated_cases'], 0)
            self.assertEqual(result['evaluated_draws'], 1)
            self.assertEqual(result['evaluation_percent'], 25.)
            self.assertTrue(result['updated_kst'].endswith('+09:00'))
            self.assertTrue(result['latest_evaluation_kst'].endswith('+09:00'))
            folder = root / 'cases/a/generated/001'
            folder.mkdir(parents=True)
            (folder / 'evaluation.json').write_text('{}')
            (root / 'cases/a/evaluation_complete.json').write_text('{}')
            later = collect_progress(root, plan, dict(pid=os.getpid(), status='running'))
            self.assertEqual(later['evaluated_cases'], 1)
            self.assertEqual(later['evaluated_draws'], 2)

    def test_observer_does_not_overwrite_controller_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status = root / 'status.json'
            status.write_text('{"evaluated_cases": 10}')
            save_progress(root / 'progress.json', dict(updated_kst=kst_now(), evaluated_draws=1630))
            self.assertEqual(json.loads(status.read_text()), dict(evaluated_cases=10))
            self.assertEqual(json.loads((root / 'progress.json').read_text())['evaluated_draws'], 1630)
            self.assertFalse(list(root.glob('*.tmp')))


if __name__ == '__main__':
    unittest.main()
